"""Explicit v0.4 feature semantics. NumPy is for local synthetic benchmarks only."""
import math,re,time
import numpy as np

GAP_SCHEMA='logit_gap_mean_v1'
STOP=set('a an the is are was were of in on at to for and or as by with from which what who when where how did does do has have had be been this that it its than name whose'.split())
TERM=re.compile(r'[A-Za-z0-9]+')
CAPS={'max_documents':5,'max_document_chars':1024,'max_document_terms':64,'max_question_chars':1024,'max_question_terms':64,'max_answer_chars':512,'max_answer_terms':64}
TEXT_NAMES=['answer_doc_overlap','question_doc_coverage','uncovered_capitalized_query_ratio','answer_in_best_doc','doc_answer_agreement']

def question_features(question):
    words=TERM.findall(question[:1024])[:64]
    return {'query_term_count':len(words),'query_capitalized_count':sum(w[:1].isupper() and w.lower() not in STOP for w in words)}

def numpy_step_signals(logits,mode='gap',top_k=64,winner=None,beta=3.):
    """O(V) maximum scans, no full-array copy for the gap path; no O(1) claim."""
    z=np.asarray(logits)
    if z.ndim!=1 or len(z)<2 or z.dtype.kind!='f':raise ValueError('Expected a floating vocabulary vector of at least two logits')
    if beta<=0:raise ValueError('beta must be positive')
    if mode=='gap':
        j=int(z.argmax()) if winner is None else int(winner)
        if j<0 or j>=len(z):raise ValueError('Greedy index outside vocabulary')
        runner=max(float(np.max(z[:j],initial=-np.inf)),float(np.max(z[j+1:],initial=-np.inf)))
        gap=float(z[j])-runner
        if not math.isfinite(gap) or gap<0:raise ValueError('Invalid logits or stale non-greedy index')
        return {'logit_gap':gap,'gap_exp_uncertainty':math.exp(-gap/beta)}
    if mode=='local_entropy':
        if not 2<=top_k<=len(z):raise ValueError('Invalid local entropy support size')
        selected=np.partition(z,len(z)-top_k)[-top_k:].astype(np.float32)
        selected.sort();gap=float(selected[-1]-selected[-2])
        q=selected-selected[-1];q=np.exp(q);q/=q.sum()
        entropy=float(-np.sum(q*np.log(q,where=q>0,out=np.zeros_like(q))))
        if not math.isfinite(entropy+gap):raise ValueError('Non-finite local signal')
        return {'local_entropy':entropy,'logit_gap':gap,'gap_exp_uncertainty':math.exp(-gap/beta),'support_k':top_k}
    if mode=='full_entropy_reference':
        p=z.astype(np.float32,copy=True);p-=p.max();p=np.exp(p);p/=p.sum()
        h=float(-np.sum(p*np.log(p,where=p>0,out=np.zeros_like(p))))
        top=np.partition(p,-2)[-2:]
        gap=np.partition(z,-2)[-2:];gap=float(gap[1]-gap[0])
        if not math.isfinite(h+gap):raise ValueError('Non-finite full signal')
        return {'full_entropy':h,'full_probability_margin':float(top[1]-top[0]),'logit_gap':gap}
    raise ValueError('Unknown feature mode')

def summarize_gap(gaps,beta=3.):
    if not gaps or any(not math.isfinite(x) or x<0 for x in gaps):raise ValueError('Invalid recorded logit gaps')
    mean=sum(gaps)/len(gaps)
    return {'feature_schema':GAP_SCHEMA,'mean_logit_gap':mean,'gap_uncertainty_of_mean':math.exp(-mean/beta),
            'targ_mean_exp_gap':sum(math.exp(-x/beta) for x in gaps)/len(gaps)}

def torch_step_signals(logits,mode='gap',top_k=64,beta=3.):
    """One top-k selection and one small materialization; greedy argmax stays separate for tie equivalence."""
    import torch
    k=2 if mode=='gap' else top_k
    values=torch.topk(logits,k,dim=-1,sorted=True).values.float()
    gap=values[...,0]-values[...,1]
    if mode=='gap':h=torch.zeros_like(gap)
    elif mode=='local_entropy':
        lp=values.log_softmax(-1);h=-(lp.exp()*lp).sum(-1)
    else:raise ValueError('Only explicit light modes are allowed')
    small=torch.stack((h,torch.exp(-gap/beta),gap),-1)
    result=small.reshape(-1,3)[0].tolist()
    if not all(math.isfinite(x) for x in result):raise ValueError('Non-finite lightweight feature')
    return tuple(result)

def terms(text,max_chars,max_terms,drop_stop=True):
    raw=TERM.findall(text[:max_chars].lower())[:max_terms]
    return {t for t in raw if not drop_stop or t not in STOP}

def bounded_text_features(question,answer,documents,caps=None):
    """O(D*L) bounded windows, NOT an unbounded O(1) text/NER algorithm."""
    start=time.perf_counter_ns();c={**CAPS,**(caps or {})}
    if set(c)!=set(CAPS) or any(not isinstance(v,int) or v<=0 for v in c.values()):raise ValueError('Invalid feature window budget')
    q=terms(question,c['max_question_chars'],c['max_question_terms']);a=terms(answer,c['max_answer_chars'],c['max_answer_terms'])
    entities=set()
    for w in TERM.findall(question[:c['max_question_chars']])[:c['max_question_terms']]:
        if w[:1].isupper() and w.lower() not in STOP:entities.add(w.lower())
    docs=[]
    for d in documents[:c['max_documents']]:
        # Bound slices BEFORE concatenation; a long raw passage must not be copied.
        n=c['max_document_chars'];prefix=d.get('title','')[:n]+' '+d['text'][:n]
        docs.append(terms(prefix,c['max_document_chars'],c['max_document_terms']))
    union=set().union(*docs) if docs else set()
    result={'answer_doc_overlap':len(a&union)/max(1,len(a)),
            'question_doc_coverage':len(q&union)/max(1,len(q)),
            'uncovered_capitalized_query_ratio':len(entities-union)/max(1,len(entities)),
            'answer_in_best_doc':max((len(a&d)/max(1,len(a)) for d in docs),default=0.),
            'doc_answer_agreement':sum(bool(a&d) for d in docs)/max(1,len(docs))}
    return result,{'kind':'bounded_text_features','ms':(time.perf_counter_ns()-start)/1e6,'documents':len(docs),
        'limits':c,'entity_proxy':'capitalized_query_tokens_not_NER','answer_terms':len(a),'query_terms':len(q)}

def predict_linear(head,values):
    names=head['feature_names']
    x=[float(values[k]) for k in names]
    if not all(math.isfinite(v) for v in x):raise ValueError('Missing/nonfinite online feature')
    if head.get('kind')=='bounded_forest':
        x=[float(np.float32(v)) for v in x]
        predictions=[]
        for tree in head['trees']:
            node=0
            while tree['left'][node]!=-1:
                node=tree['left'][node] if x[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
            predictions.append(tree['value'][node])
        return sum(predictions)/len(predictions)
    return head['coefficients'][0]+sum(w*(a-b)/s for w,a,b,s in zip(head['coefficients'][1:],x,head['mean'],head['scale']))

def utility_decision(head,features,lam,threshold=0.):
    if features.get('feature_schema')!=head['feature_schema']:raise ValueError('Feature schema differs; never reuse full-entropy coefficients for light statistics')
    gain=predict_linear(head['gain'],features);cost=max(0.,predict_linear(head['cost'],features))
    score=gain-lam*cost
    return score>threshold,{'predicted_gain':gain,'predicted_increment_ms':cost,'utility':score,'lambda':lam,'threshold':threshold}
