"""Four charged actions, no old reference gate, shared frozen budgets and prompts."""
import re,time
from v04_runtime import SharedRuntime,BudgetExceeded
from v04_features import question_features,bounded_text_features,predict_linear,TERM,STOP
from v04_expansion import expansion_query

def six_features(pilot,question):
    import math
    gaps=pilot['token_logit_gaps'];e=[math.exp(-x/3.) for x in gaps];m=sum(e)/len(e)
    return {**pilot,**question_features(question),'targ_mean_exp_gap':m,'exp_gap_variance':sum((x-m)**2 for x in e)/len(e)}

def bridge_query(question,answer,keywords,docs,options):
    start=time.perf_counter();base,_=expansion_query(question,answer,docs,'question_answer');bridge=[t.lower() for t in TERM.findall(keywords[:128]) if t.lower() not in STOP][:options['max_bridge_terms']]
    answer_terms=[t.lower() for t in TERM.findall(answer[:512]) if t.lower() not in STOP][:4];out=[]
    for word in bridge+answer_terms+base.split():
        if word not in out:out.append(word)
        if len(out)==options['max_query_terms']:break
    return ' '.join(out),{'kind':'query_expansion','ms':1000*(time.perf_counter()-start),'status':'ok','variant':'KV_Bridge_then_answer_question','bridge_terms':bridge,'query_terms':len(out),'LLM_calls':'one_charged_cached_bridge'}

def stage0(candidate,features,fold=None):
    h=candidate['deployment'] if fold is None else candidate['OOF_heads'][str(fold)];c=candidate['configuration']
    score=predict_linear(h['gain'],features)+c['positive_weight']*max(0.,predict_linear(h['positive_gain'],features))-c['lambda0']*max(0.,predict_linear(h['cost'],features))
    return score>h['threshold0'],{'score':score,'threshold':h['threshold0']}

def stage1(candidate,features,fold=None):
    h=candidate['deployment'] if fold is None else candidate['OOF_heads'][str(fold)];c=candidate['configuration'];values={'stop':0.}
    for act in ['in_situ','kv_bridge','rewrite']:
        values[act]=predict_linear(h['action_heads'][act+'_gain'],features)-c['lambda1']*max(0.,predict_linear(h['action_heads'][act+'_cost'],features))-c['action_threshold']
    return max(values,key=values.get),{'utilities':values,'tie_order':list(values)}

def merge_second(rt,question,answer,docs,query):
    more=rt.search(query or question)[:rt.cfg['retrieval']['context_documents_per_round']];ids={d['doc_id'] for d in docs};merged=list(docs);new=[]
    for d in more:
        if d['doc_id'] not in ids:merged.append(d);ids.add(d['doc_id']);new.append(d['doc_id'])
    rt.control.append({'kind':'novelty_stop','ms':0.,'status':'ok','stopped':not bool(new),'new_document_ids':new})
    return rt.generate(rt.cfg['generation']['answer_system'],rt.prompt(question,merged),rt.b['max_answer_tokens'],'multihop_final')[0] if new else answer

def kv_upgrade(rt,question,answer,docs,keep_reference=False):
    g=rt.g;_,_,ids,_=g.bridge_spec();limit=g.bridge_options['max_new_tokens'];rt.permit(1+len(ids),limit,reserve=rt.b['max_answer_tokens'])
    holder={}
    def invoke():
        result,full=g.bridge_continue(limit,keep_reference);holder['reference']=full;return result
    words,_,call=rt.invoke(invoke,'kv_bridge');query,trace=bridge_query(question,answer,words,docs,g.bridge_options);rt.control.append(trace)
    final=merge_second(rt,question,answer,docs,query)
    return final,holder.get('reference'),call['output_token_ids']

def run_candidate(question,g,retriever,config,budget,candidate,study,fold=None,technical_plan=None,binary=False):
    rt=SharedRuntime(g,retriever,config,budget,study=study);answer='';g.retain_rag1=not binary
    try:
        state,f=rt.pilot(config['generation']['answer_system'],'Question: '+question)
        if technical_plan:first=technical_plan[0]
        else:
            f0=rt.timed('observable_features',lambda:six_features(f,question));first=rt.timed('decision',lambda:stage0(candidate,f0,fold))[0]
        rt.control.append({'kind':'route23_stage0','selected':first,'ms':0.,'status':'ok'})
        if not first:answer=rt.complete(state)[0]
        else:
            g.release(state,'discarded_pilot_for_new_rag_prompt');docs=rt.search(question)[:config['retrieval']['context_documents_per_round']]
            answer,f1,_=rt.generate(config['generation']['answer_system'],rt.prompt(question,docs),budget['max_answer_tokens'],'rag1',signals=not binary and technical_plan is None)
            def observable():return {**f1,'retrieval_score_variance':rt.retrievals[0]['score_variance'],**bounded_text_features(question,answer,docs)[0]}
            if binary:act='stop'
            elif technical_plan:act=technical_plan[1]
            else:
                f1values=rt.timed('observable_features',observable);act=rt.timed('decision',lambda:stage1(candidate,f1values,fold))[0]
            rt.control.append({'kind':'route23_stage1','action':act,'ms':0.,'status':'ok'})
            if act=='kv_bridge':answer,_,_=kv_upgrade(rt,question,answer,docs)
            else:
                g.drop_retained()
                if act!='stop':answer=rt.upgrade(question,answer,docs,'in_situ' if act=='in_situ' else 'evidence_gap')
        return rt.finish('matched_binary_v2' if binary else 'cagr_v2',answer)
    except BaseException as exc:
        g.abort_active();row=rt.finish('cagr_v2',answer,'budget_exhausted' if isinstance(exc,BudgetExceeded) else 'failed',type(exc).__name__);row['error']=str(exc);return row
    finally:g.retain_rag1=False;g.drop_retained()

def collect_bridge(question,g,retriever,config,budget,study,keep_reference=False):
    rt=SharedRuntime(g,retriever,config,budget,study=study);answer='';g.retain_rag1=True;full=None;tokens=None
    try:
        state,pilot=rt.pilot(config['generation']['answer_system'],'Question: '+question);g.release(state,'discarded_pilot_for_new_rag_prompt')
        docs=rt.search(question)[:config['retrieval']['context_documents_per_round']];prompt=rt.prompt(question,docs)
        answer,h1,call=rt.generate(config['generation']['answer_system'],prompt,budget['max_answer_tokens'],'rag1',signals=True);preliminary=answer
        f0=rt.timed('observable_features',lambda:six_features(pilot,question));f1=rt.timed('observable_features',lambda:{**h1,'retrieval_score_variance':rt.retrievals[0]['score_variance'],**bounded_text_features(question,answer,docs)[0]})
        base_wall_ms=1000*(time.perf_counter()-rt.started)
        answer,full,tokens=kv_upgrade(rt,question,answer,docs,keep_reference);row=rt.finish('kv_bridge_potential',answer)
        row.update(scope='forced_potential_action_not_gate_quality',stage0_features=f0,stage1_features=f1,initial_answer=preliminary,initial_context_ids=[d['doc_id'] for d in docs],collection_H1_phase_ms=base_wall_ms)
        return row,{'initial_docs':docs,'initial_answer':preliminary,'reference_prefix':full,'bridge_tokens':tokens}
    except BaseException as exc:
        g.abort_active();row=rt.finish('kv_bridge_potential',answer,'failed',type(exc).__name__);row['error']=str(exc);return row,None
    finally:g.retain_rag1=False;g.drop_retained()

def collect_rewrite_control(question,g,retriever,config,budget,study):
    rt=SharedRuntime(g,retriever,config,budget,study=study)
    docs=rt.search(question)[:config['retrieval']['context_documents_per_round']]
    answer,_,_=rt.generate(config['generation']['answer_system'],rt.prompt(question,docs),budget['max_answer_tokens'],'rag1',signals=False)
    first_ms=1000*(time.perf_counter()-rt.started);answer=rt.upgrade(question,answer,docs)
    row=rt.finish('always_multihop',answer);row['collection_H1_phase_ms']=first_ms;return row

def collect_in_situ_increment(question,initial,g,retriever,config,budget,study):
    rt=SharedRuntime(g,retriever,config,budget,study=study);rt.tier=1;rt.context_ids=[d['doc_id'] for d in initial['initial_docs']]
    query,trace=expansion_query(question,initial['initial_answer'],initial['initial_docs'],'question_answer');rt.control.append(trace)
    answer=merge_second(rt,question,initial['initial_answer'],initial['initial_docs'],query);row=rt.finish('in_situ_H2_increment',answer)
    row.update(scope='new_H2_only_initial_H1_shared_not_policy_E2E',initial_context_ids=[d['doc_id'] for d in initial['initial_docs']]);return row

def validate_row(row,budget):
    if row['status']!='ok':raise RuntimeError('Recorded failed action: '+str(row.get('error')))
    calls=row['generation_calls']
    if row['cost']['total_input_tokens']>budget['max_total_input_tokens'] or row['cost']['total_output_tokens']>budget['max_total_output_tokens'] or row['cost']['retrieval_calls']>2:raise RuntimeError('Shared execution budget exceeded')
    for c in calls:
        if c['output_tokens']!=len(c['output_token_ids']) or c['input_tokens']>4096:raise RuntimeError('Generation counts/context mismatch')
        if c['kind']=='kv_bridge':
            if c['prefill_calls']!=0 or c['suffix_prefill_calls']!=1 or c['output_tokens']>8 or c['input_tokens']!=len(c['suffix_token_ids'])+1 or c['document_prefill_repeated']:raise RuntimeError('Bridge repeated document prefill/budget')
            initial=c['reused_context_tokens']
        else:
            if c['prefill_calls']!=1:raise RuntimeError('Ordinary path prefill count changed')
            initial=0
        for i,e in enumerate(c['forward_trace']):
            if e['cache_before']!=initial or e['cache_after']!=initial+e['input_tokens'] or (i>0 and e['input_tokens']!=1):raise RuntimeError('Cache append transition')
            initial=e['cache_after']
    return True

def path_identity(row):return {'contexts':row['context_ids'],'queries':[x['query'] for x in row['retrieval_trace']],
    'calls':[(x['kind'],x['input_tokens'],x['output_token_ids'],x.get('prompt_sha256')) for x in row['generation_calls']]}
