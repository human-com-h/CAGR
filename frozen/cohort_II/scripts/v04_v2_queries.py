"""One fixed bounded bridge extractor and one compact rewrite, no labels/fit."""
import re,time
from v04_features import STOP,TERM

BRIDGE_LIMITS={'documents':5,'document_chars':1024,'sentences_per_document':6,'selected_sentences':2,
               'phrases_per_window':16,'bridge_phrases':2,'bridge_terms':6,'question_terms':12,'query_terms':24}
COMPACT={'documents':3,'tokens_per_document':96,'max_output_tokens':24,
         'system':'Write one short search query for the missing fact. Use evidence to resolve bridge entities. Return only the query.'}
ENTITY=re.compile(r"\b[A-Z][A-Za-z0-9'-]{1,}(?:[ \t]+(?:[A-Z][A-Za-z0-9'-]{1,}|of|the|and)){0,4}")
GENERIC=STOP|set('he she his her they their also born later after before during however although according early first last known new one two three'.split())

def bridge_query(question,answer,documents):
    started=time.perf_counter();qterms=list(dict.fromkeys(t.lower() for t in TERM.findall(question[:1024]) if t.lower() not in STOP))[:64];known=set(qterms);candidates={}
    def consider(text,score,source):
        for match in list(ENTITY.finditer(text))[:BRIDGE_LIMITS['phrases_per_window']]:
            words=TERM.findall(match.group());words=[w for w in words if w.lower() not in GENERIC]
            novel=[w for w in words if w.lower() not in known]
            if not novel:continue
            key=' '.join(w.lower() for w in words)
            item={'text':' '.join(words),'terms':[w.lower() for w in words],'score':score,'source':source}
            if key not in candidates or score>candidates[key]['score']:candidates[key]=item
    for rank,doc in enumerate(documents[:BRIDGE_LIMITS['documents']]):
        title=doc.get('title','')[:128];body=doc['text'][:BRIDGE_LIMITS['document_chars']]
        title_overlap=len(set(t.lower() for t in TERM.findall(title))&known)
        consider(title,2*title_overlap+1/(rank+1),f'title:{rank}')
        sentences=re.split(r'(?<=[.!?])\s+|\n+',body)[:BRIDGE_LIMITS['sentences_per_document']]
        ranked=sorted(enumerate(sentences),key=lambda x:(-len(set(t.lower() for t in TERM.findall(x[1]))&known),x[0]))[:BRIDGE_LIMITS['selected_sentences']]
        for i,text in ranked:
            overlap=len(set(t.lower() for t in TERM.findall(text))&known)
            consider(text,2*overlap+.5/(rank+1),f'sentence:{rank}:{i}')
    chosen=sorted(candidates.values(),key=lambda c:(-c['score'],c['source'],c['text']))[:BRIDGE_LIMITS['bridge_phrases']]
    terms=list(dict.fromkeys(t for c in chosen for t in c['terms']))[:BRIDGE_LIMITS['bridge_terms']]
    terms=list(dict.fromkeys(terms+qterms[:BRIDGE_LIMITS['question_terms']]))[:BRIDGE_LIMITS['query_terms']]
    query=' '.join(terms) or question[:1024]
    return query,{'kind':'query_expansion','variant':'doc_bridge_fixed_v1','status':'ok','ms':1000*(time.perf_counter()-started),
                  'bridge_candidates':len(candidates),'selected_bridge_phrases':chosen,'query_terms':terms,'limits':BRIDGE_LIMITS,'LLM_calls':0,
                  'uses_answer':False,'entity_proxy':'bounded_capitalized_phrases_not_NER','label_or_future_document_access':False}

def compact_prompt(question,answer,documents,tokenizer):
    evidence=[]
    for doc in documents[:COMPACT['documents']]:
        ids=tokenizer.encode(doc['text'],add_special_tokens=False)[:COMPACT['tokens_per_document']]
        evidence.append('['+doc['title']+'] '+tokenizer.decode(ids,skip_special_tokens=True))
    return ('Question: '+question+'\nInitial answer (may be wrong): '+answer+'\nEvidence:\n'+'\n'.join(evidence)+'\nMissing fact query:')
