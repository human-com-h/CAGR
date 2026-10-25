"""Bounded deterministic second-query construction: no labels or LLM call."""
import time
from v04_features import CAPS,STOP,TERM,terms

def expansion_query(question,answer,documents,variant='question_answer'):
    start=time.perf_counter();q=question[:CAPS['max_question_chars']];a=answer[:CAPS['max_answer_chars']]
    covered=set()
    for d in documents[:CAPS['max_documents']]:
        n=CAPS['max_document_chars'];covered.update(terms(d.get('title','')[:n]+' '+d['text'][:n],n,CAPS['max_document_terms']))
    missing=[w for w in TERM.findall(q)[:CAPS['max_question_terms']] if w[:1].isupper() and w.lower() not in STOP and w.lower() not in covered]
    if variant=='question_answer':raw=q+' '+a
    elif variant=='answer_entities':raw=a+' '+' '.join(missing)+' '+q
    else:raise ValueError('Unknown deterministic query variant')
    ordered=list(dict.fromkeys(t.lower() for t in TERM.findall(raw) if t.lower() not in STOP))[:24]
    query=' '.join(ordered)
    return query,{'kind':'query_expansion','variant':variant,'ms':1000*(time.perf_counter()-start),'status':'ok','uncovered_capitalized_tokens':missing,
        'query_terms':ordered,'LLM_calls':0,'scope':'bounded CPU work; all previous Pilot/answers/retrieval still charged'}
