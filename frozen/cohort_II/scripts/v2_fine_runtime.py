"""Frozen fine_382 inference logic; no training, references, API or downloads."""
import math,time,re
from v04_runtime import SharedRuntime,BudgetExceeded
from v04_features import predict_linear
from route23_runtime import merge_second,bridge_query


def head_for(candidate,fold=None):return candidate['deployment'] if fold is None else candidate['fold_heads'][str(fold)]


def first_decision(candidate,features,fold=None):
    h=head_for(candidate,fold);value=features['targ_mean_exp_gap']
    if not math.isfinite(value):raise ValueError('Invalid frozen TARG signal')
    return value>h['threshold0']


def second_decision(candidate,features,dataset,fold=None):
    h=head_for(candidate,fold)
    if candidate['suppress_high_confidence_trivia'] and dataset=='triviaqa' and features['mean_logit_gap']>=h['trivia_confidence_threshold']:return 'stop'
    values={a:predict_linear(h['action_heads'][a],features) for a in ['in_situ','kv_bridge','rewrite']}
    a=max(values,key=values.get)
    return a if values[a]>h['threshold1'] else 'stop'


def bridge_is_bad(question,terms,tokens,eos):
    words=set(re.findall(r'\w+',question.lower()))
    return not terms or all(t in words for t in terms) or (tokens==8 and not eos)


def bridge_action(rt,question,answer,docs):
    g=rt.g;_,_,ids,_=g.bridge_spec();rt.permit(1+len(ids),8,reserve=64)
    words,_,call=rt.invoke(lambda:g.bridge_continue(8,False)[0],'kv_bridge')
    start=time.perf_counter();query,trace=bridge_query(question,answer,words,docs,g.bridge_options)
    bad=bridge_is_bad(question,trace['bridge_terms'],call['output_tokens'],call['stopped_on_eos'])
    trace['ms']=1000*(time.perf_counter()-start);trace.update(fine_bridge_rejected=bad,fallback='stop',probe_already_charged=True);rt.control.append(trace)
    return answer if bad else merge_second(rt,question,answer,docs,query)


def run_fine(question,dataset,g,retriever,config,budget,candidate,study,fold=None,binary=False,plan=None):
    rt=SharedRuntime(g,retriever,config,budget,study=study);answer='';g.retain_rag1=not binary
    try:
        state,f0=rt.pilot(config['generation']['answer_system'],'Question: '+question)
        take=plan['first'] if plan is not None else rt.timed('decision',lambda:first_decision(candidate,f0,fold))
        rt.control.append({'kind':'fine_stage0','selected':bool(take),'ms':0.,'status':'ok'})
        if not take:answer=rt.complete(state)[0]
        else:
            g.release(state,'discarded_pilot_for_new_rag_prompt');docs=rt.search(question)[:config['retrieval']['context_documents_per_round']]
            answer,f1,_=rt.generate(config['generation']['answer_system'],rt.prompt(question,docs),64,'rag1',signals=not binary and plan is None)
            if binary:action='stop'
            elif plan is not None:action=plan['action']
            else:
                values=rt.timed('observable_features',lambda:{**f1,'retrieval_score_variance':rt.retrievals[0]['score_variance']})
                action=rt.timed('decision',lambda:second_decision(candidate,values,dataset,fold))
            rt.control.append({'kind':'fine_stage1','action':action,'ms':0.,'status':'ok'})
            if action=='kv_bridge':answer=bridge_action(rt,question,answer,docs)
            else:
                g.drop_retained()
                if action!='stop':answer=rt.upgrade(question,answer,docs,'in_situ' if action=='in_situ' else 'evidence_gap')
        return rt.finish('matched_binary_v2' if binary else 'cagr_v2',answer)
    except BaseException as exc:
        g.abort_active();row=rt.finish('cagr_v2',answer,'budget_exhausted' if isinstance(exc,BudgetExceeded) else 'failed',type(exc).__name__);row['error']=str(exc);return row
    finally:g.retain_rag1=False;g.drop_retained()


def selected_plan(row):
    first=next(t['selected'] for t in row['control_trace'] if t['kind']=='fine_stage0')
    return {'first':first,'action':next((t['action'] for t in row['control_trace'] if t['kind']=='fine_stage1'),'stop')}


def science(row):
    return {'answer':row['answer'],'tier':row['tier'],'context_ids':row['context_ids'],
            'calls':[(c['kind'],c['input_tokens'],c['output_token_ids'],c.get('prompt_sha256')) for c in row['generation_calls']],
            'queries':[t['query'] for t in row['retrieval_trace']]}
