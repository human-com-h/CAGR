"""Two explicitly new upgrade arms; frozen v1 branches remain available."""
from v04_runtime import SharedRuntime,run_method as run_v1
from v04_features import question_features,utility_decision
from v04_v2_queries import bridge_query,compact_prompt,COMPACT

class RuntimeV2(SharedRuntime):
    def upgrade_v2(self,q,answer,docs,variant):
        if variant=='doc_bridge':
            query,trace=bridge_query(q,answer,docs);self.control.append(trace)
        elif variant=='compact':
            prompt=self.timed('context_tokenization',lambda:compact_prompt(q,answer,docs,self.g.tokenizer))
            query,_,_=self.generate(COMPACT['system'],prompt,COMPACT['max_output_tokens'],'query_rewrite')
        else:raise ValueError('Unknown independently registered v2 upgrade')
        more=self.search(query or q)[:self.cfg['retrieval']['context_documents_per_round']]
        ids={d['doc_id'] for d in docs};merged=list(docs);additions=[]
        for d in more:
            if d['doc_id'] not in ids:merged.append(d);additions.append(d['doc_id']);ids.add(d['doc_id'])
        self.control.append({'kind':'novelty_stop','ms':0.,'status':'ok','stopped':not bool(additions),'new_document_ids':additions})
        if not additions:return answer
        # New context ALWAYS gets a real charged new answer; no old quality reuse.
        return self.generate(self.cfg['generation']['answer_system'],self.prompt(q,merged),self.b['max_answer_tokens'],'multihop_final')[0]

def run_method(method,question,generator,retriever,config,budget,parameters,*,study=None,fold=None,max_retries=0,technical_plan=None):
    if method not in {'cagr_doc_bridge','cagr_compact'}:
        return run_v1(method,question,generator,retriever,config,budget,parameters,study=study,fold=fold,max_retries=max_retries,technical_plan=technical_plan)
    if technical_plan is not None:raise ValueError('New query arms are never forced scientific routes')
    rt=RuntimeV2(generator,retriever,config,budget,max_retries,study);answer='';gen=config['generation'];heads=parameters if fold is None else parameters['calibration_fold_heads'][str(fold)];routing=parameters['routing']
    try:
        state,features=rt.pilot(gen['answer_system'],'Question: '+question)
        features.update(rt.timed('observable_features',lambda:question_features(question)))
        take,decision=rt.timed('decision',lambda:utility_decision(heads['stage0'],features,routing['lambda0'],routing['threshold0']))
        rt.control.append({'kind':'decision_value','stage':0,'selected':bool(take),'decision':decision,'ms':0.,'status':'ok'})
        if not take:
            answer,_,_=rt.complete(state);return rt.finish(method,answer)
        generator.release(state,'discarded_pilot_for_new_rag_prompt');docs=rt.search(question)[:config['retrieval']['context_documents_per_round']]
        answer,features,_=rt.generate(gen['answer_system'],rt.prompt(question,docs),budget['max_answer_tokens'],'rag1',signals=True)
        values=rt.timed('observable_features',lambda:{**features,'retrieval_score_variance':rt.retrievals[-1]['score_variance']})
        take,decision=rt.timed('decision',lambda:utility_decision(heads['stage1'],values,routing['lambda1'],routing['threshold1']))
        rt.control.append({'kind':'decision_value','stage':1,'selected':bool(take),'decision':decision,'ms':0.,'status':'ok'})
        if take:answer=rt.upgrade_v2(question,answer,docs,'doc_bridge' if method=='cagr_doc_bridge' else 'compact')
        return rt.finish(method,answer)
    except Exception as exc:
        generator.abort_active();row=rt.finish(method,answer,'failed',type(exc).__name__);row['error']=str(exc);return row
