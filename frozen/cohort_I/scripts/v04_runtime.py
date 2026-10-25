"""Local-review framework. All methods share one budget and executed-cost ledger.

No CLI, downloads, credentials or cloud calls. Real inference is policy-blocked
in v04_inference until a later explicit execution authorization is recorded.
"""
import math,time
from execution_policy import guard_local_v04
from run_baselines import evidence_prompt
from run_upgrade_study import rewrite_prompt
from v04_features import GAP_SCHEMA,TEXT_NAMES,bounded_text_features,question_features,utility_decision
from v04_expansion import expansion_query

ENTAILMENT_SYSTEM='Decide whether the premise entails the hypothesis in the question context. Reply only yes or no.'

class BudgetExceeded(RuntimeError):pass


def semantic_clusters(answers,entails):
    """Deterministic first representative clustering; BOTH directions required.

    Unknown/invalid judgments keep samples apart. Missing NLI is an error;
    lexical equality is never substituted for semantic equivalence.
    """
    if entails is None:raise ValueError('A charged bidirectional entailment backend is required')
    groups=[]
    for answer in answers:
        for group in groups:
            forward=entails(answer,group[0]);backward=entails(group[0],answer)
            if forward is True and backward is True:
                group.append(answer);break
        else:groups.append([answer])
    return groups


def semantic_entropy(groups):
    if not groups or any(not g for g in groups):raise ValueError('Empty semantic sample set')
    n=sum(map(len,groups));mass=[len(g)/n for g in groups]
    return -sum(math.log(p) for p in mass)/len(mass)


class SharedRuntime:
    def __init__(self,generator,retriever,config,budget,max_retries=0,study=None):
        guard_local_v04()
        if max_retries<0:raise ValueError('Negative retry budget')
        if generator.config.get('repo_id')!='Qwen/Qwen2.5-3B-Instruct' or generator.revision!='aa8e72537993ba99e69dfaafa59ed015b17504d1':raise ValueError('All backends must use the frozen Qwen checkpoint')
        self.g=generator;self.r=retriever;self.cfg=config;self.b=budget;self.max_retries=max_retries
        self.study=study
        self.first=len(generator.calls);self.retrievals=[];self.control=[];self.retries=[]
        self.started=time.perf_counter();self.cost_unknown=False;self.tier=0;self.context_ids=[]
        self.final_input_reserve=0

    @property
    def calls(self):return self.g.calls[self.first:]

    def timed(self,kind,fn):
        start=time.perf_counter()
        try:
            result=fn();self.control.append({'kind':kind,'ms':1000*(time.perf_counter()-start),'status':'ok'});return result
        except Exception as exc:
            self.control.append({'kind':kind,'ms':1000*(time.perf_counter()-start),'status':'failed','error_type':type(exc).__name__});raise

    def token_count(self,system,user):
        text=self.g.tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
        return len(self.g.tokenizer.encode(text,add_special_tokens=False))

    def permit(self,n_input,n_output,reserve=0):
        if n_input>self.b['max_input_tokens_per_call']:raise BudgetExceeded('4096-token per-call context exceeded')
        if sum(c['input_tokens'] for c in self.calls)+n_input+(self.final_input_reserve if reserve else 0)>self.b['max_total_input_tokens']:raise BudgetExceeded('Common total input budget exhausted')
        if sum(c['output_tokens'] for c in self.calls)+n_output+reserve>self.b['max_total_output_tokens']:raise BudgetExceeded('Common total output budget exhausted; final answer reserve retained')

    def invoke(self,fn,kind):
        for attempt in range(self.max_retries+1):
            before=len(self.g.calls)
            try:
                out=fn()
                if len(self.g.calls)==before:
                    self.cost_unknown=True;raise RuntimeError('Backend did not append an executed-call ledger entry')
                for c in self.g.calls[before:]:c['attempt']=attempt;c['retry_of_kind']=kind if attempt else None
                if sum(c['output_tokens'] for c in self.calls)>self.b['max_total_output_tokens']:
                    raise RuntimeError('Backend violated the prechecked output cap')
                return out
            except BudgetExceeded:raise
            except Exception as exc:
                if len(self.g.calls)==before:self.cost_unknown=True
                for c in self.g.calls[before:]:c['attempt']=attempt;c['retry_of_kind']=kind if attempt else None
                self.retries.append({'kind':kind,'attempt':attempt,'error_type':type(exc).__name__,'will_retry':attempt<self.max_retries})
                self.g.abort_active()
                if attempt==self.max_retries:raise

    def generate(self,system,user,limit,kind,signals=False,reserve=0,sample=False):
        if limit<=0 or limit>self.b['max_answer_tokens']:raise BudgetExceeded('Invalid per-call output budget')
        if kind=='query_rewrite' and limit>self.b['max_rewrite_tokens']:raise BudgetExceeded('Rewrite budget exceeded')
        def call():
            n=self.timed('input_budget_tokenization',lambda:self.token_count(system,user));self.permit(n,limit,reserve)
            if sample:
                if not hasattr(self.g,'generate_sample'):raise ValueError('Sampling backend unavailable; no greedy-sampling substitution')
                return self.g.generate_sample(system,user,limit,kind,temperature=1.)
            return self.g.generate(system,user,limit,kind,collect_signals=signals)
        result=self.invoke(call,kind);result[2]['decoded_text']=result[0]
        return result

    def pilot(self,system,user):
        def call():
            n=self.timed('input_budget_tokenization',lambda:self.token_count(system,user))
            self.permit(n,self.b['max_answer_tokens'])
            return self.g.pilot(system,user,self.b['max_answer_tokens'],'pilot')
        return self.invoke(call,'pilot')

    def complete(self,state):
        self.permit(0,state['max_tokens']-state['call']['output_tokens'])
        # Same call object is updated: the Pilot is never appended/billed twice.
        return self.g.complete(state)

    def search(self,query):
        if len(self.retrievals)>=self.b['max_retrieval_calls']:raise BudgetExceeded('Common retrieval budget exhausted')
        start=time.perf_counter()
        try:
            docs,trace=self.r.search(query);trace['wall_ms']=1000*(time.perf_counter()-start);self.retrievals.append(trace)
            self.tier=len(self.retrievals);return docs
        except Exception as exc:
            self.retrievals.append({'query':query,'status':'failed','ms':1000*(time.perf_counter()-start),'error_type':type(exc).__name__});raise

    def prompt(self,q,docs):
        self.context_ids=[d['doc_id'] for d in docs]
        return self.timed('context_tokenization',lambda:evidence_prompt(q,docs,self.g,self.cfg['retrieval']))

    def upgrade(self,q,answer,docs,mode='evidence_gap',variant='question_answer'):
        if mode=='in_situ':
            query,trace=expansion_query(q,answer,docs,variant);self.control.append(trace)
        elif mode=='evidence_gap':
            if self.study is None:raise ValueError('The frozen evidence-gap rewrite study configuration is required')
            user=self.timed('context_tokenization',lambda:rewrite_prompt(q,{'answer':answer},docs,self.g,self.study,'evidence_gap'))
            query,_,_=self.generate(self.study['gap_rewrite_system'],user,self.b['max_rewrite_tokens'],'query_rewrite')
        else:raise ValueError('Unknown upgrade mode')
        more=self.search(query or q)[:self.cfg['retrieval']['context_documents_per_round']]
        self.retrievals[-1]['empty_rewrite_fallback']=not bool(query)
        ids={d['doc_id'] for d in docs};merged=list(docs);additions=[]
        for d in more:
            if d['doc_id'] not in ids:merged.append(d);additions.append(d['doc_id']);ids.add(d['doc_id'])
        self.control.append({'kind':'novelty_stop','ms':0.,'status':'ok','stopped':not bool(additions),'new_document_ids':additions})
        if not additions:return answer
        return self.generate(self.cfg['generation']['answer_system'],self.prompt(q,merged),self.b['max_answer_tokens'],'multihop_final')[0]

    def finish(self,method,answer,status='ok',error=None):
        calls=self.calls
        cost={'total_input_tokens':sum(c['input_tokens'] for c in calls),'total_output_tokens':sum(c['output_tokens'] for c in calls),
              'generation_calls':len(calls),'pilot_input_tokens':sum(c['input_tokens'] for c in calls if c['kind']=='pilot'),
              'pilot_output_tokens':sum(c.get('pilot_output_tokens',min(5,c['output_tokens'])) for c in calls if c['kind']=='pilot'),
              'retrieval_calls':len(self.retrievals),'signal_ms':sum(c.get('signal_ms',0.) for c in calls),
              'feature_materialization_ms':sum(c.get('feature_materialization_ms',0.) for c in calls),
              'control_ms':sum(t['ms'] for t in self.control),'generation_ms':sum(c.get('ms',0.) for c in calls),
              'retrieval_ms':sum(t['ms'] for t in self.retrievals),'end_to_end_ms':1000*(time.perf_counter()-self.started),
              'failed_generation_calls':sum(c['status']=='failed' for c in calls),'retry_calls':sum(c.get('attempt',0)>0 for c in calls),
              'failed_retrieval_calls':sum(t['status']=='failed' for t in self.retrievals),
              'sampling_ms':sum(e.get('sampling_ms',0.) for c in calls for e in c.get('forward_trace',[])),
              'entailment_judge_calls':sum(c['kind']=='entailment_judge' for c in calls),
              'untracked_backend_cost':self.cost_unknown,'asset_preparation_ms':None}
        cost['feature_decision_ms']=cost['signal_ms']+cost['feature_materialization_ms']+sum(t['ms'] for t in self.control if t['kind'] in {'observable_features','decision'})+sum(t.get('retrieval_feature_ms',0.) for t in self.retrievals)
        cost['feature_ms']=cost['feature_decision_ms']-sum(t['ms'] for t in self.control if t['kind']=='decision')
        cost['decision_ms']=sum(t['ms'] for t in self.control if t['kind']=='decision')
        cost['routing_extra_model_ms']=sum(c.get('ms',0.) for c in calls if c['kind'] in {'semantic_sample','entailment_judge'})
        cost['semantic_statistic_ms']=sum(t['ms'] for t in self.control if t['kind']=='semantic_clustering_statistic')
        cost['routing_total_ms']=cost['feature_decision_ms']+cost['routing_extra_model_ms']+cost['semantic_statistic_ms']
        cost['query_expansion_ms']=sum(t['ms'] for t in self.control if t['kind']=='query_expansion')
        cost['routing_total_ms']+=cost['query_expansion_ms']
        cost['failures']=int(status in {'failed','budget_exhausted'});cost['automatic_retries']=cost['retry_calls']
        cost['budget_limited_paths']=int(status in {'budget_exhausted','budget_limited_answer'})
        return {'method':method,'status':status,'error_type':error,'answer':answer,'tier':self.tier,
                'initial_tier':int(bool(self.retrievals)),'final_tier':self.tier,'escalated':len(self.retrievals)>1,'generation_calls':calls,
                'retrieval_trace':self.retrievals,'control_trace':self.control,'retry_trace':self.retries,'context_ids':self.context_ids,'cost':cost,
                'scope':getattr(self,'scope','development_online_comparison'),
                'budget':dict(self.b),'same_backbone_required':True,'simulation_or_measurement':'determined by injected backend; framework alone is not an experiment'}


def run_method(method,question,generator,retriever,config,budget,parameters,*,study=None,fold=None,max_retries=0,technical_plan=None):
    """Common interface; no label, reference answer, dataset or future evidence input."""
    rt=SharedRuntime(generator,retriever,config,budget,max_retries,study);gen=config['generation'];limit=budget['max_answer_tokens'];answer=''
    try:
        if technical_plan is not None:
            if method!='cagr' or len(technical_plan)!=2 or any(type(x)!=bool for x in technical_plan) or (not technical_plan[0] and technical_plan[1]):raise ValueError('Invalid separately labelled technical plan')
            rt.scope='forced_branch_or_timing_technical_check'
        cagr_methods={'cagr','cagr_in_situ','binary_gate'}
        if method in cagr_methods|{'targ_style'}:
            if method=='targ_style':
                threshold=parameters.get('threshold')
                if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('TARG threshold must be calibrated and explicit before generation')
            state,features=rt.pilot(gen['answer_system'],'Question: '+question)
            if features.get('feature_schema')!=GAP_SCHEMA:raise ValueError('Expected new gap schema, never v03 entropy coefficients')
            if technical_plan is None:features.update(rt.timed('observable_features',lambda:question_features(question)))
            heads=parameters
            if method in cagr_methods:
                if fold is not None:heads=parameters['calibration_fold_heads'][str(fold)]
                routing=parameters.get('routing',{})
                if technical_plan is not None:take=technical_plan[0];decision={'technical_replay':True,'ms':0.}
                else:take,decision=rt.timed('decision',lambda:utility_decision(heads['stage0'],features,routing.get('lambda0',parameters['lambda']),routing.get('threshold0',0.)))
                if parameters.get('first_retrieval_mode')=='conservative_always_rag':take=True;decision['conservative_first_retrieval']=True
            else:
                threshold=parameters['threshold']
                take=rt.timed('decision',lambda:features['targ_mean_exp_gap']>threshold);decision={'score':features['targ_mean_exp_gap'],'threshold':threshold,'beta':3.,'prefix_k':5}
            rt.control.append({'kind':'decision_value','stage':0,'selected':bool(take),'decision':decision,'ms':0.,'status':'ok'})
            if not take:
                answer,_,_=rt.complete(state);return rt.finish(method,answer)
            rt.g.release(state,'discarded_pilot_for_new_rag_prompt')
            docs=rt.search(question)[:config['retrieval']['context_documents_per_round']]
            answer,features,_=rt.generate(gen['answer_system'],rt.prompt(question,docs),limit,'rag1',signals=method in cagr_methods-{'binary_gate'} and technical_plan is None)
            if method in {'targ_style','binary_gate'}:return rt.finish(method,answer)
            def observable():
                values={**features,'retrieval_score_variance':rt.retrievals[-1]['score_variance']}
                if any(k in TEXT_NAMES for k in heads['stage1']['gain']['feature_names']):
                    values.update(bounded_text_features(question,answer,docs,parameters['text_feature_limits'])[0])
                return values
            if technical_plan is not None:take=technical_plan[1];decision={'technical_replay':True,'ms':0.}
            else:
                values=rt.timed('observable_features',observable)
                take,decision=rt.timed('decision',lambda:utility_decision(heads['stage1'],values,routing.get('lambda1',parameters['lambda']),routing.get('threshold1',0.)))
            rt.control.append({'kind':'decision_value','stage':1,'selected':bool(take),'decision':decision,'ms':0.,'status':'ok'})
            if take:
                answer=rt.upgrade(question,answer,docs,'in_situ' if method=='cagr_in_situ' else parameters.get('upgrade_mode','evidence_gap'),parameters.get('expansion_variant','question_answer'))
        elif method in {'direct','always_rag','always_multihop'}:
            docs=[]
            if method!='direct':docs=rt.search(question)[:config['retrieval']['context_documents_per_round']]
            answer,_,_=rt.generate(gen['answer_system'],rt.prompt(question,docs) if docs else 'Question: '+question,limit,'direct' if method=='direct' else 'rag1')
            if method=='always_multihop':
                answer=rt.upgrade(question,answer,docs)
        elif method=='sugar_l_style_budgeted':
            # This is the explicitly budget-adapted framework, not paper reproduction.
            low,high=parameters['low_threshold'],parameters['high_threshold']
            if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in [low,high]) or not 0<=low<high:raise ValueError('Calibrated ordered semantic thresholds required before generation')
            if not hasattr(generator,'generate_sample') or parameters.get('entailment_backend')!='same_backbone_bidirectional_judge':raise ValueError('Charged sampling/entailment backend required')
            docs=[];last_samples=[];stopped=False
            while True:
                prompt=rt.prompt(question,docs) if docs else 'Question: '+question
                rt.final_input_reserve=rt.timed('input_budget_tokenization',lambda:rt.token_count(gen['answer_system'],prompt))
                try:
                    samples=[rt.generate(gen['answer_system'],prompt,parameters['sample_max_tokens'],'semantic_sample',reserve=limit,sample=True)[0] for _ in range(parameters['samples'])]
                    last_samples=samples
                    def entails(a,b):
                        user='Question: '+question+'\nPremise: '+a+'\nHypothesis: '+b
                        text,_,_=rt.generate(ENTAILMENT_SYSTEM,user,parameters['entailment_max_tokens'],'entailment_judge',reserve=limit)
                        value=text.strip().lower()
                        return True if value=='yes' else False if value=='no' else None
                    groups=semantic_clusters(samples,entails)
                    entropy=rt.timed('semantic_clustering_statistic',lambda:semantic_entropy(groups))
                    rt.control.append({'kind':'semantic_feedback','ms':0.,'status':'ok','tier':rt.tier,'entropy':entropy,'clusters':[len(g) for g in groups],'samples':samples,'cluster_answers':groups})
                except BudgetExceeded:
                    stopped=True;break
                if rt.tier==0 and entropy<=low:break
                if rt.tier>=1 and (entropy<=high or rt.tier>=budget['max_retrieval_calls']):break
                # Medium initial uncertainty gets one retrieval, followed by paid reassessment.
                # High uncertainty can use a second round, within the SAME global budget.
                if rt.tier==0:query=question
                else:query,trace=expansion_query(question,last_samples[0] if last_samples else '',docs,'question_answer');rt.control.append(trace)
                more=rt.search(query)[:config['retrieval']['context_documents_per_round']];ids={d['doc_id'] for d in docs}
                docs=docs+[d for d in more if d['doc_id'] not in ids]
            answer,_,_=rt.generate(gen['answer_system'],rt.prompt(question,docs) if docs else 'Question: '+question,limit,'semantic_final')
            if stopped:return rt.finish(method,answer,'budget_limited_answer')
        else:raise ValueError('Unknown method')
        return rt.finish(method,answer)
    except Exception as exc:
        generator.abort_active()
        row=rt.finish(method,answer,'budget_exhausted' if isinstance(exc,BudgetExceeded) else 'failed',type(exc).__name__)
        row['error']=str(exc);return row
