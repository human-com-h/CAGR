"""Paid T4 numerical gate, independent Torch greedy/topk oracle, no QA labels."""
import math,time
from v04_features import torch_step_signals
from v04_v3_inference import K,V

CASES=['random','tie_cross','all_tie','negative','fp16_close','early_eos','unaligned','lowest_index_zero','lowest_index_last','signed_zero']

def numeric_gate(g):
    start=time.perf_counter();torch=g.torch;torch.manual_seed(42);checks=[];eos=min(g.eos)
    for mode in ['A','B']:
        for count in range(1,6):
            for case in CASES:
                width=V+1 if case=='unaligned' else V
                backing=torch.randn((count,width),device=g.config['device'],dtype=torch.float16)
                x=backing[:,1:] if case=='unaligned' else backing
                x[:,17]=10;x[:,V-1]=9.75
                if case=='tie_cross':x[:,18991]=10;x[:,18992]=10;x[:,V-1]=10
                elif case=='all_tie':x.fill_(2.)
                elif case=='negative':x.copy_(-x.abs()-1);x[:,17]=-0.125;x[:,V-1]=-0.25
                elif case=='fp16_close':x[:,17]=100.;x[:,V-1]=99.9375
                elif case=='early_eos':x[count-1,eos]=20
                elif case=='lowest_index_zero':x[:,0]=10
                elif case=='lowest_index_last':x[:,V-1]=20
                elif case=='signed_zero':x.fill_(-0.);x[:,17]=0.
                reference=[torch_step_signals(x[i:i+1]) for i in range(count)]
                expected=[int(x[i].argmax().item()) for i in range(count)]
                limit=5 if case=='early_eos' else count
                for i in range(count):
                    token,pending,raw=g.pool.fused_step(x[i:i+1],i+1,limit,mode)
                    if token!=expected[i] or int(pending.item())!=expected[i]:raise RuntimeError('Fused greedy lowest-index Token mismatch: '+case)
                    a,b,z,s,e,_=raw
                    if not a<=b<=z or e<s:raise RuntimeError('Invalid real causal signal interval')
                actual=g.pool.values(count);ref=[[r[2],r[1]] for r in reference]
                # IEEE +0/-0 have identical numerical semantics; real QA is
                # subsequently checked by the unchanged STRICT scientific SHA.
                if actual!=ref:raise RuntimeError('Frozen FP32 gap/exp numerical mismatch: '+case)
                checks.append({'mode':mode,'count':count,'case':case,'greedy_token_equal':True,'numeric_exact_equal':True,'shared_pending_equal':True})
        for case in ['nan','positive_inf','hidden_negative_inf','all_negative_inf','all_positive_inf','all_nan']:
            x=torch.zeros((1,V),device=g.config['device'],dtype=torch.float16);x[0,17]=10
            if case=='nan':x[0,123]=float('nan')
            elif case=='positive_inf':x[0,123]=float('inf')
            elif case=='hidden_negative_inf':x[0,123]=-float('inf')
            else:x.fill_(float('nan') if case=='all_nan' else float('inf') if case=='all_positive_inf' else -float('inf'))
            token,_,_=g.pool.fused_step(x,1,1,mode)
            if token<V:raise RuntimeError('Non-finite vocabulary entry escaped rejection: '+case)
            checks.append({'mode':mode,'case':case,'invalid_rejected':True})
        for n,limit in [(0,5),(6,5),(2,1),(1,6)]:
            try:g.pool.fused_step(torch.zeros((1,V),device=g.config['device'],dtype=torch.float16),n,limit,mode)
            except RuntimeError:pass
            else:raise RuntimeError('Invalid K5 window accepted')
            checks.append({'mode':mode,'n':n,'limit':limit,'invalid_window_rejected':True})
    if len(checks)!=120:raise RuntimeError('GPU numerical gate coverage changed')
    return {'checks':checks,'total':len(checks),'ms':1000*(time.perf_counter()-start),
        'scope':'synthetic device operator checks, not QA latency; IEEE zero signs numerically equal, real QA strict SHA unchanged'}
