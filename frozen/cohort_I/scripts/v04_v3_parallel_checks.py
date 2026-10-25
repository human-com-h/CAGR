"""Additional paid CUDA checks for the parallel helper, no new model inputs."""
import time
from v04_v3_benchmark import numeric_gate as original_numeric_gate
from v04_features import torch_step_signals
from v04_v3_inference import V

def parallel_numeric_gate(g):
    start=time.perf_counter();result=original_numeric_gate(g);torch=g.torch;extra=[]
    cases=[('cross_partition',(V-1,18991),(10.,9.75),False),
        ('boundary_tie',(18991,18992),(10.,10.),False),
        ('negative_maxima',(75967,75968),(-.5,-.515625),False),
        ('FP16_close_large',(0,V-1),(1000.,999.5),False),
        ('unaligned_scalar_copy',(V-1,18992),(10.,9.5),True)]
    for mode in ['A','B']:
        for name,indices,values,unaligned in cases:
            backing=torch.full((1,V+1 if unaligned else V),-20.,device=g.config['device'],dtype=torch.float16)
            logits=backing[:,1:] if unaligned else backing
            for j,value in zip(indices,values):logits[0,j]=value
            expected=torch_step_signals(logits);pending=torch.tensor([17],device=g.config['device'],dtype=torch.int64)
            token,_=g.pool.step(logits,pending,1,1,mode)
            if token!=17 or g.pool.values(1)!=[[expected[2],expected[1]]]:raise RuntimeError('Parallel GPU partition/FP16/unaligned equivalence failed: '+name)
            extra.append({'mode':mode,'case':name,'bitwise_equal':True})
        # Nonfinite error semantics are preserved. Deliberate invalid inputs
        # are paid technical checks; they are never model question outputs.
        for name,value in [('nan',float('nan')),('all_negative_infinity',float('-inf'))]:
            logits=torch.full((1,V),value,device=g.config['device'],dtype=torch.float16)
            try:torch_step_signals(logits)
            except ValueError:pass
            else:raise RuntimeError('Expected frozen reference nonfinite rejection')
            pending=torch.tensor([17],device=g.config['device'],dtype=torch.int64);g.pool.step(logits,pending,1,1,mode)
            try:g.pool.values(1)
            except RuntimeError:pass
            else:raise RuntimeError('Parallel helper failed to reject nonfinite input')
            extra.append({'mode':mode,'case':name,'both_rejected':True})
    result['parallel_partition_checks']=extra;result['ms']=1000*(time.perf_counter()-start)
    return result
