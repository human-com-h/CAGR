"""Frozen 120 numeric cases plus paid persistent-state safety checks on T4."""
import time
from v04_v3_argmax_checks import numeric_gate as frozen_numeric_gate
from v04_v3_inference import V

def numeric_gate(g):
    start=time.perf_counter();report=frozen_numeric_gate(g);t=g.torch;s=g.pool.state;checks=[]
    cases=[('FP32',lambda:t.zeros((1,V),device=g.config['device'],dtype=t.float32)),
        ('one_dimensional',lambda:t.zeros(V,device=g.config['device'],dtype=t.float16)),
        ('wrong_shape_same_numel',lambda:t.zeros((2,V//2),device=g.config['device'],dtype=t.float16)),
        ('CPU',lambda:t.zeros((1,V),dtype=t.float16))]
    for name,make in cases:
        try:s.step(make(),1,1,False,time.perf_counter_ns())
        except RuntimeError:checks.append({'case':name,'rejected':True})
        else:raise RuntimeError('Native state accepted unsafe logits: '+name)
    x=t.zeros((1,V),device=g.config['device'],dtype=t.float16)
    s.step(x,1,5,False,time.perf_counter_ns())
    try:s.step(x,3,5,False,time.perf_counter_ns())
    except RuntimeError:checks.append({'case':'nonsequential','rejected':True})
    else:raise RuntimeError('Native state accepted a missing observation')
    # First position resets native phase, including after failed validation.
    token=s.step(x,1,1,False,time.perf_counter_ns())
    if token!=0:raise RuntimeError('Native reset/tie-lowest-index mismatch')
    g.pool.boundary_view[0]=time.perf_counter_ns();batch=s.finalize(1,True)
    if len(batch['records'])!=1 or list(batch['values'][0])!=[0.,1.]:raise RuntimeError('Native phase materialization mismatch')
    report.update(native_state_safety_checks=checks,native_clock_epoch_check=g.pool.clock_epoch_check,
        native_state_safety_passed=True,ms=1000*(time.perf_counter()-start))
    return report
