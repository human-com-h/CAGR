"""Per-dispatch causal clock enclosure; unchanged compiled parallel signals.

No persistent CPU/GPU offset is assumed. The existing token read brackets
each GPU work interval, without another device read, event or synchronization.
"""
from pathlib import Path
import time
from v04_inference import LightweightGenerator
from v04_v3_inference import K,decode_packet,union_upper_ns
from v04_v3_parallel_inference import ParallelGenerator,ParallelSignalPool,load_parallel_native


def causal_union_ns(cpu_start,cpu_end,read_done,gpu_start,gpu_end):
    if not cpu_start<=cpu_end<=read_done or gpu_end<gpu_start:
        raise RuntimeError('Invalid per-dispatch causal timing order')
    # CPU dispatch starts before GPU begin; the common token read completes
    # after GPU finish. These inequalities hold for THIS dispatch, even when
    # the host/device offset has drifted since a previous question.
    bounds=(cpu_start-gpu_start,read_done-gpu_end)
    if bounds[0]>bounds[1]:
        raise RuntimeError('GPU work exceeds its causal host read enclosure')
    return union_upper_ns(cpu_start,cpu_end,gpu_start,gpu_end,bounds),bounds


class CausalSignalPool(ParallelSignalPool):
    def calibrate(self):
        # Paid diagnostics only. None of these offsets is used for production
        # work accounting; validity is checked independently at EVERY step.
        start=time.perf_counter();out=self.t.empty(1,dtype=self.t.int64,device=self.device);intervals=[]
        for _ in range(32):
            before=time.perf_counter_ns();self.native.stamp(out);stamp=int(out.item());after=time.perf_counter_ns()
            intervals.append((before-stamp,after-stamp))
        lower=max(a for a,b in intervals);upper=min(b for a,b in intervals)
        previous=self.bounds
        self.bounds=(min(a for a,b in intervals),max(b for a,b in intervals))
        record={'scope':'paid clock drift diagnostic; per-step causal enclosures govern acceptance',
            'offset_lower_ns':lower,'offset_upper_ns':upper,'common_offset_within_batch':lower<=upper,
            'envelope_ns':list(self.bounds),'previous_diagnostic_envelope_ns':list(previous) if previous else None,
            'previous_batch_disjoint':bool(previous and (self.bounds[0]>previous[1] or self.bounds[1]<previous[0])),
            'raw_offset_intervals_ns':[list(v) for v in intervals],'ms':1000*(time.perf_counter()-start)}
        self.clock_records.append(record)
        return record

    def step(self,logits,pending,n,limit,mode):
        if mode not in ['A','B'] or not 1<=n<=limit<=K:raise RuntimeError('Invalid bounded production signal step')
        cpu_start=time.perf_counter_ns()
        self.native.dispatch(logits,pending,self.window,self.pairs,self.partials,self.meta,self.eos,self.packet,
            self.mapped_features,self.mapped_starts,n,limit,mode=='A')
        cpu_end=time.perf_counter_ns()
        value=int(self.packet.item())  # The original shared greedy/EOS read.
        read_done=time.perf_counter_ns();gpu_start=int(self.starts_view[n-1]);token,gpu_end=decode_packet(value,gpu_start)
        ns,bounds=causal_union_ns(cpu_start,cpu_end,read_done,gpu_start,gpu_end)
        record={'cpu_start_ns':cpu_start,'cpu_end_ns':cpu_end,'shared_read_done_ns':read_done,
            'gpu_start_ns':gpu_start,'gpu_end_ns':gpu_end,'work_union_upper_ns':ns,
            'causal_offset_lower_ns':bounds[0],'causal_offset_upper_ns':bounds[1],
            'decode_cpu_ns':time.perf_counter_ns()-read_done,'shared_token_waits':1,'extra_device_syncs':0,
            'native_entry_calls':1,'native_kernel_launches':4 if mode=='A' else 3,
            'accounting':'per-dispatch CPU/GPU union upper; common token wait excluded from signal, retained in generation wall'}
        return token,record


class ClockSafeGenerator(ParallelGenerator):
    def __init__(self,*args,**kwargs):
        LightweightGenerator.__init__(self,*args,**kwargs)
        if self.signal_mode!='gap' or self.config['pilot_k']!=K:raise RuntimeError('Frozen gap/K5 required')
        self.native=load_parallel_native(self.torch,Path(__file__).resolve().parents[1])
        self.pool=CausalSignalPool(self.torch,self.native,self.config['device'],self.eos)
        self.pool.calibrate();self.signal_variant='A';self.production_event_records=0

    def begin(self,*args,**kwargs):
        state=super().begin(*args,**kwargs);start=time.perf_counter()
        state['call'].update(signal_accounting='per-dispatch causal CPU/GPU interval union upper + exclusive CPU materialization; no persistent cross-clock offset',
            signal_clock_bounds_scope='batch diagnostics only; each work interval carries its own causal enclosure')
        elapsed=1000*(time.perf_counter()-start);state['call']['ms']+=elapsed
        if state['collect'] and getattr(self,'signal_collection_enabled',True):state['call']['signal_ms']+=elapsed
        return state
