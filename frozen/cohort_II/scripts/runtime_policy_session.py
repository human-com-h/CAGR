"""One validated immutable policy per process; verify exact bytes on every access.

The existing guard functions are unchanged. Their loader is scoped to this
context and restored on exit, including failure. No cached model authorization
is available across workspace roots or after the policy bytes change.
"""
from pathlib import Path
from types import MappingProxyType
import hashlib,time
import execution_policy


def immutable(value):
    if isinstance(value,dict):return MappingProxyType({k:immutable(v) for k,v in value.items()})
    if isinstance(value,list):return tuple(immutable(v) for v in value)
    return value


class RuntimePolicySession:
    def __init__(self,root):
        self.root=Path(root).resolve();self.path=self.root/'configs/execution_policy.yaml'
        self.original=None;self.bound_loader=self.load
        self.record={'kind':'immutable_execution_policy_session','status':'not_started',
                     'parse_count':0,'binding_checks':0,'binding_check_ms':0.,'binding_failures':0,
                     'per_access_exact_bytes_SHA256':True,'existing_guards_unchanged':True}

    def __enter__(self):
        if getattr(execution_policy.load_policy,'__self__',None) is not None:
            raise execution_policy.PolicyError('Nested policy sessions are forbidden')
        start=time.perf_counter();self.original=execution_policy.load_policy
        before=self.path.read_bytes();parsed=self.original(self.root)
        if self.path.read_bytes()!=before:raise execution_policy.PolicyError('Policy changed during validation')
        self.digest=hashlib.sha256(before).hexdigest();self.policy=immutable(parsed)
        self.record.update(status='active',parse_count=1,policy_sha256=self.digest,
                           initialization_ms=1000*(time.perf_counter()-start),root=str(self.root))
        execution_policy.load_policy=self.bound_loader
        return self

    def load(self,root=execution_policy.ROOT):
        start=time.perf_counter()
        try:
            if Path(root).resolve()!=self.root:raise execution_policy.PolicyError('Cached policy belongs to a different workspace')
            if hashlib.sha256(self.path.read_bytes()).hexdigest()!=self.digest:
                raise execution_policy.PolicyError('Frozen execution policy bytes changed; run is blocked')
            self.record['binding_checks']+=1
            return self.policy
        except BaseException:
            self.record['binding_failures']+=1
            raise
        finally:self.record['binding_check_ms']+=1000*(time.perf_counter()-start)

    def __exit__(self,kind,value,trace):
        execution_policy.load_policy=self.original
        self.record['status']='closed' if kind is None else 'closed_after_failure'

