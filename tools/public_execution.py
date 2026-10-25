"""Replace private project permission plumbing in this explicit public process.

Algorithm, generation, retrieval, gate and budget functions remain untouched.
No context is activated merely by importing this module.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
@contextmanager
def public_runtime():
    import execution_policy
    p=ROOT/'configs/public_execution.json';raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();value=json.loads(raw)
    expected={'scope':'explicit_public_command_only','seed':42,'runtime_downloads':False,'cloud_submission':False,'automatic_retries':0,'frozen_parameters_only':True}
    if value!=expected:raise RuntimeError('Public execution policy differs')
    def guard(*args,**kwargs):
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise RuntimeError('Public policy changed during execution')
        return value.copy()
    original={k:getattr(execution_policy,k) for k in ['guard_model_execution','guard_local_v04']}
    try:
        execution_policy.guard_model_execution=guard;execution_policy.guard_local_v04=guard
        yield {'policy_sha256':digest,'scope':value['scope'],'private_project_guard_adapter':True}
    finally:
        for k,v in original.items():setattr(execution_policy,k,v)
