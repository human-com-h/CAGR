"""Separate a documented scientific gate rejection from an unexpected crash."""
from pathlib import Path
import hashlib,json


class TechnicalGateRejected(RuntimeError):
    def __init__(self,stage,evidence):
        super().__init__('Scientific technical gate rejected: '+stage);self.stage=stage;self.evidence=evidence


def verify_smoke_rejection(root):
    root=Path(root);out=root/'results/online-v04/smoke'
    gate=json.loads((out/'v3_gate.json').read_text());receipt=json.loads((out/'acceptance.json').read_text())
    exact=json.loads((out/'v3_equivalence.json').read_text());report=json.loads((root/'logs/v3_20_double_drop.json').read_text())
    if gate['status']!='failed' or report['passed'] is not False or receipt['questions']!=20:
        raise RuntimeError('Gate rejection has no matching failed evidence')
    if exact['status']!='passed' or (exact['all_method_question_pairs'],exact['routed_pairs'],exact['routed_decisions'])!=(160,80,94):
        raise RuntimeError('Scientific equivalence failure is NOT a controlled timing rejection')
    for name,h in receipt['evidence_sha256'].items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Rejected gate evidence SHA differs')
    if (root/'results/online-v04/calibration').exists():raise RuntimeError('Calibration incorrectly started despite failed smoke gate')
    return {'stage':'smoke','gate':'results/online-v04/smoke/v3_gate.json','twenty':'logs/v3_20_double_drop.json',
        'scientific_acceptance':False,'calibration_started':False}


def guarded_partition(original,root,partition,g,*args,**kwargs):
    try:return original(partition,g,*args,**kwargs)
    except RuntimeError as exc:
        if partition=='smoke' and str(exc)=='20 technical no-rebound/overhead gate failed; 200 blocked':
            raise TechnicalGateRejected('smoke',verify_smoke_rejection(root)) from exc
        raise
