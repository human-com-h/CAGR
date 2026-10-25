"""Exercise the actual frozen feature and gate implementations without a GPU."""
from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'frozen/cohort_II/scripts'))
from v04_features import summarize_gap,question_features,utility_decision
from v2_fine_runtime import first_decision,second_decision

def main():
    r=json.loads((ROOT/'frozen/cohort_I/configs/deployment_test1000.json').read_text())['router']
    f=json.loads((ROOT/'frozen/cohort_II/configs/heldout_v2_deployment.json').read_text())['fine']
    signals={**summarize_gap([1.,2.,3.,4.,5.]),**question_features('Who wrote Hamlet?')}
    first,trace=utility_decision(r['stage0'],signals,r.get('routing',{}).get('lambda0',r['lambda']),r.get('routing',{}).get('threshold0',0.))
    evidence={**summarize_gap([3.,4.,5.,6.,7.]),'retrieval_score_variance':2.,'answer_length_tokens':3}
    print(json.dumps({'scope':'CPU example with synthetic features; no model loading or measured paper result',
        'CAGR-R':{'retrieve':first,'trace':trace},'CAGR-F':{'retrieve':first_decision(f,signals),
        'second_action':second_decision(f,evidence,'hotpotqa')},'model_calls':0},indent=2))
if __name__=='__main__':main()
