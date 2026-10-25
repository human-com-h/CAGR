from pathlib import Path
import csv,hashlib,json,math,subprocess,sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT/'frozen/cohort_II/scripts'))
from v04_features import numpy_step_signals,summarize_gap,utility_decision
from v2_fine_runtime import first_decision,second_decision
from v2_evaluation_plan import method_order,pair_orders,METHODS
from run_baselines import answer_metrics
from verify_evidence import verify

class Reproduction(unittest.TestCase):
    def test_frozen_sources_and_means(self):
        r=verify();self.assertEqual(r['main_scalar_checks'],128);self.assertEqual(r['frozen_source_files'],153)
    def test_split_isolation(self):
        data=json.loads((ROOT/'configs/split_ids.json').read_text());ids={k:{r['sample_id'] for r in v} for k,v in data.items()}
        self.assertEqual([len(ids[k]) for k in ['I','II','calibration','smoke']],[1000,1000,200,20])
        for a in ids:
            for b in ids:
                if a!=b:self.assertFalse(ids[a]&ids[b])
        for c in ['I','II']:
            self.assertEqual([r['global_index'] for r in data[c]],list(range(1000)))
    def test_greedy_signal_ties_shift_and_invalid_values(self):
        for values in [[3.,3.,1.],[1.,2.,3.],[-5.,-1.,-2.]]:
            v=np.array(values);expected=sorted(values)[-1]-sorted(values)[-2]
            self.assertEqual(numpy_step_signals(v)['logit_gap'],expected)
            self.assertEqual(numpy_step_signals(v+100)['logit_gap'],expected)
        with self.assertRaises(ValueError):numpy_step_signals(np.array([1.,3.]),winner=0)
        with self.assertRaises(ValueError):summarize_gap([float('nan')])
        s=summarize_gap([0.,6.]);self.assertGreater(s['targ_mean_exp_gap'],s['gap_uncertainty_of_mean'])
    def test_fine_gate_strict_threshold_and_dataset_stop(self):
        f=json.loads((ROOT/'frozen/cohort_II/configs/heldout_v2_deployment.json').read_text())['fine'];h=f['deployment']
        self.assertFalse(first_decision(f,{'targ_mean_exp_gap':h['threshold0']}))
        self.assertTrue(first_decision(f,{'targ_mean_exp_gap':h['threshold0']+1e-6}))
        features={'mean_logit_gap':h['trivia_confidence_threshold'],'retrieval_score_variance':1.,'answer_length_tokens':3}
        self.assertEqual(second_decision(f,features,'triviaqa'),'stop')
    def test_charged_method_and_pair_coverage(self):
        for index in range(16):
            self.assertEqual(set(method_order(index)),set(METHODS));self.assertEqual(len(method_order(index)),8)
            orders=pair_orders(index,2);self.assertEqual({tuple(x) for x in orders},{('on','off'),('off','on')})
    def test_official_answer_normalization(self):
        self.assertEqual(answer_metrics('The Hamlet.',['Hamlet'],'triviaqa'),{'em':1.,'f1':1.})
        self.assertEqual(answer_metrics('yes',['no'],'hotpotqa'),{'em':0.,'f1':0.})
        self.assertEqual(answer_metrics('London',['London','LONDON'],'triviaqa')['em'],1.)
    def test_all_fifteen_table_cells(self):
        for script in ['reproduce_tables.py','reproduce_quota_table.py']:
            subprocess.run([sys.executable,str(ROOT/'tools'/script)],check=True,capture_output=True)
        expected=list((ROOT/'evidence/aggregate_tables').glob('*.csv'));self.assertEqual(len(expected),15)
        for p in expected:
            with p.open(encoding='utf-8',newline='') as f:a=list(csv.reader(f))
            with (ROOT/'reproduced/source_data'/p.name).open(encoding='utf-8',newline='') as f:b=list(csv.reader(f))
            self.assertEqual(a,b,p.name)
    def test_selected_parameters_reconstruct(self):
        subprocess.run([sys.executable,str(ROOT/'tools/rebuild_parameters.py')],check=True,capture_output=True)
        r=json.loads((ROOT/'reproduced/parameter_rebuild.json').read_text());self.assertEqual(r['development_decisions_replayed'],400)
    def test_entry_points_help_without_gpu(self):
        for script in ['prepare_resources.py','evaluate.py','score_predictions.py','recompute_statistics.py','merge_runs.py']:
            subprocess.run([sys.executable,str(ROOT/'tools'/script),'--help'],check=True,capture_output=True)
    def test_public_permission_adapter_imports_runtime_without_private_assets(self):
        code="""from pathlib import Path
import sys
root=Path.cwd()
sys.path[:0]=[str(root/'tools'),str(root/'frozen/cohort_II/scripts')]
from public_execution import public_runtime
with public_runtime() as receipt:
    from types import SimpleNamespace
    from v04_runtime import SharedRuntime
    from v04_inference import guard_model_execution
    guard_model_execution()
    g=SimpleNamespace(config={'repo_id':'Qwen/Qwen2.5-3B-Instruct'},revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',calls=[])
    rt=SharedRuntime(g,None,{}, {})
    assert receipt['private_project_guard_adapter'] and rt.max_retries==0
"""
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True,capture_output=True)

if __name__=='__main__':unittest.main()
