# Manuscript-to-code map

All table stems below refer to manuscript LaTeX labels and `evidence/aggregate_tables/`. `python tools/reproduce_tables.py` exports the first 14 tables; `python tools/reproduce_quota_table.py` exports the fifteenth. No published table is a handwritten numeric constant in these public exporters.

| Manuscript result | Actual implementation / inputs | Command / output |
| --- | --- | --- |
| Cohort I main quality/efficiency (`cohort_i`, `cost_i`, `contrasts_i`) | `frozen/cohort_I/scripts/v04_runtime.py`, `v04_v3_native_state_inference.py`; `deployment_test1000.json`; 500 HotpotQA + 500 TriviaQA | `tools/evaluate.py --cohort I ...`; eight methods; `verify_evidence.py`, `recompute_statistics.py` |
| Cohort II main quality/efficiency (`cohort_ii`, `cost_ii`, `contrasts_ii`) | Same native B state; `v2_fine_runtime.py`, `kv_bridge_inference.py`; `heldout_v2_deployment.json`; separate 500+500 questions | `tools/evaluate.py --cohort II ...`; eight methods, with dataset-conditioned TriviaQA stopping disclosed |
| Incremental overhead (`overhead`) | Original on/off technical executions and component ledgers; two pairs/question | `recompute_statistics.py`; `reproduced/statistics/cohort_*.json` |
| Stage 0/1 feature studies (`stage0`, `stage1`) | Archived OOF development analyses under `results/`; feature functions in `v04_features.py` | `reproduce_tables.py`; exploratory, no independent claim |
| Four forced actions / cache extension (`actions`, `bridge`) | `route23_runtime.py`, `kv_bridge_inference.py`; archived Cal200 action-quality and bridge diagnostics | `reproduce_tables.py`; potential outcomes, not a deployed policy |
| Prefix truncation / random gate (`prefix`, `random`) | Archived saved-answer evaluation outputs; fixed K=5 heads; 20,000 seed-42 assignments | `reproduce_tables.py`; no new K-specific training or timing |
| Retrieval cost sensitivity (`cost_spectrum`) | Cohort II measured paths and retrieval counts | `reproduce_tables.py`; fixed-path algebraic scenarios |
| First-stage quota diagnostic (`quota_alignment`) | Saved Direct/single-retrieval outcomes in existing diagnostic JSON | `reproduce_quota_table.py`; post hoc, no new policy selection |
| Learned state | Four CAGR-R ridge heads; CAGR-F three action heads × Full/five folds and training quantile thresholds | `rebuild_parameters.py`; numeric target sets isolate calibration from both tests |

Data versions, software, resources, seeds and permitted deviations are in `REPRODUCIBILITY.md`, `RESOURCES.md`, `TIMING.md` and the frozen configurations. Numerical source data for the reported figures are supplied; manuscript files and finished paper figures are maintained separately. The backbone and neural weights are unchanged; there is no additional backbone-training procedure.
