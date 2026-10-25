# Reproducibility status

This is a release candidate, not a verified fresh GPU reproduction. Original GPU measurements remain immutable; preparation of this repository made no new model calls or scientific parameter search.

| Experiment / material | Entry point | Resources | Actual status |
| --- | --- | --- | --- |
| Core signal and two frozen gates | `python tools/quickstart.py` | NumPy, fixed JSON heads | Executed on CPU; synthetic example |
| Main EM/F1, latency and retrieval means | `python tools/verify_evidence.py` | 16,000 question-method rows | 128 scalar comparisons passed |
| Original bootstrap intervals and timing estimators | `python tools/recompute_statistics.py` | 4,000 pairs; seed 42; 20,000 resamples | Executed locally; receipt in `provenance/validation.json` |
| All 15 manuscript table CSV/LaTeX exports | `python tools/reproduce_tables.py` then `python tools/reproduce_quota_table.py` | Archived evaluation JSONs | Executed; CSV cells compared to original tables |
| Selected ridge-head reconstruction | `python tools/rebuild_parameters.py` | Two Cal200 numeric target sets | 22 heads, six CAGR-F threshold sets and 400 OOF/Full decisions passed; tolerance 1e-11 |
| Public benchmark reconstruction | `python tools/prepare_resources.py --datasets` | Official locked validation Parquets | 2,220 questions restored and matched by SHA; no resampling |
| Train-only BM25 reconstruction | `python tools/prepare_resources.py --corpus` | Two official HotpotQA train Parquets | 483,690 documents; ordered-ID SHA and entire SQLite file SHA match original |
| Official model download and integrity check | `python tools/prepare_resources.py --model` | ~6.18 GB locked checkpoint | Public entry implemented; not downloaded again locally |
| Public native GPU smoke / full cohorts | `python tools/evaluate.py ...` | Linux/CUDA/T4, checkpoint and corpus | Not executed in this release; requires GPU validation |
| Re-score new predictions | `python tools/score_predictions.py ...` | Committed predictions and restored official references | Scoring implementation retained; no newly generated prediction set available |
| New full action collection / exploratory search | Original source in `frozen/` and fixed numerical targets | Historical development protocol | Not re-run; unnecessary for evaluating the shipped frozen parameters |

## Identity and variance

The backbone revision is `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Prompts, 5-token observation window, 64 answer tokens, 48 rewrite tokens, at most two retrievals, 181 total output tokens and 16,384 total input tokens are frozen. Inference and bootstrap seed is 42. IDs preserve historical sampling; the date-shaped seed in original historical configuration is not reused for new sampling.

Saved-result table reconstruction should agree exactly at manuscript rounding. Reconstructed heads are compared at 1e-11 absolute/relative numerical tolerance. Original greedy inference requires token/route/context equality under its original runtime. A fresh public-adapter run has no empirically established EM/F1 tolerance yet; do not invent one. Small FP16/runtime differences can change ties and routes. Latency has no promised universal tolerance: report GPU, CPU, SQLite, CUDA, warmups and exact runtime, and rerun the registered paired estimator.

The new adapter changes public filesystem/resource access, execution orchestration and private-project permission plumbing. The explicit public process binds a hash-checked, offline/no-submission policy before importing the original runtime; it leaves the archived source and the research workspace's policy unchanged. Native inference, gate decisions, prompts, budgets, baseline behavior and kernel sources are reused directly. GPU scientific-equivalence and cold clean-environment installation are outstanding gates before a stable reproducibility release. Frozen controllers retain original private-environment checks for provenance; the documented public entry is `tools/evaluate.py`.
