# CAGR — Cost-Aware Gated Retrieval

Research code for **Cost-Aware Gated Retrieval: Decoupling Retrieval Initiation from Evidence-Conditioned Escalation**, by Zida Hao and Yubo Li, School of Software, Xinjiang University.

CAGR separates the decision to retrieve from the decision to escalate after observing evidence. This repository contains the Python/C++/CUDA implementations, frozen parameters, benchmark splits, and evaluation materials needed to reproduce the reported experiments.

**Status:** `v0.1.0-rc.3`. CPU evaluation replay, all 15 table exports, parameter reconstruction, and data/index reconstruction have been validated. The public GPU adapter still requires a fresh end-to-end run; see [reproducibility status](REPRODUCIBILITY.md).

A link to the paper will be added when a public preprint or publication is available.

## Results and scope

- **Test Cohort I:** CAGR-R saves 49.9% of retrieval calls; HotpotQA F1 improves by 0.96 percentage points over its matched binary gate (95% CI [0.22, 1.83]).
- **Test Cohort II:** CAGR-F averages 491.23 ms versus 563.13 ms for Always-RAG, a 12.77% reduction, and uses 44.7% fewer retrieval calls.
- Savings coexist with a 2.80 percentage point HotpotQA F1 deficit for CAGR-F versus Always-RAG. CAGR does not jointly dominate the strong TARG-style binary gate.

These are separate 1,000-question cohorts and frozen operating points. Do not pool them into one unchanged-policy experiment. Retrieval uses **483,690 HotpotQA training-context passages**, not the full Wikipedia benchmark. TARG-style and SUGAR-L-style are explicitly documented same-backbone adaptations.

## Level 1 — Quick Start

Python 3.11–3.13; no GPU or model weights needed for the core gate example:

```bash
git clone https://github.com/human-com-h/CAGR.git
cd CAGR
python -m venv .venv
# Activate .venv using your shell's usual command.
python -m pip install -r requirements.txt
python tools/quickstart.py
python tools/verify_evidence.py
python -m unittest discover -s tests -v
```

The quick start uses synthetic features with the actual frozen gates. It performs no language-model inference. To check real model loading and native CUDA inference, follow Level 3 with `--smoke 2`; this GPU route remains unverified in this release.

## Level 2 — Main Evaluation from released measurements

```bash
python tools/recompute_statistics.py
python tools/reproduce_tables.py
python tools/reproduce_quota_table.py
python tools/rebuild_parameters.py
```

Outputs are written under `reproduced/`: question-level bootstrap estimates, 15 CSV/LaTeX tables, and a parameter reconstruction receipt. No inference is needed. Main quality/cost estimates and intervals are recomputed from **16,000 measured question-method rows and 4,000 paired observations**. Development tables are rebuilt from archived evaluation outputs; their scope remains exploratory. Frozen expected CSVs are under `evidence/aggregate_tables/`.

## Level 3 — Full inference reproduction

There is no backbone fine-tuning. The learned components are small ridge heads, shipped in JSON and reconstructible from calibration-only numerical targets. Download official benchmark data and the locked Qwen checkpoint, and rebuild the train-only index:

```bash
python -m pip install -r requirements-resources.txt
python tools/prepare_resources.py --datasets --corpus
python tools/prepare_resources.py --model
```

Model use is subject to the **Qwen Research License**. The software's MIT license does not change the model or dataset terms. Downloads use exact revisions and SHA-256 checks; no private Kaggle assets are needed.

Use Linux, CUDA 12.8 with `nvcc`, a C++ compiler, Ninja, and a T4-compatible GPU. Original measurements used Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 4.51.3, FP16/SDPA, one designated T4, batch size 1 and concurrency 1, on a machine with two T4s. Other hardware can run functional checks but cannot reproduce the recorded absolute latency.

```bash
python -m pip install -r requirements-gpu.txt
python tools/evaluate.py --cohort II --shard 1 --model-dir resources/model --index resources/corpus.sqlite --questions resources/II_questions.jsonl --output runs/smoke-ii --smoke 2
# Run each shard in a separate process/session, preserving the original two-shard protocol.
python tools/evaluate.py --cohort I --shard 1 --model-dir resources/model --index resources/corpus.sqlite --questions resources/I_questions.jsonl --output runs/i-1
python tools/evaluate.py --cohort I --shard 2 --model-dir resources/model --index resources/corpus.sqlite --questions resources/I_questions.jsonl --output runs/i-2
python tools/score_predictions.py --run runs/i-1 --references resources/I_references.jsonl
python tools/score_predictions.py --run runs/i-2 --references resources/I_references.jsonl
python tools/merge_runs.py --shard-1 runs/i-1 --shard-2 runs/i-2 --output runs/cohort-i
python tools/recompute_statistics.py --cohort I --metrics runs/cohort-i/question_metrics.csv --pairs runs/cohort-i/paired_observations.jsonl --output reproduced/new-i
# For Cohort II use --cohort II, II_questions/II_references and separate ii-1 / ii-2 outputs.
```

The evaluation writes eight normal method executions and four technical on/off executions per question; two balanced pairs estimate overhead. It retains failures and partial costs and refuses existing output directories. Scoring only starts after a prediction hash receipt. See [timing protocol](docs/TIMING.md) and [experiment map](docs/EXPERIMENTS.md). No new GPU experiments were launched while preparing this repository.

## Repository layout

| Path | Purpose |
| --- | --- |
| `frozen/cohort_I/`, `frozen/cohort_II/` | 153 byte-preserved scientific source/configuration files |
| `tools/` | Public resource, inference, scoring, evidence, and table entry points |
| `configs/` | Locked public sources, benchmark IDs, prompts, budgets, operating points |
| `evidence/` | Question measurements, paired timing observations, calibration targets, expected tables |
| `results/` | Archived evaluation outputs required by the original table exporter |
| `provenance/` | SHA manifests, source transformations and validation receipts |
| `tests/`, `.github/` | Reproduction checks, CI and maintenance configuration |

See [resources and licenses](docs/RESOURCES.md), [release notes](CHANGELOG.md), [maintenance policy](MAINTENANCE.md), and [contributing](CONTRIBUTING.md). No DOI or arXiv identifier is claimed. Use [CITATION.cff](CITATION.cff) for software citation.
