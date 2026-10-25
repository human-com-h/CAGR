# Resources, attribution and redistribution

| Category | Material | Provided / reconstruction |
| --- | --- | --- |
| A — provided | Original source, JSON ridge weights, thresholds, prompts, budgets, test IDs, fold IDs, numerical calibration targets, measured scalar outputs | Versioned directly in this repository; SHA manifests in `provenance/` |
| B — official download | [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/tree/aa8e72537993ba99e69dfaafa59ed015b17504d1) | `prepare_resources.py --model`; ~6.18 GB; [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE), research/evaluation use; separate commercial terms |
| B — official download | [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab) | CC BY-SA 4.0; [original attribution and terms](https://hotpotqa.github.io/); distractor validation and two train Parquets |
| B — official download | [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa/tree/0f7faf33a3908546c6fd5b73a660e0f8ff173c2f) | unfiltered.nocontext validation; [author repository terms](https://github.com/mandarjoshi90/triviaqa); Apache-2.0 as described by the authors; Hub metadata alone says unknown |
| C — deterministic rebuild | 483,690-passage FTS5/BM25 index | `prepare_resources.py --corpus`; train context only; ~466 MB SQLite; CC BY-SA attribution retained, raw text stays outside Git |
| C — deterministic rebuild | Fixed question/reference files | `prepare_resources.py --datasets`; source SHA and question SHA checked; original ID order and folds retained |
| C — reconstruction | Selected linear heads | `rebuild_parameters.py`; calibration targets only, no test selection |
| D — excluded | Credentials, private notebooks/mounts, keys, local paths, environment caches, wheel bundles, exploratory checkpoints | Not required by the public adapter; never uploaded |

No neural router checkpoint or fine-tuned backbone exists. The required learned state consists of the supplied small JSON heads. No large binary needs separate hosting. Model weights and source benchmark text are downloaded from the providers instead of redistributed through Git. Budget-adapted TARG/SUGAR methods use original local code and common Qwen weights, not third-party checkpoints.

Software and repository documentation use MIT, as confirmed by the authors. Third-party data, model weights and libraries retain their own terms; no software license overrides upstream rights. The HotpotQA-derived corpus is a train-context experimental corpus, not a claim to distribute or evaluate fullwiki. Resource SHA values are recorded in `configs/generation_settings.json`, `configs/corpus_expected.json` and `frozen/*/configs/model_asset_lock.json`.
