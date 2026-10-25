# Contributing

For a reproduction issue, include the tag/commit, command, operating system, Python/PyTorch/Transformers/CUDA/SQLite versions, GPU model, resource hashes, and the first failing traceback. Remove tokens, raw private data and local account paths. Use the issue templates.

Keep changes outside `frozen/`; those files are an immutable scientific reference. Add a meaningful regression for a scientific or data-integrity bug. Run `python -m unittest discover -s tests -v`, `python tools/verify_evidence.py`, `python tools/recompute_statistics.py` and `python tools/rebuild_parameters.py` as relevant. Describe any effects on prompts, budgets, token selection, routes, timing or benchmark membership. Documentation-only changes need appropriate link/content checks.

Preserve reference isolation, seed 42 for new operations, full cost accounting and honest verification status. Ordinary dependency upgrades must not silently replace the original paper environment. Contributors retain attribution; original software contributions are accepted under MIT.
