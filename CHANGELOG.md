# Changelog

## 0.1.0-rc.3

- Rebuild the repository with a new Git root containing only reviewed code and reproduction materials.
- Do not import prior Git history, tags, pull requests, or release assets.
- Retain manuscript publication guards and frozen scientific source/evidence.
- Remain private for author review; GPU verification remains outstanding.

## 0.1.0-rc.2

- Focus the repository on research code and reproducibility materials: remove the manuscript, bibliography, and finished paper figures from the default branch and this release's source archives.
- Keep frozen scientific implementations, parameters, benchmark IDs, numerical evidence, and all 15 table exporters unchanged.
- Update README, citation metadata, and resource documentation for the code-only layout. A paper link will be added once publicly available.
- GPU verification status is unchanged: this remains a prerelease.

## 0.1.0-rc.1

- Publish two immutable scientific source snapshots and frozen parameters for CAGR-R/CAGR-F.
- Provide official-source reconstruction of fixed benchmark splits and the train-only BM25 index.
- Add CPU gate example, scalar measurement checks, original bootstrap estimators, 15 table exporters and selected-head reconstruction.
- Add a public filesystem adapter for native GPU inference and separate scoring, with explicit unverified GPU status.
- Add CI, security checks, link maintenance, dependency update configuration and contributor guidance.

This release candidate does not certify a fresh GPU run or a stable full-reproduction release. Historical measurements and limitations are preserved.
