# Maintenance and version policy

`v0.1.0-rc.3` is the code-only snapshot of the rebuilt repository. Earlier repository history and releases are retained only in offline backups and must not be imported. Do not move or overwrite a published tag. A stable paper reproduction release requires a clean Linux/T4 smoke run, scientific route/token/context equivalence checks, full documented inference/scoring execution and validated environment installation. No GPU run is automatically launched by this repository's workflows.

GitHub Actions runs CPU tests, frozen-source hashes, scalar statistics, parameter reconstruction and security checks on pushes and pull requests. A weekly/manual maintenance workflow checks pinned provider links. Dependabot proposes monthly dependency and Actions updates. These mechanisms run on GitHub after the workflows are pushed and enabled; they do not imply a continuously running agent.

Future agent sessions triage reproduction Issues, investigate CI/resource failures, prepare a branch and PR, run relevant checks, merge authorized repairs and issue patch releases. Changes affecting scientific behavior require a new version and a documented comparison against the immutable original snapshots. Never retune on the two published test cohorts. Mark evidence replay, synthetic tests and new GPU observations separately.

The account owner handles access/identity challenges, copyright decisions and platform contracts. Authors confirm publication metadata/DOI and any changed submission policy. Zenodo DOI minting is deferred until an authenticated Zenodo integration and a stable release are available; no DOI is promised or fabricated. No local scheduled wakeup or background monitoring is installed.
