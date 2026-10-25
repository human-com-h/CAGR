# Public code scope

Only code, frozen configurations, minimal numerical evidence, reproduction tools,
tests and repository documentation are approved for routine publication.

The paper is internal input for implementation and scientific checking. A request
to release code does not authorize uploading the paper. Do not publish manuscript
PDFs, LaTeX projects, complete Markdown/Word papers, submission or revision
versions, reviewer reports, response letters or archives containing them without
separate explicit author approval. Keep originals in the author's research tree.

README may contain the title, authors, an abstract, citation metadata and links to
an already public DOI/arXiv page. Do not invent a publication link.

Before every push or release, run `python tools/check_publication_scope.py`
against the staged index and all reachable Git refs. Inspect every release ZIP or
TAR with `--archive PATH`. CI must fetch complete history. `.gitignore` is an
additional safeguard, not an access control or a substitute for this scan.

Enable the supplied pre-push hook with the repository-local setting
`git config --local core.hooksPath .githooks`. Prepare release archives through
`python tools/package_release.py --output dist/source.zip`; this refuses dirty
trees, contaminated history, and manuscript-bearing archives before writing.

If exposure is discovered, restrict visibility, retain an offline backup and
audit branches, tags, releases, PR refs, artifacts and cache access. A clean latest
commit alone is insufficient. History rewriting or deleting published assets
requires a concrete impact assessment and author confirmation. Keep the repository
private while that cleanup or author review is pending. External clones, downloads
and platform caches cannot be certified erased by this repository's tooling.
