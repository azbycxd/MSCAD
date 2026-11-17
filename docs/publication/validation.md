# GitHub snapshot validation

Date: 2026-11-17 (Asia/Shanghai).

- Source: `mscad-thesis`, branch `thesis-clean`; exact source commit and every copied file hash are recorded in `publication_manifest.json`.
- 463 source/document/config/test files and 50 curated, small evidence artifacts were copied.
- Each copied artifact was compared by SHA256. The source copies are byte-identical, except the explicitly replaced publication `.gitignore` and `AGENTS.md`. The new README explains snapshot scope and historical document status.
- 266 Python files passed `ast.parse` syntax validation; this does not establish all runtime dependencies or complete experiment reproducibility.
- Credential-pattern scan found no private keys, common GitHub/AWS/OpenAI token formats, embedded URL passwords, or matching literal-secret assignments. This is a scoped pattern scan, not a guarantee about every possible secret format.
- All selected files are below 5 MiB. Raw data, PCAP, model weights, optimizer/RNG training states, raw score arrays, external clones and bulk outputs are excluded.
- Data-free tests: **9 passed, 2 deselected** in 15.42 s using the existing `p310` interpreter.

Command (from the publication repository):

```powershell
python -m pytest -q -p no:cacheprovider tests/test_project_paths.py tests/test_thesis_environment.py -k "not current_a12_compatibility_path_is_read_only and not reconstruction_backbone_preserves_btc_shape"
```

The two excluded tests require local A12 data or execute a reconstruction model. This publication did not run model training, model inference, or any experiment evaluation. Core environment imports and tensor arithmetic were checked. Full artifact-dependent test suites were not run.

Git ignore probes passed for dataset(s), outputs, checkpoints, external clones, `.env`, temporary files and `.codex_tmp`.

The staged allowlist contains 517 files including publication metadata. CRLF line endings are retained; the CR-aware whitespace check reports 59 inherited trailing-whitespace/end-of-file warnings. These are preserved to keep research source/evidence hashes unchanged; no conflict markers were found.

The original `mscad`, `mscad-swat2026`, and `mscad-thesis` branches/remotes are retained. GitHub receives an independent initial snapshot, not the original GitCode history. No reset, clean, force push, dataset movement, checkpoint deletion, or algorithm change is part of this publication.
