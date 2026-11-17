# GitHub publication workspace

- Remote: `git@github.com:azbycxd/MSCAD.git`.
- This is an independent publication snapshot of the MSCAD thesis workspace. Original source branches and GitCode history remain local and unchanged.
- Before changes, inspect `git status --short`, `git branch --show-current`, `git remote -v`, and `git diff --stat`.
- Never run reset-hard, clean, or force push. Preserve unrelated changes.
- Commit and push only when the user requests them.
- Never commit datasets, PCAPs, checkpoints, optimizer states, raw score arrays, secrets, caches, or large outputs.
- Keep HAI and SWaT data/configs/outputs isolated. Preserve the frozen experimental protocols and conclusions.
- Small reviewed publication evidence belongs in `docs/thesis/evidence/`; `outputs/` remains ignored.
- Read `README.md` and `docs/publication/publication_manifest.json` for snapshot provenance and reproduction limits.
