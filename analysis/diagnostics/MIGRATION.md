# Checkout migration and preservation — 1 October 2026

The user explicitly replaced the earlier personal-only development instruction with shared development on Aidan’s repository, using the shared `Vikas` branch.

- Active folder: `.`.
- Working remote `origin`: `https://github.com/aidanccc/qrc-volatility-research.git`.
- Active branch: `Vikas`, tracking `origin/Vikas`, starting commit `008a1d152a50147b5429bcca6cc81a8cf6d5da3e`.
- Personal remote retained as `personal`: `https://github.com/vikasyarlagadda/quantum-reservoir-realized-volatility-predictor.git`.
- Previous local `Vikas` preserved as `backup/personal-Vikas-2026-10-01`. Personal `main` remains at `e88ceec`; no history was rewritten.
- Personal modified and untracked work preserved in the stash titled `Personal work preserved before shared Vikas switch 2026-10-01`.
- Filesystem backup: `../qrc-volatility-research-personal-backup-2026-10-01`. All 7,970 copied files were individually SHA-256 verified before switching. `BACKUP_INVENTORY.json`, `BACKUP_STATUS.txt`, and `BACKUP_REFS.txt` record the source snapshot. The `.venv` remained in place; disposable Python bytecode was excluded.
- 7,721 ignored files under personal data/results/docs were moved out of the active tree into the backup’s `isolated-ignored-files` directory after verified copying. `ISOLATED_IGNORED.json` lists them. This prevents old generated inputs/caches from contaminating the shared study. These files were archived, not discarded.
- Local `AGENTS.md` was restored as working guidance. No personal analysis implementation was imported. The existing `.venv`, local application configuration, and unrelated branch references were retained.

No commits, pushes, remote branch updates, or merges were performed as part of this migration. Published shared artifacts stay in their original locations; the new diagnostics live in a separate dated run. No artifacts were deleted merely to tidy the repository.

To recover personal work later, use the verified backup as the primary reference. Do not apply the personal stash directly on shared Vikas: it belongs to the earlier personal main tree. Restore on a separate personal branch/checkout after accounting for any ongoing work.

After archiving, 122 empty directory shells were removed from the active data/results trees. No files were deleted. The exact list and reason are recorded in `results/diagnostics-2026-10-01/cleanup.json`.
