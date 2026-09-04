---
name: repository-due-diligence
description: Verify an open-source project's engineering health and claimed capabilities from repository evidence.
---

# Repository due diligence

Treat the repository as primary evidence for what is shipped, while distinguishing implementation from documentation claims.

- Establish the repository identity, default branch, archive state, recent push, releases, license, and issue activity first.
- Use `github_list_directory` to locate manifests, lockfiles, CI workflows, deployment files, examples, and docs. Read only files relevant to the active requirement with `github_get_file`.
- Prefer a tagged release or commit `ref` when the task names a version. Otherwise state that evidence comes from the default branch.
- Do not infer production readiness from stars or README claims alone. Look for tests/CI, release cadence, maintained examples, typed configuration, migration notes, and operational guidance.
- Record contradictory signals explicitly. Absence from the inspected files means “not verified,” not “unsupported.”
- Stop inspecting when each active hard constraint has either direct evidence or a clearly stated evidence gap.
