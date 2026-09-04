---
name: dependency-license-audit
description: Inspect repository manifests and license signals for dependency, packaging, and commercial-use risks.
---

# Dependency and license audit

Use manifest and configuration evidence to assess integration burden; do not provide a legal conclusion.

- Locate the ecosystem's primary manifest and lockfile, such as `pyproject.toml`, `package.json`, `go.mod`, or `Cargo.toml`, plus container/deployment files when relevant.
- Separate required runtime dependencies from optional, development, and example dependencies. Note external infrastructure implied by SDKs or deployment configuration.
- Compare the repository metadata license with the checked-in license file. Flag missing, custom, source-available, copyleft, or inconsistent signals for human legal review.
- Treat transitive-license compatibility as unverified unless a dependency inventory or SBOM directly supports it.
- Report concrete file paths and refs with every material dependency or license finding.
