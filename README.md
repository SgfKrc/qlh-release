# qlh-release

Standalone QLH packaging and release workspace.

This repository owns the `packaging` sources, installer/spec resources,
release-only tests, signing/publication flow, release documentation, and the
CPU/CUDA packaging virtual environments. It consumes a checked-out QLH core
repository; it is not part of the edge runtime and does not add image
generation to QLH.

The core model sidecar requirement files live in the core repository under
`requirements/`. The release-only requirement files remain under
`packaging/`.

The canonical remote is `https://github.com/SgfKrc/qlh-release.git`. This
checkout is the local migration baseline.
