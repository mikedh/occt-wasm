# mikedh/occt-wasm — the rmesh fork

This fork (branch `rmesh-minimal`) is a **single commit** on top of
`andymai/occt-wasm` for consumption by [rmesh](https://github.com/mikedh)'s
CAD system:

- **Runtime module supply + epoch hooks + wasmtime re-export** (`crate/`) —
  embedding-host features; upstream-PR candidates.
- **IR-native BrepModel lift** (`facade/src/rmesh_brep.cpp`) — an out-of-tree
  facade extension discovered by the `facade/src/*.cpp` glob.
- **Derived `--minimal` profile** — `cargo xtask build-wasi --release
  --minimal` builds `dist/occt-wasm-minimal.wasm.br` with everything outside
  the construction-op surface dead-code-eliminated. The policy lives in two
  validated consts (`config::OPTIONAL_CATEGORIES` — exchange, projection,
  xcaf, offsetting, primitives, query, curve, evolution, sweep, io — and the
  per-spec `config::OPTIONAL_SPECS`); the export root set is derived from the
  codegen specs through the real emitter — there is no hand-maintained
  symbol list — and the crate binds optional wrappers lazily, returning
  `OcctError::MissingCapability` when an export is absent. Core keeps
  construction, booleans (+ `booleanFuzzy` and the healing category for the
  reliability ladder), fillet/chamfer, transforms, topology, tessellation.
- **`.github/workflows/rmesh-minimal.yml`** — the only CI that runs here; the
  seven upstream workflows stay in-tree but are disabled server-side
  (`gh workflow disable <file> -R mikedh/occt-wasm`). A green push publishes
  the minimal blob to the rolling `rmesh-kernel-v1` release, which rmesh's
  `make fetch-kernel` downloads.

## Tracking upstream

The branch is a single commit **rebased** onto `upstream/main` (no merge
commits):

```bash
git fetch upstream
git rebase upstream/main
# ... post-rebase ritual below ...
git push --force-with-lease origin rmesh-minimal
```

The force-push re-runs CI and re-clobbers the release asset — that's fine.

### Post-rebase ritual

1. **Regenerate, don't resolve**: for any conflict in `crate/src/kernel_generated.rs`
   or `facade/generated/*`, take either side, then:
   `cargo xtask codegen && cargo fmt --all`
2. `cargo test -p xtask` — spec validation, emitter shapes, derived-set
   partition.
3. Rebuild both blobs (needs the builder image or emsdk + `occt/build` libs):
   `cargo xtask build-wasi --release && cargo xtask build-wasi --release --minimal`
4. `cargo test --release -p occt-wasm` and
   `cargo test --release -p occt-wasm --test minimal` — full-blob suite plus
   the minimal load-verification (instantiation is the invariant: every
   eagerly-bound export must exist in the minimal blob).
5. Commit refreshed `crate/src/occt-wasm.wasm.br` and
   `dist/occt-wasm-minimal.wasm.br` if the bytes changed (both are
   force-tracked; CI stale-checks the minimal one).

New workflow files arriving from upstream start **enabled** — disable them:
`gh workflow disable <file> -R mikedh/occt-wasm`.

Without the builder image locally, run builds in the container:

```bash
docker run --rm -v "$PWD":/src -w /src -e CARGO_TARGET_DIR=/src/target-docker \
  ghcr.io/andymai/occt-wasm-builder:latest \
  bash -c "cargo xtask build-wasi --release --minimal"
```
