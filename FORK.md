# mikedh/occt-wasm — the rmesh fork

Branch `rmesh-minimal`, **ten commits** on top of `andymai/occt-wasm`, for
consumption by [rmesh](https://github.com/mikedh)'s CAD system.

## What the fork adds

- **Runtime module supply + epoch hooks + wasmtime re-export** (`crate/`) —
  embedding-host features; upstream-PR candidates.
- **Two out-of-tree facade extensions**, both discovered by the
  `facade/src/*.cpp` glob rather than named anywhere:
  - `facade/src/rmesh_brep.cpp` — the IR-native `BrepModel` lift.
  - `facade/src/rmesh_history.cpp` — 20 exports, the six named provenance
    channels rmesh reads instead of a framed stream. **rmesh hard-depends on
    this file existing in the checkout**: `crates/rmesh/tests/engine_occt.rs`
    reads it to assert the enumeration order the IR lift relies on.
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
  `make kernel` downloads.

The fork **deletes no files** from upstream, but it is not additive at the line
level either: 6896 insertions against 1203 deletions across 23 files, mostly in
the three emitters. The CI header's "additive-only" claim describes the file
inventory, not the diff.

### Which commits could go upstream

Offering work upstream is the only thing that shrinks a fork permanently. Four
of the ten are upstream-PR shaped as they stand:

| commit | what it is |
|---|---|
| `e183548` | `exports(name)` — ask the loaded module whether it has a symbol |
| `318d636` + `6ca510d` | a missing blob fails the test rather than skipping it |
| `39c89e2` | a trap must not be replaced by the complaint about cleaning up after it |
| the embedding-hooks half of `830d4a2` | runtime module supply, epoch hooks, wasmtime re-export |

The rest (the minimal profile, the IR lift, the provenance channels) are rmesh's
own shape and belong here.

## Branch inventory

| branch | what it is |
|---|---|
| `rmesh-minimal` | **the fork.** The only branch that matters. |
| `main` | upstream tracking. |

`origin/rmesh-old{,1,2}` are still on GitHub — three dead ends from the
2026-08-01 rebase-vs-merge episode. Nothing reads them; delete them when
convenient (`git push origin --delete rmesh-old rmesh-old1 rmesh-old2`).

**`rmesh-crate` is gone**, along with `scripts/make-rmesh-crate.sh`. It was a
generated orphan branch holding `crate/` alone, published so rmesh could take a
cargo git dependency without cargo cloning `occt/` — 335 MB of checkout, 520 MB
of it OpenCASCADE source, resolved at ZERO features — to obtain 370 KB of Rust.

It was stale by fourteen commits, local-only, and wired into nothing. rmesh takes
the **plain git dependency** on `rmesh-minimal` instead and pays that 335 MB once
per machine — on machines that already carry a ~1.2 GB checkout of this repo. One
pin, one branch, nothing generated.

The kernel BLOB is the other artifact this repo produces, and it is the one whose
cost actually matters: a ~3 MB brotli download from the `rmesh-kernel-v1`
release, digest-pinned, fetched lazily by rmesh's `make kernel` — or built here
and copied into place by `make build-wasm`. It never goes through cargo.

## Tracking upstream

**Merge, do not rebase.** The branch is published and rmesh's `Cargo.toml` pins
a rev on it; a rebase invalidates that pin and forces a push everything
downstream has to be told about. This file used to document the opposite
(`git rebase upstream/main` + `--force-with-lease`), which is how the three dead
branches above came to exist.

```bash
git fetch upstream
git merge upstream/main
# ... post-merge ritual below ...
git push origin rmesh-minimal
```

### Post-merge ritual

1. **Regenerate, don't resolve**: for any conflict in `crate/src/kernel_generated.rs`
   or `facade/generated/*`, take either side, then:
   `cargo xtask codegen && cargo fmt --all`
2. `cargo test -p xtask` — spec validation, emitter shapes, derived-set
   partition. (Not in CI here; see the workflow's TODO.)
3. Rebuild both blobs (needs the builder image or emsdk + `occt/build` libs):
   `cargo xtask build-wasi --release && cargo xtask build-wasi --release --minimal`
4. `cargo test --release -p occt-wasm` and
   `cargo test --release -p occt-wasm --test minimal` — full-blob suite plus
   the minimal load-verification (instantiation is the invariant: every
   eagerly-bound export must exist in the minimal blob).
5. Commit refreshed `crate/src/occt-wasm.wasm.br` and
   `dist/occt-wasm-minimal.wasm.br` if the bytes changed. Neither is
   force-tracked — `.gitignore`'s `*.wasm` does not match `*.wasm.br`, so both
   are ordinary tracked files. CI stale-checks the minimal one.

**Holding the OCCT submodule.** Upstream bumps `occt/` in the course of ordinary
releases (`e774ad0` moves it to V8.0.1). A submodule bump invalidates all 49
prebuilt static libs under `occt/build/lin32/clang/lib/` and turns a
five-minute container build into a multi-hour one, so a merge that is only after
upstream's Rust and facade work should keep the gitlink:

```bash
git merge upstream/main
git checkout HEAD -- occt        # hold the submodule at the built revision
```

Take the bump deliberately, on its own, when there is time for the rebuild and
for reading whatever moved in the goldens.

New workflow files arriving from upstream start **enabled** — disable them:
`gh workflow disable <file> -R mikedh/occt-wasm`.

Without the builder image locally, run builds in the container — this is what
rmesh's `make build-wasm` does for you:

```bash
docker run --rm -v "$PWD":/src -w /src -e CARGO_TARGET_DIR=/src/target-docker \
  ghcr.io/andymai/occt-wasm-builder:latest \
  bash -c "cargo xtask build-wasi --release --minimal"
```

`target-docker/` is that container's output directory: root-owned, ~145 MB, and
`.gitignore`'s `target/` does not match it. It is ignored explicitly now.
