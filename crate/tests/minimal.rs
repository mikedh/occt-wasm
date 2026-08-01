//! Load-verification for the `--minimal` kernel blob.
//!
//! The minimal profile is the construction-op surface (see
//! `config::OPTIONAL_CATEGORIES`); the crate binds everything else lazily.
//! This file asserts the contract from the consumer side:
//!
//! 1. A minimal blob INSTANTIATES — i.e. every eagerly-bound export (generated
//!    core wrappers plus the hand-written infra/accessor bindings in
//!    `kernel.rs`) survives the minimal build. This is the invariant that was
//!    silently broken when the export list was maintained by hand.
//! 2. Core operations work end-to-end via rmesh's real call path
//!    (lines → wire → face → extrude → boolean → tessellate → IR lift),
//!    plus fillet/chamfer, the boolean-reliability primitives
//!    (`boolean_pipeline`, `boolean_fuzzy`, `is_valid`, `fix_shape`), and the
//!    out-of-tree facade extension (`to_brep_ir`).
//! 3. Optional-capability methods fail with `OcctError::MissingCapability`,
//!    not a trap or an instantiation error.
//!
//! The blob is read from `OCCT_WASM_MINIMAL` when set, else from
//! `dist/occt-wasm-minimal.wasm.br` (where `cargo xtask build-wasi --minimal`
//! installs it). Missing blob skips, mirroring `integration.rs`.

#![allow(clippy::unwrap_used, clippy::panic)]

use occt_wasm::{OcctError, OcctKernel, ShapeHandle};

/// The minimal blob's compressed bytes, or None to skip (absent blob, or
/// debug mode where WASM compilation is ~100x slower).
fn try_minimal_bytes() -> Option<Vec<u8>> {
    if cfg!(debug_assertions) {
        eprintln!(
            "Skipping test: WASM compilation too slow in debug mode. Use `cargo test --release`."
        );
        return None;
    }
    let path = std::env::var_os("OCCT_WASM_MINIMAL").map_or_else(
        || {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../dist/occt-wasm-minimal.wasm.br")
        },
        std::path::PathBuf::from,
    );
    match std::fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(_) => {
            eprintln!(
                "Skipping test: no minimal blob at {}. Run `cargo xtask build-wasi --release --minimal`.",
                path.display()
            );
            None
        }
    }
}

/// A `size`-cube at `origin` built the way rmesh builds every solid: line
/// edges → wire → face → extrude. No primitives category required.
fn constructed_cube(kernel: &mut OcctKernel, x: f64, y: f64, size: f64) -> ShapeHandle {
    let s = size;
    let e1 = kernel.make_line_edge(x, y, 0.0, x + s, y, 0.0).unwrap();
    let e2 = kernel
        .make_line_edge(x + s, y, 0.0, x + s, y + s, 0.0)
        .unwrap();
    let e3 = kernel
        .make_line_edge(x + s, y + s, 0.0, x, y + s, 0.0)
        .unwrap();
    let e4 = kernel.make_line_edge(x, y + s, 0.0, x, y, 0.0).unwrap();
    let wire = kernel.make_wire(&[e1, e2, e3, e4]).unwrap();
    let face = kernel.make_face(wire).unwrap();
    kernel.extrude(face, 0.0, 0.0, s).unwrap()
}

/// The load-bearing assertion: a minimal blob must instantiate. Every export
/// the crate binds eagerly has to exist in it; a failure here means the
/// derived export set and the crate's binding surface have diverged.
#[test]
fn minimal_blob_instantiates() {
    let Some(bytes) = try_minimal_bytes() else {
        return;
    };
    let kernel = OcctKernel::from_compressed_module_bytes(&bytes);
    assert!(
        kernel.is_ok(),
        "minimal blob failed to instantiate: {:?}",
        kernel.err()
    );
}

#[test]
fn minimal_blob_construction_path_works() {
    let Some(bytes) = try_minimal_bytes() else {
        return;
    };
    let mut kernel = OcctKernel::from_compressed_module_bytes(&bytes).unwrap();

    // rmesh's real lowering path.
    let solid = constructed_cube(&mut kernel, 0.0, 0.0, 10.0);

    let mesh = kernel.tessellate_relative(solid, 0.01, 0.5).unwrap();
    assert!(!mesh.positions.is_empty());

    // The facade extension exports (facade/src/rmesh_brep.cpp) must survive
    // the minimal build — they are scraped outside the derived core set.
    let (topology, geometry) = kernel.to_brep_ir(solid).unwrap();
    assert!(!topology.is_empty() && !geometry.is_empty());

    // Fillet/chamfer are core by policy; they need topology's sub-shape
    // enumeration, which is why `topology` is core too.
    let edges: Vec<ShapeHandle> = kernel
        .get_sub_shapes(solid, "edge")
        .unwrap()
        .into_iter()
        .map(ShapeHandle::from_raw)
        .collect();
    assert!(!edges.is_empty());
    let filleted = kernel.fillet(solid, &edges[..1], 1.0).unwrap();
    assert!(kernel.is_valid(filleted).unwrap());
}

#[test]
fn minimal_blob_boolean_reliability_primitives_work() {
    let Some(bytes) = try_minimal_bytes() else {
        return;
    };
    let mut kernel = OcctKernel::from_compressed_module_bytes(&bytes).unwrap();

    // Overlapping cubes: 10-cube at origin, 10-cube shifted by (5, 5).
    let a = constructed_cube(&mut kernel, 0.0, 0.0, 10.0);
    let b = constructed_cube(&mut kernel, 5.0, 5.0, 10.0);

    // Rung 1: pipeline single-step cut (includes the UnifySameDomain refine).
    let cut = kernel.boolean_pipeline(a, &[1], &[b]).unwrap();
    assert!(kernel.is_valid(cut).unwrap());

    // Rung 2: fuzzy retry entry point.
    let fuzzy = kernel.boolean_fuzzy(a, b, 1, 1e-6).unwrap();
    assert!(kernel.is_valid(fuzzy).unwrap());

    // Rungs 3-4: the validity gate and repair must exist on minimal blobs.
    let fixed = kernel.fix_shape(cut).unwrap();
    assert!(kernel.is_valid(fixed).unwrap());
}

#[test]
fn minimal_blob_reports_missing_capabilities() {
    let Some(bytes) = try_minimal_bytes() else {
        return;
    };
    let mut kernel = OcctKernel::from_compressed_module_bytes(&bytes).unwrap();
    let solid = constructed_cube(&mut kernel, 0.0, 0.0, 10.0);

    let missing = |r: Result<(), OcctError>, what: &str| {
        assert!(
            matches!(r, Err(OcctError::MissingCapability(_))),
            "{what} should be MissingCapability on the minimal blob"
        );
    };

    // One probe per optional subsystem.
    missing(kernel.make_box(1.0, 1.0, 1.0).map(drop), "primitives");
    missing(kernel.get_volume(solid).map(drop), "query");
    missing(kernel.offset(solid, 1.0, 1e-4).map(drop), "offsetting");
    missing(kernel.curve_length(solid).map(drop), "curve");
    missing(
        kernel.fuse_with_history(solid, solid, &[], 0).map(drop),
        "evolution",
    );
    missing(kernel.to_brep(solid).map(drop), "io");
    missing(kernel.import_step("not step data").map(drop), "exchange");
    missing(
        kernel
            .project_edges(solid, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, false)
            .map(drop),
        "projection",
    );
    missing(kernel.xcaf_new_document().map(drop), "xcaf");
}
