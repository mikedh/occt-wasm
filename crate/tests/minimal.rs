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

/// The minimal blob's compressed bytes. **Absence is a failure, never a skip.**
///
/// This used to return `Option` and every test opened with
/// `let Some(..) else { return; }`, so `cargo test -p occt-wasm` passed without
/// executing a single line of `rmesh_history.cpp` — in debug mode always, and in
/// release whenever the blob was missing. A suite that reports green without the
/// thing it exists to test is worse than no suite.
///
/// Debug is still refused rather than run: wasm compilation there is ~100x
/// slower, so running it would be a different kind of lie (a suite nobody waits
/// for). But it refuses LOUDLY and says what to do, instead of passing.
fn minimal_bytes() -> Vec<u8> {
    // The condition IS constant, and that is the point: this build either can
    // run the kernel or it cannot, and the answer is known at compile time.
    // Clippy's `assertions_on_constants` assumes a constant assertion is a
    // mistake; here it is the mechanism.
    #[allow(clippy::assertions_on_constants)]
    {
        assert!(
            !cfg!(debug_assertions),
            "the kernel tests need `--release`: wasm compilation is ~100x slower \
             in debug. Run `cargo test --release -p occt-wasm`."
        );
    }
    let path = std::env::var_os("OCCT_WASM_MINIMAL").map_or_else(
        || {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../dist/occt-wasm-minimal.wasm.br")
        },
        std::path::PathBuf::from,
    );
    std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "no minimal blob at {} ({error}). Run \
             `cargo xtask build-wasi --release --minimal`.",
            path.display()
        )
    })
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
    let bytes = minimal_bytes();
    let kernel = OcctKernel::from_compressed_module_bytes(&bytes);
    assert!(
        kernel.is_ok(),
        "minimal blob failed to instantiate: {:?}",
        kernel.err()
    );
}

#[test]
fn minimal_blob_construction_path_works() {
    let bytes = minimal_bytes();
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
    let bytes = minimal_bytes();
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

    // `check_shape` is the self-intersection half of the gate — BRepCheck
    // does not look for crossing faces, so a caller needs both.
    assert!(kernel.check_shape(fixed).unwrap());
}

#[test]
fn minimal_blob_reports_missing_capabilities() {
    let bytes = minimal_bytes();
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
    missing(kernel.to_brep(solid).map(drop), "io");
    missing(kernel.import_step("not step data").map(drop), "exchange");
    missing(
        kernel
            .project_edges(solid, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, false)
            .map(drop),
        "projection",
    );
    missing(kernel.xcaf_new_document().map(drop), "xcaf");

    // `evolution` is deliberately NOT on this list — see the assertion below.
}

/// The index-keyed history exports work end to end, on the profile the app
/// ships.
///
/// This is the smoke test for `facade/src/rmesh_history.cpp`. It asserts the
/// SHAPE of the six claim channels, not their content — the embedding that
/// defines the enumeration is the one that gives the relations meaning, so the
/// semantics are pinned in rmesh.
///
/// There is no framing to validate any more. The predecessor of this test walked
/// a record stream checking headers, per-record lengths and a declared record
/// count; all of that went away with the format. What is left is the only thing
/// a flat pair-array can get wrong: an odd length, or an index outside the
/// counts.
fn check_channel(pairs: &[u32], sources: u32, results: u32, label: &str) {
    assert_eq!(
        pairs.len() % 2,
        0,
        "{label}: {} entries is not a whole number of (source, result) pairs",
        pairs.len()
    );
    for pair in pairs.chunks_exact(2) {
        assert!(
            pair[0] < sources,
            "{label}: source {} but the input has {sources}",
            pair[0]
        );
        assert!(
            pair[1] < results,
            "{label}: result {} but the result has {results}",
            pair[1]
        );
    }
}

fn check_history(history: &occt_wasm::ShapeHistoryData, label: &str) {
    let (in_f, out_f) = (history.input_faces, history.result_faces);
    let (in_e, out_e) = (history.input_edges, history.result_edges);
    check_channel(
        &history.modified_faces,
        in_f,
        out_f,
        &format!("{label}/mod-faces"),
    );
    check_channel(
        &history.generated_faces,
        in_f,
        out_f,
        &format!("{label}/gen-faces"),
    );
    check_channel(
        &history.modified_edges,
        in_e,
        out_e,
        &format!("{label}/mod-edges"),
    );
    check_channel(
        &history.generated_edges,
        in_e,
        out_e,
        &format!("{label}/gen-edges"),
    );
    check_channel(
        &history.faces_from_edges,
        in_e,
        out_f,
        &format!("{label}/face-from-edge"),
    );
    check_channel(
        &history.edges_from_faces,
        in_f,
        out_e,
        &format!("{label}/edge-from-face"),
    );
}

#[test]
fn minimal_blob_reports_index_keyed_history() {
    let bytes = minimal_bytes();
    let mut kernel = OcctKernel::from_compressed_module_bytes(&bytes).unwrap();

    // A fuse of two overlapping cubes: both operands are inputs, so the counts
    // must describe the pair, not just the first.
    let a = constructed_cube(&mut kernel, 0.0, 0.0, 10.0);
    let b = constructed_cube(&mut kernel, 5.0, 0.0, 10.0);
    let fused = kernel.history_boolean(a, b, 0, -1.0, true).unwrap();
    assert_ne!(fused.result_id, 0, "the fuse must produce a shape");
    check_history(&fused, "fuse");
    assert_eq!(
        fused.input_faces, 12,
        "two cubes are twelve input faces: the history domain is BOTH operands, \
         and a count of 6 would mean only the first was enumerated"
    );
    assert!(
        fused.result_faces > 0 && fused.result_edges > 0,
        "the fused result has faces and edges"
    );
    assert!(
        !fused.modified_faces.is_empty(),
        "a fuse leaves most of both operands' faces in place, so something must \
         be claimed as still-that-face"
    );

    // A fillet: the interesting relation is a FACE grown from an EDGE, which is
    // the blend surface. Nothing else in the kernel reports that, and it is the
    // one channel that cannot be expressed by any same-kind relation.
    let solid = constructed_cube(&mut kernel, 0.0, 0.0, 10.0);
    let edges: Vec<ShapeHandle> = kernel
        .get_sub_shapes(solid, "edge")
        .unwrap()
        .into_iter()
        .map(ShapeHandle::from_raw)
        .collect();
    let blended = kernel.history_blend(solid, &edges[..1], 1.0, 0).unwrap();
    assert_ne!(blended.result_id, 0, "the fillet must produce a shape");
    check_history(&blended, "fillet");
    assert!(
        !blended.faces_from_edges.is_empty(),
        "a fillet grows its blend surface along an edge, and that cross-kind \
         relation is the whole reason edges are enumerated here"
    );
    assert!(
        !blended.modified_edges.is_empty(),
        "a one-edge fillet leaves the body's other edges alone, so they must be \
         claimed as still-themselves"
    );
}

/// A bad op code reaches the error half of the ABI, which nothing else does.
///
/// Five things are unexecuted without this test: the `default:` arm, the
/// `catch (const std::exception&)` block, `occt_rmesh_history_error`,
/// `occt_rmesh_history_error_len`, and the `OcctError::Operation` the reader
/// builds from them. rmesh cannot reach any of them — its op codes come from a
/// three-variant enum — so the coverage has to come from here.
#[test]
fn an_unknown_op_code_reports_through_the_error_channel() {
    let bytes = minimal_bytes();
    let mut kernel = OcctKernel::from_compressed_module_bytes(&bytes).unwrap();
    let a = constructed_cube(&mut kernel, 0.0, 0.0, 10.0);
    let b = constructed_cube(&mut kernel, 5.0, 0.0, 10.0);

    let outcome = kernel.history_boolean(a, b, 7, -1.0, true);
    let Err(OcctError::Operation { operation, message }) = outcome else {
        panic!("an unknown op code must be an Operation error, got {outcome:?}");
    };
    assert_eq!(operation, "history_boolean");
    assert!(
        message.contains("unknown boolean op code"),
        "the message must name the fault, got {message:?}"
    );

    // The kernel is still usable: an error path that left the arena or the
    // globals inconsistent would show up as the next call failing.
    let fused = kernel
        .history_boolean(a, b, 0, -1.0, true)
        .expect("the kernel survives a rejected op code");
    assert_ne!(fused.result_id, 0);
}

/// The same for a blend kind rmesh's two-variant enum cannot produce.
#[test]
fn an_unknown_blend_kind_is_refused() {
    let bytes = minimal_bytes();
    let mut kernel = OcctKernel::from_compressed_module_bytes(&bytes).unwrap();
    let solid = constructed_cube(&mut kernel, 0.0, 0.0, 10.0);
    let edges: Vec<ShapeHandle> = kernel
        .get_sub_shapes(solid, "edge")
        .unwrap()
        .into_iter()
        .map(ShapeHandle::from_raw)
        .collect();

    let outcome = kernel.history_blend(solid, &edges[..1], 1.0, 9);
    assert!(
        matches!(&outcome, Err(OcctError::Operation { message, .. })
                 if message.contains("unknown blend kind")),
        "got {outcome:?}"
    );

    // An empty edge list is the other argument fault, and it must be refused
    // before any kernel work rather than producing an unblended copy.
    let empty = kernel.history_blend(solid, &[], 1.0, 0);
    assert!(
        matches!(&empty, Err(OcctError::Operation { message, .. })
                 if message.contains("no edges")),
        "got {empty:?}"
    );
}

/// The HASH-KEYED history builders are absent from the minimal blob, while the
/// capability they were kept for is present.
///
/// This assertion has now been written in both directions, and the reason is
/// worth keeping. `evolution` was optional; it was made core because a stable
/// NAME for a face cannot be derived from geometry alone, and naming is a
/// construction-op concern. That argument was right and is unchanged — it just
/// stopped pointing at these builders once `rmesh_history.cpp` answered the same
/// question by index, for edges too, and in every profile.
///
/// So what is asserted is the pair: the twelve hash-keyed builders are gone from
/// the blob the app ships, and provenance is still reachable. Asserting only the
/// first would let the capability regress silently; asserting only the second
/// would let twelve dead exports (and the offset family they root) creep back
/// in.
#[test]
fn the_minimal_blob_drops_the_hash_keyed_builders_but_keeps_provenance() {
    let bytes = minimal_bytes();
    let mut kernel = OcctKernel::from_compressed_module_bytes(&bytes).unwrap();
    let a = constructed_cube(&mut kernel, 0.0, 0.0, 10.0);
    let b = constructed_cube(&mut kernel, 5.0, 0.0, 10.0);

    let outcome = kernel.fuse_with_history(a, b, &[], 0);
    assert!(
        matches!(outcome, Err(OcctError::MissingCapability(_))),
        "the hash-keyed builders must be optional, got {outcome:?}"
    );

    // ...and the thing they were kept core for still works.
    let reported = kernel
        .history_boolean(a, b, 0, -1.0, true)
        .expect("index-keyed provenance is core in every profile");
    assert_ne!(reported.result_id, 0);
    assert!(reported.input_faces > 0 && reported.result_faces > 0);
}
