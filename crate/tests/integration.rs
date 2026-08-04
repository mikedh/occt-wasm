//! Integration tests for the occt-wasm Rust crate.
//!
//! These tests require a real WASI WASM binary at `crate/src/occt-wasm.wasm.br`.
//! They are skipped (not failed) when the binary is a placeholder.
//!
//! To run: `cargo xtask build-wasi && cargo test -p occt-wasm`

#![allow(clippy::unwrap_used, clippy::panic)]

use occt_wasm::OcctKernel;

/// Try to create a kernel. Returns None if the embedded WASM is a placeholder
/// The kernel, or a loud failure. **Absence is never a skip.**
///
/// This returned `Option` and every test opened with
/// `let Some(..) else { return; }`, so the suite reported green without ever
/// instantiating a kernel — in debug always, and in release whenever the
/// embedded module was a placeholder. Refusing loudly says what to do; passing
/// says nothing at all.
fn kernel() -> OcctKernel {
    assert!(
        !cfg!(debug_assertions),
        "the kernel tests need `--release`: wasm compilation is ~100x slower in \
         debug. Run `cargo test --release -p occt-wasm`."
    );
    OcctKernel::new().unwrap_or_else(|error| {
        panic!(
            "the embedded kernel module must instantiate ({error}). Run \
             `cargo xtask build-wasi --release`."
        )
    })
}

#[test]
fn kernel_init_and_drop() {
    let _kernel = kernel();
    // Kernel initializes and drops cleanly
}

#[test]
fn make_box() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 20.0, 30.0).unwrap();
    assert!(shape.id() > 0);
}

#[test]
fn box_volume() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 20.0, 30.0).unwrap();
    let vol = kernel.get_volume(shape).unwrap();
    assert!((vol - 6000.0).abs() < 1.0, "expected ~6000, got {vol}");
}

#[test]
fn fuse_two_boxes() {
    let mut kernel = kernel();
    let a = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let b = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let result = kernel.fuse(a, b).unwrap();
    let vol = kernel.get_volume(result).unwrap();
    // Fuse of identical boxes at origin = same volume (overlapping)
    assert!((vol - 1000.0).abs() < 1.0, "expected ~1000, got {vol}");
}

#[test]
fn cut_operation() {
    let mut kernel = kernel();
    let box_shape = kernel.make_box(20.0, 20.0, 20.0).unwrap();
    let sphere = kernel.make_sphere(5.0).unwrap();
    let result = kernel.cut(box_shape, sphere).unwrap();
    let vol = kernel.get_volume(result).unwrap();
    // Box volume - sphere volume (partially inside)
    assert!(vol > 0.0 && vol < 8000.0, "unexpected volume: {vol}");
}

#[test]
fn bounding_box() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 20.0, 30.0).unwrap();
    let bbox = kernel.get_bounding_box(shape, true).unwrap();
    assert!((bbox.min.x).abs() < 0.01);
    assert!((bbox.min.y).abs() < 0.01);
    assert!((bbox.min.z).abs() < 0.01);
    assert!((bbox.max.x - 10.0).abs() < 0.01);
    assert!((bbox.max.y - 20.0).abs() < 0.01);
    assert!((bbox.max.z - 30.0).abs() < 0.01);
}

#[test]
fn tessellate_box() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let mesh = kernel.tessellate(shape, 0.1, 0.5).unwrap();
    // A box has at least 8 vertices and 12 triangles
    assert!(
        mesh.positions.len() >= 24,
        "expected at least 24 position floats, got {}",
        mesh.positions.len()
    );
    assert!(
        mesh.indices.len() >= 36,
        "expected at least 36 index values, got {}",
        mesh.indices.len()
    );
    assert_eq!(mesh.positions.len(), mesh.normals.len());
}

#[test]
fn step_roundtrip() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 20.0, 30.0).unwrap();
    let step_data = kernel.export_step(shape).unwrap();
    assert!(step_data.contains("STEP"), "expected STEP format data");

    let imported = kernel.import_step(&step_data).unwrap();
    let vol = kernel.get_volume(imported).unwrap();
    assert!(
        (vol - 6000.0).abs() < 1.0,
        "roundtrip volume mismatch: {vol}"
    );
}

#[test]
fn make_cylinder_and_query() {
    let mut kernel = kernel();
    let cyl = kernel.make_cylinder(5.0, 10.0).unwrap();
    let vol = kernel.get_volume(cyl).unwrap();
    let expected = std::f64::consts::PI * 25.0 * 10.0;
    assert!(
        (vol - expected).abs() < 1.0,
        "expected ~{expected:.0}, got {vol}"
    );
}

#[test]
fn make_sphere_surface_area() {
    let mut kernel = kernel();
    let sphere = kernel.make_sphere(5.0).unwrap();
    let area = kernel.get_surface_area(sphere).unwrap();
    let expected = 4.0 * std::f64::consts::PI * 25.0;
    assert!(
        (area - expected).abs() < 1.0,
        "expected ~{expected:.0}, got {area}"
    );
}

#[test]
fn release_shape() {
    let mut kernel = kernel();
    let shape = kernel.make_box(1.0, 1.0, 1.0).unwrap();
    kernel.release(shape).unwrap();
    // After release, the handle is invalid
}

#[test]
fn get_shape_type() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let shape_type = kernel.get_shape_type(shape).unwrap();
    assert_eq!(shape_type, "Solid", "box should be a Solid");
}

#[test]
fn get_sub_shapes() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let faces = kernel.get_sub_shapes(shape, "Face").unwrap();
    assert_eq!(faces.len(), 6, "box should have 6 faces");
}

#[test]
fn sub_shape_count_and_hashes() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    assert_eq!(kernel.sub_shape_count(shape, "face").unwrap(), 6);
    let hashes = kernel.sub_shape_hashes(shape, "face", 1_000_000).unwrap();
    assert_eq!(hashes.len(), 6, "one hash per face, no handles allocated");
}

#[test]
fn checkpoint_release_since() {
    let mut kernel = kernel();
    let _keep = kernel.make_box(1.0, 1.0, 1.0).unwrap();
    let mark = kernel.checkpoint().unwrap();
    let _a = kernel.make_box(2.0, 2.0, 2.0).unwrap();
    let _b = kernel.make_box(3.0, 3.0, 3.0).unwrap();
    assert_eq!(kernel.get_shape_count().unwrap(), 3);
    kernel.release_since(mark).unwrap();
    // Only the pre-checkpoint handle survives.
    assert_eq!(kernel.get_shape_count().unwrap(), 1);
}

#[test]
fn extrude_rectangle() {
    let mut kernel = kernel();
    let rect = kernel.make_rectangle(10.0, 20.0).unwrap();
    let solid = kernel.extrude(rect, 0.0, 0.0, 5.0).unwrap();
    let vol = kernel.get_volume(solid).unwrap();
    assert!(
        (vol - 1000.0).abs() < 1.0,
        "expected 10*20*5=1000, got {vol}"
    );
}
