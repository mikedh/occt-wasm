//! WASM host implementation for the OCCT kernel.
//!
//! Manages the wasmtime `Engine`, `Store`, and `Instance`, and provides
//! helper methods for memory transfer across the WASM boundary.

use wasmtime::{Engine, Instance, Linker, Memory, Module, Store, TypedFunc};

use crate::error::{OcctError, OcctResult};
use crate::kernel_generated::GeneratedFuncs;
use crate::types::{
    BoundingBox, EdgeData, EvolutionData, LabelInfo, Mesh, MeshBatch, NurbsCurveData,
    ProjectionData, ShapeHandle, Vec3,
};

/// Brotli-compressed WASM binary, embedded at compile time.
///
/// The uncompressed binary is ~21 MB; brotli brings it to ~4 MB.
/// This is decompressed once during `OcctKernel::new()`. Disable the
/// `embed-module` feature to keep kernel bytes out of your binary entirely and
/// supply the module at runtime via [`OcctKernel::from_module_bytes`].
#[cfg(feature = "embed-module")]
static WASM_BINARY: &[u8] = include_bytes!("occt-wasm.wasm.br");

/// The OCCT CAD kernel, backed by a sandboxed WASM module.
///
/// Create an instance with [`OcctKernel::new()`], then call methods to
/// create and manipulate shapes. All shapes live in an arena inside the
/// WASM module and are referenced by [`ShapeHandle`].
///
/// # Example
///
/// ```no_run
/// use occt_wasm::{OcctKernel, ShapeHandle};
///
/// let mut kernel = OcctKernel::new().unwrap();
/// let box_shape = kernel.make_box(10.0, 20.0, 30.0).unwrap();
/// let volume = kernel.get_volume(box_shape).unwrap();
/// assert!((volume - 6000.0).abs() < 1.0);
/// ```
pub struct OcctKernel {
    pub(crate) store: Store<()>,
    pub(crate) instance: Instance,
    pub(crate) memory: Memory,

    // Lifecycle + error functions
    pub(crate) fn_has_error: TypedFunc<(), i32>,
    pub(crate) fn_get_error: TypedFunc<(), i32>,
    pub(crate) fn_get_error_len: TypedFunc<(), u32>,

    // Memory management
    pub(crate) fn_alloc: TypedFunc<u32, u32>,
    pub(crate) fn_free: TypedFunc<u32, ()>,

    // Result buffer accessors
    pub(crate) fn_get_string_result: TypedFunc<(), i32>,
    pub(crate) fn_get_string_result_len: TypedFunc<(), u32>,
    pub(crate) fn_get_vec_u32_result: TypedFunc<(), i32>,
    pub(crate) fn_get_vec_u32_result_len: TypedFunc<(), u32>,
    pub(crate) fn_get_vec_f64_result: TypedFunc<(), i32>,
    pub(crate) fn_get_vec_f64_result_len: TypedFunc<(), u32>,
    pub(crate) fn_get_vec_i32_result: TypedFunc<(), i32>,
    pub(crate) fn_get_vec_i32_result_len: TypedFunc<(), u32>,

    // BBox accessors
    pub(crate) fn_get_bbox_xmin: TypedFunc<(), f64>,
    pub(crate) fn_get_bbox_ymin: TypedFunc<(), f64>,
    pub(crate) fn_get_bbox_zmin: TypedFunc<(), f64>,
    pub(crate) fn_get_bbox_xmax: TypedFunc<(), f64>,
    pub(crate) fn_get_bbox_ymax: TypedFunc<(), f64>,
    pub(crate) fn_get_bbox_zmax: TypedFunc<(), f64>,

    // Mesh accessors
    pub(crate) fn_get_mesh_positions: TypedFunc<(), i32>,
    pub(crate) fn_get_mesh_positions_len: TypedFunc<(), i32>,
    pub(crate) fn_get_mesh_normals: TypedFunc<(), i32>,
    pub(crate) fn_get_mesh_normals_len: TypedFunc<(), i32>,
    pub(crate) fn_get_mesh_indices: TypedFunc<(), i32>,
    pub(crate) fn_get_mesh_indices_len: TypedFunc<(), i32>,
    pub(crate) fn_get_mesh_face_groups: TypedFunc<(), i32>,
    pub(crate) fn_get_mesh_face_groups_len: TypedFunc<(), i32>,

    // MeshBatch accessors
    pub(crate) fn_get_mesh_batch_positions: TypedFunc<(), i32>,
    pub(crate) fn_get_mesh_batch_positions_len: TypedFunc<(), i32>,
    pub(crate) fn_get_mesh_batch_normals: TypedFunc<(), i32>,
    pub(crate) fn_get_mesh_batch_normals_len: TypedFunc<(), i32>,
    pub(crate) fn_get_mesh_batch_indices: TypedFunc<(), i32>,
    pub(crate) fn_get_mesh_batch_indices_len: TypedFunc<(), i32>,
    pub(crate) fn_get_mesh_batch_shape_offsets: TypedFunc<(), i32>,
    pub(crate) fn_get_mesh_batch_shape_count: TypedFunc<(), i32>,

    // Edge accessors
    pub(crate) fn_get_edge_points: TypedFunc<(), i32>,
    pub(crate) fn_get_edge_points_len: TypedFunc<(), i32>,
    pub(crate) fn_get_edge_groups: TypedFunc<(), i32>,
    pub(crate) fn_get_edge_groups_len: TypedFunc<(), i32>,

    // NURBS accessors
    pub(crate) fn_get_nurbs_degree: TypedFunc<(), i32>,
    pub(crate) fn_get_nurbs_rational: TypedFunc<(), i32>,
    pub(crate) fn_get_nurbs_periodic: TypedFunc<(), i32>,
    pub(crate) fn_get_nurbs_knots: TypedFunc<(), i32>,
    pub(crate) fn_get_nurbs_knots_len: TypedFunc<(), u32>,
    pub(crate) fn_get_nurbs_multiplicities: TypedFunc<(), i32>,
    pub(crate) fn_get_nurbs_multiplicities_len: TypedFunc<(), u32>,
    pub(crate) fn_get_nurbs_poles: TypedFunc<(), i32>,
    pub(crate) fn_get_nurbs_poles_len: TypedFunc<(), u32>,
    pub(crate) fn_get_nurbs_weights: TypedFunc<(), i32>,
    pub(crate) fn_get_nurbs_weights_len: TypedFunc<(), u32>,

    // Evolution accessors
    pub(crate) fn_get_evo_result_id: TypedFunc<(), u32>,
    pub(crate) fn_get_evo_modified: TypedFunc<(), i32>,
    pub(crate) fn_get_evo_modified_len: TypedFunc<(), u32>,
    pub(crate) fn_get_evo_generated: TypedFunc<(), i32>,
    pub(crate) fn_get_evo_generated_len: TypedFunc<(), u32>,
    pub(crate) fn_get_evo_deleted: TypedFunc<(), i32>,
    pub(crate) fn_get_evo_deleted_len: TypedFunc<(), u32>,

    // Projection accessors
    pub(crate) fn_get_proj_visible_outline: TypedFunc<(), u32>,
    pub(crate) fn_get_proj_visible_smooth: TypedFunc<(), u32>,
    pub(crate) fn_get_proj_visible_sharp: TypedFunc<(), u32>,
    pub(crate) fn_get_proj_hidden_outline: TypedFunc<(), u32>,
    pub(crate) fn_get_proj_hidden_smooth: TypedFunc<(), u32>,
    pub(crate) fn_get_proj_hidden_sharp: TypedFunc<(), u32>,

    // Label info accessors
    pub(crate) fn_get_label_info_label_id: TypedFunc<(), i32>,
    pub(crate) fn_get_label_info_name: TypedFunc<(), i32>,
    pub(crate) fn_get_label_info_name_len: TypedFunc<(), u32>,
    pub(crate) fn_get_label_info_has_color: TypedFunc<(), i32>,
    pub(crate) fn_get_label_info_r: TypedFunc<(), f64>,
    pub(crate) fn_get_label_info_g: TypedFunc<(), f64>,
    pub(crate) fn_get_label_info_b: TypedFunc<(), f64>,
    pub(crate) fn_get_label_info_is_assembly: TypedFunc<(), i32>,
    pub(crate) fn_get_label_info_is_component: TypedFunc<(), i32>,
    pub(crate) fn_get_label_info_shape_id: TypedFunc<(), u32>,

    // Generated method handles
    pub(crate) generated: GeneratedFuncs,
}

impl OcctKernel {
    /// Create a new OCCT kernel instance from the embedded WASM binary.
    ///
    /// Decompresses the embedded WASM binary, compiles it with `wasmtime`,
    /// and initializes the OCCT runtime. This takes ~100-500ms depending
    /// on the platform.
    #[cfg(feature = "embed-module")]
    pub fn new() -> OcctResult<Self> {
        Self::from_compressed_module_bytes(WASM_BINARY)
    }

    /// Create a kernel from caller-supplied brotli-compressed module bytes
    /// (the format shipped as `occt-wasm.wasm.br`).
    pub fn from_compressed_module_bytes(compressed: &[u8]) -> OcctResult<Self> {
        let mut wasm_bytes = Vec::new();
        let mut input = compressed;
        brotli::BrotliDecompress(&mut input, &mut wasm_bytes)
            .map_err(|e| OcctError::Memory(format!("brotli decompression failed: {e}")))?;
        Self::from_module_bytes(&wasm_bytes)
    }

    /// Create a kernel from caller-supplied uncompressed WASM module bytes.
    ///
    /// This is the custody-free constructor: with `default-features = false`
    /// the crate embeds no kernel bytes and the application decides where the
    /// module comes from (a data dir, a lazy fetch, a test fixture).
    pub fn from_module_bytes(wasm_bytes: &[u8]) -> OcctResult<Self> {
        let engine = Engine::new(&Self::engine_config())?;
        let module = Module::new(&engine, wasm_bytes)?;
        Self::from_module(&engine, &module)
    }

    /// Ahead-of-time compile module bytes into wasmtime's serialized form for
    /// [`OcctKernel::from_precompiled_file`]. Compiling takes seconds; loading
    /// the precompiled artifact takes milliseconds. The artifact is
    /// wasmtime-version-specific — cache it keyed on the module hash and
    /// rebuild on mismatch.
    pub fn precompile_module(wasm_bytes: &[u8]) -> OcctResult<Vec<u8>> {
        let engine = Engine::new(&Self::engine_config())?;
        Ok(engine.precompile_module(wasm_bytes)?)
    }

    /// Create a kernel from a [`OcctKernel::precompile_module`] artifact on
    /// disk (~ms instead of the multi-second JIT).
    // SAFETY-BOUNDARY: `Module::deserialize_file` is `unsafe` because a
    // corrupted or attacker-controlled artifact is undefined behavior. The
    // contract here is that callers pass only paths they wrote themselves via
    // `precompile_module` (a private cache dir keyed on content hash).
    #[allow(unsafe_code)]
    pub fn from_precompiled_file(path: &std::path::Path) -> OcctResult<Self> {
        let engine = Engine::new(&Self::engine_config())?;
        // SAFETY: see SAFETY-BOUNDARY above — the artifact must originate from
        // `precompile_module` on this host.
        let module = unsafe { Module::deserialize_file(&engine, path)? };
        Self::from_module(&engine, &module)
    }

    fn engine_config() -> wasmtime::Config {
        let mut config = wasmtime::Config::new();
        config.wasm_simd(true);
        config.wasm_tail_call(true);
        // The WASM binary uses wasm-opt --experimental-new-eh to convert
        // Emscripten's legacy exceptions to the new (exnref) encoding.
        config.wasm_exceptions(true);
        // Epoch interruption is near-zero-cost when unused and lets a host
        // kill a runaway kernel operation (a stuck boolean) by ticking the
        // engine's epoch from another thread — see `set_epoch_deadline` /
        // `engine_handle`. The store starts with an effectively-infinite
        // deadline so hosts that never tick see no behavior change.
        config.epoch_interruption(true);
        config
    }

    /// Arm the epoch kill switch: the current operation traps once the
    /// engine's epoch advances `ticks_from_now` past its current value.
    /// Pair with a thread calling [`wasmtime::Engine::increment_epoch`] on
    /// the handle from [`OcctKernel::engine_handle`] at a fixed cadence.
    pub fn set_epoch_deadline(&mut self, ticks_from_now: u64) {
        self.store.set_epoch_deadline(ticks_from_now);
    }

    /// A cheap clone of the underlying wasmtime engine, for an epoch-ticker
    /// thread. Cloning shares the engine; it does not copy compiled code.
    pub fn engine_handle(&self) -> Engine {
        self.store.engine().clone()
    }

    /// Instantiate a compiled module and initialize the OCCT runtime.
    #[allow(clippy::too_many_lines)]
    fn from_module(engine: &Engine, module: &Module) -> OcctResult<Self> {
        let mut store = Store::new(engine, ());
        // Effectively-infinite default: epoch interruption only bites when a
        // host arms `set_epoch_deadline` and ticks the engine.
        store.set_epoch_deadline(u64::MAX);
        let mut linker = Linker::new(engine);
        define_host_imports(&mut linker, module)?;
        let instance = linker.instantiate(&mut store, module)?;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| OcctError::Memory("no memory export".to_owned()))?;

        // Run emscripten-standalone constructors (vtables, static init) before
        // any other export — without this the first kernel call traps with
        // "uninitialized element".
        if let Some(initialize) = instance.get_func(&mut store, "_initialize") {
            initialize.typed::<(), ()>(&store)?.call(&mut store, ())?;
        }

        // Call occt_init
        let init: TypedFunc<(), i32> = instance.get_typed_func(&mut store, "occt_init")?;
        let result = init.call(&mut store, ())?;
        if result != 0 {
            return Err(OcctError::Memory("occt_init failed".to_owned()));
        }

        // Resolve all accessor functions
        macro_rules! get_fn {
            ($name:expr => $ret:ty) => {
                instance.get_typed_func::<(), $ret>(&mut store, $name)?
            };
            ($name:expr, $param:ty => $ret:ty) => {
                instance.get_typed_func::<$param, $ret>(&mut store, $name)?
            };
        }

        let generated = GeneratedFuncs::resolve(&instance, &mut store)?;

        Ok(Self {
            memory,
            instance,

            fn_has_error: get_fn!("occt_has_error" => i32),
            fn_get_error: get_fn!("occt_get_error" => i32),
            fn_get_error_len: get_fn!("occt_get_error_len" => u32),
            fn_alloc: get_fn!("occt_alloc", u32 => u32),
            fn_free: get_fn!("occt_free", u32 => ()),

            fn_get_string_result: get_fn!("occt_get_string_result" => i32),
            fn_get_string_result_len: get_fn!("occt_get_string_result_len" => u32),
            fn_get_vec_u32_result: get_fn!("occt_get_vec_u32_result" => i32),
            fn_get_vec_u32_result_len: get_fn!("occt_get_vec_u32_result_len" => u32),
            fn_get_vec_f64_result: get_fn!("occt_get_vec_f64_result" => i32),
            fn_get_vec_f64_result_len: get_fn!("occt_get_vec_f64_result_len" => u32),
            fn_get_vec_i32_result: get_fn!("occt_get_vec_i32_result" => i32),
            fn_get_vec_i32_result_len: get_fn!("occt_get_vec_i32_result_len" => u32),

            fn_get_bbox_xmin: get_fn!("occt_get_bbox_xmin" => f64),
            fn_get_bbox_ymin: get_fn!("occt_get_bbox_ymin" => f64),
            fn_get_bbox_zmin: get_fn!("occt_get_bbox_zmin" => f64),
            fn_get_bbox_xmax: get_fn!("occt_get_bbox_xmax" => f64),
            fn_get_bbox_ymax: get_fn!("occt_get_bbox_ymax" => f64),
            fn_get_bbox_zmax: get_fn!("occt_get_bbox_zmax" => f64),

            fn_get_mesh_positions: get_fn!("occt_get_mesh_positions" => i32),
            fn_get_mesh_positions_len: get_fn!("occt_get_mesh_positions_len" => i32),
            fn_get_mesh_normals: get_fn!("occt_get_mesh_normals" => i32),
            fn_get_mesh_normals_len: get_fn!("occt_get_mesh_normals_len" => i32),
            fn_get_mesh_indices: get_fn!("occt_get_mesh_indices" => i32),
            fn_get_mesh_indices_len: get_fn!("occt_get_mesh_indices_len" => i32),
            fn_get_mesh_face_groups: get_fn!("occt_get_mesh_face_groups" => i32),
            fn_get_mesh_face_groups_len: get_fn!("occt_get_mesh_face_groups_len" => i32),

            fn_get_mesh_batch_positions: get_fn!("occt_get_mesh_batch_positions" => i32),
            fn_get_mesh_batch_positions_len: get_fn!("occt_get_mesh_batch_positions_len" => i32),
            fn_get_mesh_batch_normals: get_fn!("occt_get_mesh_batch_normals" => i32),
            fn_get_mesh_batch_normals_len: get_fn!("occt_get_mesh_batch_normals_len" => i32),
            fn_get_mesh_batch_indices: get_fn!("occt_get_mesh_batch_indices" => i32),
            fn_get_mesh_batch_indices_len: get_fn!("occt_get_mesh_batch_indices_len" => i32),
            fn_get_mesh_batch_shape_offsets: get_fn!("occt_get_mesh_batch_shape_offsets" => i32),
            fn_get_mesh_batch_shape_count: get_fn!("occt_get_mesh_batch_shape_count" => i32),

            fn_get_edge_points: get_fn!("occt_get_edge_points" => i32),
            fn_get_edge_points_len: get_fn!("occt_get_edge_points_len" => i32),
            fn_get_edge_groups: get_fn!("occt_get_edge_groups" => i32),
            fn_get_edge_groups_len: get_fn!("occt_get_edge_groups_len" => i32),

            fn_get_nurbs_degree: get_fn!("occt_get_nurbs_degree" => i32),
            fn_get_nurbs_rational: get_fn!("occt_get_nurbs_rational" => i32),
            fn_get_nurbs_periodic: get_fn!("occt_get_nurbs_periodic" => i32),
            fn_get_nurbs_knots: get_fn!("occt_get_nurbs_knots" => i32),
            fn_get_nurbs_knots_len: get_fn!("occt_get_nurbs_knots_len" => u32),
            fn_get_nurbs_multiplicities: get_fn!("occt_get_nurbs_multiplicities" => i32),
            fn_get_nurbs_multiplicities_len: get_fn!("occt_get_nurbs_multiplicities_len" => u32),
            fn_get_nurbs_poles: get_fn!("occt_get_nurbs_poles" => i32),
            fn_get_nurbs_poles_len: get_fn!("occt_get_nurbs_poles_len" => u32),
            fn_get_nurbs_weights: get_fn!("occt_get_nurbs_weights" => i32),
            fn_get_nurbs_weights_len: get_fn!("occt_get_nurbs_weights_len" => u32),

            fn_get_evo_result_id: get_fn!("occt_get_evo_result_id" => u32),
            fn_get_evo_modified: get_fn!("occt_get_evo_modified" => i32),
            fn_get_evo_modified_len: get_fn!("occt_get_evo_modified_len" => u32),
            fn_get_evo_generated: get_fn!("occt_get_evo_generated" => i32),
            fn_get_evo_generated_len: get_fn!("occt_get_evo_generated_len" => u32),
            fn_get_evo_deleted: get_fn!("occt_get_evo_deleted" => i32),
            fn_get_evo_deleted_len: get_fn!("occt_get_evo_deleted_len" => u32),

            fn_get_proj_visible_outline: get_fn!("occt_get_proj_visible_outline" => u32),
            fn_get_proj_visible_smooth: get_fn!("occt_get_proj_visible_smooth" => u32),
            fn_get_proj_visible_sharp: get_fn!("occt_get_proj_visible_sharp" => u32),
            fn_get_proj_hidden_outline: get_fn!("occt_get_proj_hidden_outline" => u32),
            fn_get_proj_hidden_smooth: get_fn!("occt_get_proj_hidden_smooth" => u32),
            fn_get_proj_hidden_sharp: get_fn!("occt_get_proj_hidden_sharp" => u32),

            fn_get_label_info_label_id: get_fn!("occt_get_label_info_label_id" => i32),
            fn_get_label_info_name: get_fn!("occt_get_label_info_name" => i32),
            fn_get_label_info_name_len: get_fn!("occt_get_label_info_name_len" => u32),
            fn_get_label_info_has_color: get_fn!("occt_get_label_info_has_color" => i32),
            fn_get_label_info_r: get_fn!("occt_get_label_info_r" => f64),
            fn_get_label_info_g: get_fn!("occt_get_label_info_g" => f64),
            fn_get_label_info_b: get_fn!("occt_get_label_info_b" => f64),
            fn_get_label_info_is_assembly: get_fn!("occt_get_label_info_is_assembly" => i32),
            fn_get_label_info_is_component: get_fn!("occt_get_label_info_is_component" => i32),
            fn_get_label_info_shape_id: get_fn!("occt_get_label_info_shape_id" => u32),

            generated,
            store,
        })
    }

    /// Fetch the rmesh `BrepModel` IR streams for a shape — the IR-native BREP
    /// lift (see `facade/src/rmesh_brep.cpp` for the stream layout). Returns
    /// the `(u32 topology, f64 geometry)` streams. Resolved lazily so blobs
    /// built without the extension simply error here instead of failing to
    /// instantiate.
    pub fn to_brep_ir(&mut self, id: ShapeHandle) -> OcctResult<(Vec<u32>, Vec<f64>)> {
        let run: TypedFunc<u32, i32> = self
            .instance
            .get_typed_func(&mut self.store, "occt_to_brep_ir")?;
        let status = run.call(&mut self.store, id.0)?;
        if status != 0 {
            let error_ptr: TypedFunc<(), i32> = self
                .instance
                .get_typed_func(&mut self.store, "occt_rmesh_ir_error")?;
            let error_len: TypedFunc<(), u32> = self
                .instance
                .get_typed_func(&mut self.store, "occt_rmesh_ir_error_len")?;
            let ptr = error_ptr.call(&mut self.store, ())?;
            let len = error_len.call(&mut self.store, ())?;
            let bytes = self.read_bytes(ptr.cast_unsigned(), len)?;
            return Err(OcctError::Operation {
                operation: "to_brep_ir".to_owned(),
                message: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }
        let u32_ptr: TypedFunc<(), i32> = self
            .instance
            .get_typed_func(&mut self.store, "occt_rmesh_ir_u32")?;
        let u32_len: TypedFunc<(), u32> = self
            .instance
            .get_typed_func(&mut self.store, "occt_rmesh_ir_u32_len")?;
        let f64_ptr: TypedFunc<(), i32> = self
            .instance
            .get_typed_func(&mut self.store, "occt_rmesh_ir_f64")?;
        let f64_len: TypedFunc<(), u32> = self
            .instance
            .get_typed_func(&mut self.store, "occt_rmesh_ir_f64_len")?;
        let topology_ptr = u32_ptr.call(&mut self.store, ())?;
        let topology_len = u32_len.call(&mut self.store, ())?;
        let geometry_ptr = f64_ptr.call(&mut self.store, ())?;
        let geometry_len = f64_len.call(&mut self.store, ())?;
        let topology = self.read_u32_slice(topology_ptr.cast_unsigned(), topology_len)?;
        let geometry = self.read_f64_slice(geometry_ptr.cast_unsigned(), geometry_len)?;
        Ok((topology, geometry))
    }

    // === Memory helpers ===

    /// Write bytes into WASM linear memory via `occt_alloc`.
    pub(crate) fn write_bytes(&mut self, data: &[u8]) -> OcctResult<u32> {
        let ptr = self.fn_alloc.call(&mut self.store, data.len() as u32)?;
        if ptr == 0 {
            core::hint::cold_path();
            return Err(OcctError::Memory("allocation failed".to_owned()));
        }
        let mem = self.memory.data_mut(&mut self.store);
        let start = ptr as usize;
        let end = start + data.len();
        if end > mem.len() {
            core::hint::cold_path();
            return Err(OcctError::Memory("write out of bounds".to_owned()));
        }
        mem[start..end].copy_from_slice(data);
        Ok(ptr)
    }

    /// Free previously allocated WASM memory.
    pub(crate) fn free_bytes(&mut self, ptr: u32) -> OcctResult<()> {
        self.fn_free.call(&mut self.store, ptr)?;
        Ok(())
    }

    /// Read a byte slice from WASM memory.
    fn read_bytes(&self, ptr: u32, len: u32) -> OcctResult<Vec<u8>> {
        let mem = self.memory.data(&self.store);
        let start = ptr as usize;
        let end = start + len as usize;
        if end > mem.len() {
            core::hint::cold_path();
            return Err(OcctError::Memory("read out of bounds".to_owned()));
        }
        Ok(mem[start..end].to_vec())
    }

    /// Read a typed slice from WASM memory as `f32` values.
    fn read_f32_slice(&self, ptr: u32, count: u32) -> OcctResult<Vec<f32>> {
        let bytes = self.read_bytes(ptr, count * 4)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Read a typed slice from WASM memory as `u32` values.
    fn read_u32_slice(&self, ptr: u32, count: u32) -> OcctResult<Vec<u32>> {
        let bytes = self.read_bytes(ptr, count * 4)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Read a typed slice from WASM memory as `i32` values.
    fn read_i32_slice(&self, ptr: u32, count: u32) -> OcctResult<Vec<i32>> {
        let bytes = self.read_bytes(ptr, count * 4)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Read a typed slice from WASM memory as `f64` values.
    fn read_f64_slice(&self, ptr: u32, count: u32) -> OcctResult<Vec<f64>> {
        let bytes = self.read_bytes(ptr, count * 8)?;
        Ok(bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    // === Error handling ===

    /// Check if the WASM module has a pending error and return it.
    pub(crate) fn check_error(&mut self, operation: &str) -> OcctResult<()> {
        let has_error = self.fn_has_error.call(&mut self.store, ())?;
        if has_error != 0 {
            // Errors are the exception, not the norm — this branch runs after
            // every facade call. Hint to LLVM that it's cold so the happy path
            // stays tight in the instruction cache. (stabilized in Rust 1.95)
            core::hint::cold_path();
            return Err(self.read_last_error(operation));
        }
        Ok(())
    }

    /// Read the last error message from the WASM module.
    #[cold]
    pub(crate) fn read_last_error(&mut self, operation: &str) -> OcctError {
        let ptr = self.fn_get_error.call(&mut self.store, ()).unwrap_or(0);
        let len = self.fn_get_error_len.call(&mut self.store, ()).unwrap_or(0);
        let message = if ptr != 0 && len > 0 {
            self.read_bytes(ptr as u32, len)
                .ok()
                .and_then(|b| String::from_utf8(b).ok())
                .unwrap_or_else(|| "unknown error".to_owned())
        } else {
            "unknown error".to_owned()
        };
        OcctError::Operation {
            operation: operation.to_owned(),
            message,
        }
    }

    // === Result buffer readers ===

    /// Read a string result from the WASM string buffer.
    pub(crate) fn read_string_result(&mut self) -> OcctResult<String> {
        let ptr = self.fn_get_string_result.call(&mut self.store, ())?;
        let len = self.fn_get_string_result_len.call(&mut self.store, ())?;
        let bytes = self.read_bytes(ptr as u32, len)?;
        String::from_utf8(bytes).map_err(|e| OcctError::Memory(e.to_string()))
    }

    /// Read a `Vec<u32>` result.
    pub(crate) fn read_vec_u32_result(&mut self) -> OcctResult<Vec<u32>> {
        let ptr = self.fn_get_vec_u32_result.call(&mut self.store, ())?;
        let len = self.fn_get_vec_u32_result_len.call(&mut self.store, ())?;
        self.read_u32_slice(ptr as u32, len)
    }

    /// Read a `Vec<f64>` result.
    pub(crate) fn read_vec_f64_result(&mut self) -> OcctResult<Vec<f64>> {
        let ptr = self.fn_get_vec_f64_result.call(&mut self.store, ())?;
        let len = self.fn_get_vec_f64_result_len.call(&mut self.store, ())?;
        self.read_f64_slice(ptr as u32, len)
    }

    /// Read a `Vec<i32>` result.
    pub(crate) fn read_vec_i32_result(&mut self) -> OcctResult<Vec<i32>> {
        let ptr = self.fn_get_vec_i32_result.call(&mut self.store, ())?;
        let len = self.fn_get_vec_i32_result_len.call(&mut self.store, ())?;
        self.read_i32_slice(ptr as u32, len)
    }

    /// Read a bounding box result.
    pub(crate) fn read_bbox_result(&mut self) -> OcctResult<BoundingBox> {
        Ok(BoundingBox {
            min: Vec3 {
                x: self.fn_get_bbox_xmin.call(&mut self.store, ())?,
                y: self.fn_get_bbox_ymin.call(&mut self.store, ())?,
                z: self.fn_get_bbox_zmin.call(&mut self.store, ())?,
            },
            max: Vec3 {
                x: self.fn_get_bbox_xmax.call(&mut self.store, ())?,
                y: self.fn_get_bbox_ymax.call(&mut self.store, ())?,
                z: self.fn_get_bbox_zmax.call(&mut self.store, ())?,
            },
        })
    }

    /// Read a mesh result.
    pub(crate) fn read_mesh_result(&mut self) -> OcctResult<Mesh> {
        let pos_ptr = self.fn_get_mesh_positions.call(&mut self.store, ())?;
        let pos_len = self.fn_get_mesh_positions_len.call(&mut self.store, ())?;
        let norm_ptr = self.fn_get_mesh_normals.call(&mut self.store, ())?;
        let norm_len = self.fn_get_mesh_normals_len.call(&mut self.store, ())?;
        let idx_ptr = self.fn_get_mesh_indices.call(&mut self.store, ())?;
        let idx_len = self.fn_get_mesh_indices_len.call(&mut self.store, ())?;
        let fg_ptr = self.fn_get_mesh_face_groups.call(&mut self.store, ())?;
        let fg_len = self.fn_get_mesh_face_groups_len.call(&mut self.store, ())?;

        Ok(Mesh {
            positions: self.read_f32_slice(pos_ptr as u32, pos_len as u32)?,
            normals: self.read_f32_slice(norm_ptr as u32, norm_len as u32)?,
            indices: self.read_u32_slice(idx_ptr as u32, idx_len as u32)?,
            face_groups: self.read_i32_slice(fg_ptr as u32, fg_len as u32)?,
        })
    }

    /// Read a mesh batch result.
    pub(crate) fn read_mesh_batch_result(&mut self) -> OcctResult<MeshBatch> {
        let pos_ptr = self.fn_get_mesh_batch_positions.call(&mut self.store, ())?;
        let pos_len = self
            .fn_get_mesh_batch_positions_len
            .call(&mut self.store, ())?;
        let norm_ptr = self.fn_get_mesh_batch_normals.call(&mut self.store, ())?;
        let norm_len = self
            .fn_get_mesh_batch_normals_len
            .call(&mut self.store, ())?;
        let idx_ptr = self.fn_get_mesh_batch_indices.call(&mut self.store, ())?;
        let idx_len = self
            .fn_get_mesh_batch_indices_len
            .call(&mut self.store, ())?;
        let off_ptr = self
            .fn_get_mesh_batch_shape_offsets
            .call(&mut self.store, ())?;
        let shape_count = self
            .fn_get_mesh_batch_shape_count
            .call(&mut self.store, ())?;

        Ok(MeshBatch {
            positions: self.read_f32_slice(pos_ptr as u32, pos_len as u32)?,
            normals: self.read_f32_slice(norm_ptr as u32, norm_len as u32)?,
            indices: self.read_u32_slice(idx_ptr as u32, idx_len as u32)?,
            shape_offsets: self.read_i32_slice(off_ptr as u32, (shape_count * 4) as u32)?,
        })
    }

    /// Read an edge data result.
    pub(crate) fn read_edge_result(&mut self) -> OcctResult<EdgeData> {
        let pts_ptr = self.fn_get_edge_points.call(&mut self.store, ())?;
        let pts_len = self.fn_get_edge_points_len.call(&mut self.store, ())?;
        let grp_ptr = self.fn_get_edge_groups.call(&mut self.store, ())?;
        let grp_len = self.fn_get_edge_groups_len.call(&mut self.store, ())?;

        Ok(EdgeData {
            points: self.read_f32_slice(pts_ptr as u32, pts_len as u32)?,
            edge_groups: self.read_i32_slice(grp_ptr as u32, grp_len as u32)?,
        })
    }

    /// Read a NURBS curve data result.
    pub(crate) fn read_nurbs_result(&mut self) -> OcctResult<NurbsCurveData> {
        let degree = self.fn_get_nurbs_degree.call(&mut self.store, ())?;
        let rational = self.fn_get_nurbs_rational.call(&mut self.store, ())? != 0;
        let periodic = self.fn_get_nurbs_periodic.call(&mut self.store, ())? != 0;

        let knots_ptr = self.fn_get_nurbs_knots.call(&mut self.store, ())?;
        let knots_len = self.fn_get_nurbs_knots_len.call(&mut self.store, ())?;
        let mult_ptr = self.fn_get_nurbs_multiplicities.call(&mut self.store, ())?;
        let mult_len = self
            .fn_get_nurbs_multiplicities_len
            .call(&mut self.store, ())?;
        let poles_ptr = self.fn_get_nurbs_poles.call(&mut self.store, ())?;
        let poles_len = self.fn_get_nurbs_poles_len.call(&mut self.store, ())?;
        let weights_ptr = self.fn_get_nurbs_weights.call(&mut self.store, ())?;
        let weights_len = self.fn_get_nurbs_weights_len.call(&mut self.store, ())?;

        Ok(NurbsCurveData {
            degree,
            rational,
            periodic,
            knots: self.read_f64_slice(knots_ptr as u32, knots_len)?,
            multiplicities: self.read_i32_slice(mult_ptr as u32, mult_len)?,
            poles: self.read_f64_slice(poles_ptr as u32, poles_len)?,
            weights: self.read_f64_slice(weights_ptr as u32, weights_len)?,
        })
    }

    /// Read an evolution data result.
    pub(crate) fn read_evolution_result(&mut self) -> OcctResult<EvolutionData> {
        let result_id = self.fn_get_evo_result_id.call(&mut self.store, ())?;
        let mod_ptr = self.fn_get_evo_modified.call(&mut self.store, ())?;
        let mod_len = self.fn_get_evo_modified_len.call(&mut self.store, ())?;
        let gen_ptr = self.fn_get_evo_generated.call(&mut self.store, ())?;
        let gen_len = self.fn_get_evo_generated_len.call(&mut self.store, ())?;
        let del_ptr = self.fn_get_evo_deleted.call(&mut self.store, ())?;
        let del_len = self.fn_get_evo_deleted_len.call(&mut self.store, ())?;

        Ok(EvolutionData {
            result_id,
            modified: self.read_i32_slice(mod_ptr as u32, mod_len)?,
            generated: self.read_i32_slice(gen_ptr as u32, gen_len)?,
            deleted: self.read_i32_slice(del_ptr as u32, del_len)?,
        })
    }

    /// Read a projection data result.
    pub(crate) fn read_projection_result(&mut self) -> OcctResult<ProjectionData> {
        Ok(ProjectionData {
            visible_outline: ShapeHandle(
                self.fn_get_proj_visible_outline.call(&mut self.store, ())?,
            ),
            visible_smooth: ShapeHandle(self.fn_get_proj_visible_smooth.call(&mut self.store, ())?),
            visible_sharp: ShapeHandle(self.fn_get_proj_visible_sharp.call(&mut self.store, ())?),
            hidden_outline: ShapeHandle(self.fn_get_proj_hidden_outline.call(&mut self.store, ())?),
            hidden_smooth: ShapeHandle(self.fn_get_proj_hidden_smooth.call(&mut self.store, ())?),
            hidden_sharp: ShapeHandle(self.fn_get_proj_hidden_sharp.call(&mut self.store, ())?),
        })
    }

    /// Read a label info result.
    pub(crate) fn read_label_info_result(&mut self) -> OcctResult<LabelInfo> {
        let label_id = self.fn_get_label_info_label_id.call(&mut self.store, ())?;
        let name_ptr = self.fn_get_label_info_name.call(&mut self.store, ())?;
        let name_len = self.fn_get_label_info_name_len.call(&mut self.store, ())?;
        let name_bytes = self.read_bytes(name_ptr as u32, name_len)?;
        let name = String::from_utf8(name_bytes).map_err(|e| OcctError::Memory(e.to_string()))?;

        Ok(LabelInfo {
            label_id,
            name,
            has_color: self.fn_get_label_info_has_color.call(&mut self.store, ())? != 0,
            r: self.fn_get_label_info_r.call(&mut self.store, ())?,
            g: self.fn_get_label_info_g.call(&mut self.store, ())?,
            b: self.fn_get_label_info_b.call(&mut self.store, ())?,
            is_assembly: self
                .fn_get_label_info_is_assembly
                .call(&mut self.store, ())?
                != 0,
            is_component: self
                .fn_get_label_info_is_component
                .call(&mut self.store, ())?
                != 0,
            shape_id: self.fn_get_label_info_shape_id.call(&mut self.store, ())?,
        })
    }
}

impl Drop for OcctKernel {
    fn drop(&mut self) {
        // Best-effort cleanup
        if let Ok(destroy) = self
            .instance
            .get_typed_func::<(), ()>(&mut self.store, "occt_destroy")
        {
            let _ = destroy.call(&mut self.store, ());
        }
    }
}

/// Satisfy the module's imports so the emcc-standalone blob instantiates under
/// a bare wasmtime `Linker`.
///
/// The blob is mostly WASI-free but emcc leaves a handful of
/// `wasi_snapshot_preview1::*` and `env::*` imports. Two need real behavior:
/// `clock_time_get` (OCCT date handling rejects a zero clock) and `fd_write`
/// (the writev loop spins forever unless `nwritten` is set). Everything else
/// (`emscripten_*`, `__syscall_*` filesystem shims) is stubbed to return zero —
/// the compute paths never exercise them, and an unexpected use fails loudly at
/// the operation level rather than at instantiation.
fn define_host_imports(linker: &mut Linker<()>, module: &Module) -> OcctResult<()> {
    const WASI: &str = "wasi_snapshot_preview1";
    linker.func_wrap(
        WASI,
        "clock_time_get",
        |mut caller: wasmtime::Caller<'_, ()>, _clock_id: i32, _precision: i64, out: i32| -> i32 {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0u64, |elapsed| {
                    u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
                });
            let Some(wasmtime::Extern::Memory(memory)) = caller.get_export("memory") else {
                return 8; // WASI errno: badf
            };
            let Ok(offset) = usize::try_from(out) else {
                return 21; // WASI errno: fault
            };
            match memory.write(&mut caller, offset, &now.to_le_bytes()) {
                Ok(()) => 0,
                Err(_) => 21,
            }
        },
    )?;
    linker.func_wrap(
        WASI,
        "fd_write",
        |mut caller: wasmtime::Caller<'_, ()>,
         _fd: i32,
         iovs: i32,
         iovs_len: i32,
         nwritten: i32|
         -> i32 {
            // Pretend every byte was written: sum the iovec lengths and report
            // them back. Returning success WITHOUT setting `nwritten` makes
            // emcc's writev loop spin forever.
            let Some(wasmtime::Extern::Memory(memory)) = caller.get_export("memory") else {
                return 8;
            };
            let (Ok(base), Ok(count), Ok(out)) = (
                usize::try_from(iovs),
                usize::try_from(iovs_len),
                usize::try_from(nwritten),
            ) else {
                return 21;
            };
            let mut total: u32 = 0;
            for index in 0..count {
                let mut length_bytes = [0u8; 4];
                if memory
                    .read(&caller, base + index * 8 + 4, &mut length_bytes)
                    .is_err()
                {
                    return 21;
                }
                total = total.wrapping_add(u32::from_le_bytes(length_bytes));
            }
            match memory.write(&mut caller, out, &total.to_le_bytes()) {
                Ok(()) => 0,
                Err(_) => 21,
            }
        },
    )?;
    let already_defined = [(WASI, "clock_time_get"), (WASI, "fd_write")];
    for import in module.imports() {
        if already_defined
            .iter()
            .any(|(module_name, name)| *module_name == import.module() && *name == import.name())
        {
            continue;
        }
        if let wasmtime::ExternType::Func(func_type) = import.ty() {
            let results: Vec<wasmtime::ValType> = func_type.results().collect();
            linker.func_new(
                import.module(),
                import.name(),
                func_type.clone(),
                move |_caller, _params, out| {
                    for (slot, value_type) in out.iter_mut().zip(&results) {
                        *slot = match value_type {
                            wasmtime::ValType::I32 => wasmtime::Val::I32(0),
                            wasmtime::ValType::I64 => wasmtime::Val::I64(0),
                            wasmtime::ValType::F32 => wasmtime::Val::F32(0),
                            wasmtime::ValType::F64 => wasmtime::Val::F64(0),
                            other => {
                                return Err(wasmtime::Error::msg(format!(
                                    "unstubable import result type {other}"
                                )));
                            }
                        };
                    }
                    Ok(())
                },
            )?;
        }
    }
    Ok(())
}
