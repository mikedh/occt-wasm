//! Facade code generator for occt-wasm.
//!
//! Declares the IR types and declarative method configuration used to
//! auto-generate C++ facade implementations from OCCT class patterns.

pub mod config;
pub mod emitter;
pub mod run;
pub mod rust_emitter;
pub mod types;
pub mod wasi_emitter;

/// The `--minimal` dead-code root set, derived from the specs: every `occt_*`
/// export the WASI emitter itself produces for the core spec set — method
/// wrappers, lifecycle, the error protocol, and exactly the accessor groups
/// those specs' return types require. Core means not [`MethodKind::Skip`], not
/// the Embind-only `"marshal"` helpers (mirroring `run::run`), and not in
/// [`config::OPTIONAL_CATEGORIES`]. No name list is maintained by hand, and
/// the Rust host lazily binds exactly the optional-category wrappers, so
/// "every eagerly-bound export exists in a minimal blob" holds by
/// construction.
///
/// Hand-written facade extensions (`facade/src/*.cpp`) export their own
/// symbols; `build_wasi` scrapes and appends those separately.
pub fn minimal_export_names() -> anyhow::Result<Vec<String>> {
    let all = config::target_methods();
    config::validate(all)?;
    let core: Vec<&types::MethodSpec> = all
        .iter()
        .filter(|m| !matches!(m.kind, types::MethodKind::Skip))
        .filter(|m| m.category != "marshal")
        .filter(|m| !config::spec_is_optional(m))
        .collect();
    Ok(wasi_emitter::export_names(
        &wasi_emitter::emit_wasi_exports(&core),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derived minimal root set: core method wrappers in, optional
    /// wrappers and their accessor groups out, lifecycle always in.
    #[test]
    fn minimal_export_names_partitions_by_category() {
        let names = minimal_export_names().expect("derivation failed");
        let has = |n: &str| names.iter().any(|x| x == n);

        for m in config::target_methods() {
            if matches!(m.kind, types::MethodKind::Skip) || m.category == "marshal" {
                continue;
            }
            let export = format!("occt_{}", wasi_emitter::camel_to_snake(m.name));
            let expected = !config::spec_is_optional(m);
            assert_eq!(
                has(&export),
                expected,
                "{export} (category '{}') in minimal set: {} — expected {}",
                m.category,
                !expected,
                expected
            );
        }

        assert!(has("occt_init") && has("occt_destroy") && has("occt_alloc"));
        assert!(names.iter().all(|n| !n.starts_with("occt_get_proj_")));
        assert!(names.iter().all(|n| !n.starts_with("occt_get_label_info_")));
    }
}
