// occt_rmesh_history_*: per-operation provenance for the rmesh embedding.
//
// Answers "which input entity produced this output entity" for an operation,
// in the ONE vocabulary both sides of the seam already share: **indices into
// rmesh's own sub-shape enumeration**. No hashes cross the wire.
//
// Hand-written on purpose (kept OUT of the generated facade) so the fork's
// delta stays a few isolated files; it reaches the live kernel through the
// generated file's extension accessor `occt_wasi_kernel_for_extensions()`.
// `build_wasi.rs` scans `facade/src/*.cpp` for exports, so these land in BOTH
// the full and the minimal profile without touching the codegen specs.
//
// WHY NOT `BRepTools_History`: it is the right object to *persist and compose*
// a history, and OCCT's `Merge` is exactly the composition rmesh's
// `boundary::history` implements. But rmesh composes on its own side, where
// the algebra is proptested, so the kernel only ever needs to report ONE
// operation. Every algorithm here derives from `BRepBuilderAPI_MakeShape` and
// overrides `Modified`/`Generated`/`IsDeleted` — and for the boolean API those
// three already fold in the simplification history merged by `SimplifyResult`.
// So one template serves both op families, and no history arena is needed.
//
// ENUMERATION, the thing that must agree exactly. rmesh indexes faces and
// edges by their position in `BrepModel`, which `rmesh_brep.cpp`'s `seed_order`
// defines as:
//   faces: `TopExp::MapShapes(FACE)` order.
//   edges: `TopExp::MapShapes(EDGE)` order **minus degenerate edges**.
// The edge exclusion is not cosmetic: a pole seam carries no 3D curve and so
// never reaches the IR, and every filleted corner has three of them. Reporting
// raw `MapShapes` edge indices would silently shift every edge reference on any
// blended body. `enumerate()` below is that rule, and it is the only place it
// is spelled in this file.
//
// Stream layout (one u32 stream; counts up front, so a truncated stream is
// detectable and no record needs a sentinel):
//   header: [n_input_faces, n_result_faces, n_input_edges, n_result_edges,
//            n_records]
//   record: [source_kind, source_index, relation, result_kind, n_results,
//            result_index * n_results]
//     kind:     0 = face, 1 = edge
//     relation: 0 = modified, 1 = generated, 2 = removed (n_results == 0)
//
// A record may cross kinds, and that is the point: a fillet's new face is
// `Generated` BY THE EDGE it was built on, which is the single most useful
// naming relation the kernel knows. A same-kind `modified` record is the
// "still the same entity, possibly reshaped" claim.
//
// UNCHANGED entities: OCCT reports them by saying nothing at all — `Modified`
// is empty and `IsDeleted` is false for an entity the operation left alone
// (`BRepTools_History.hxx`: "Each output shape should be an input shape or
// generated or modified from an input shape"). Left implicit that would read as
// "brand new", the opposite of the truth, orphaning every reference to every
// face an operation did not touch. So this file makes it explicit: an untouched
// input that is still present in the result is emitted as `modified` to itself.
// The presence test is an `NCollection_IndexedMap` lookup with the same hasher
// `seed_order` uses, i.e. OCCT's own within-session shape identity — a map
// lookup, never a value on the wire.

#include "occt_kernel.h"

#include <BRepAlgoAPI_BooleanOperation.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepFilletAPI_MakeChamfer.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <Message_ProgressRange.hxx>
#include <NCollection_IndexedMap.hxx>
#include <NCollection_List.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopTools_ShapeMapHasher.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Defined in the generated wasi_exports.cpp (inside its extern "C" block).
extern "C" OcctKernel* occt_wasi_kernel_for_extensions();

namespace {

std::vector<uint32_t> g_hist_u32;
std::string g_hist_error;
uint32_t g_hist_result = 0;

constexpr uint32_t KIND_FACE = 0;
constexpr uint32_t KIND_EDGE = 1;

constexpr uint32_t RELATION_MODIFIED = 0;
constexpr uint32_t RELATION_GENERATED = 1;
constexpr uint32_t RELATION_REMOVED = 2;

using ShapeMap = NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher>;
using ShapeList = NCollection_List<TopoDS_Shape>;

/// rmesh's sub-shape enumeration for one entity kind of one shape.
///
/// `map` is OCCT's full `MapShapes` order; `rmesh_index[i - 1]` is the index
/// rmesh assigns to map entry `i`, or -1 for an entity rmesh does not enumerate
/// (a degenerate edge). Keeping both means a lookup can answer "is this in the
/// result, and if so what does rmesh call it" in one step.
struct Enumeration {
    ShapeMap map;
    std::vector<int32_t> rmesh_index;
    uint32_t count = 0;
};

Enumeration enumerate(const TopoDS_Shape& shape, TopAbs_ShapeEnum kind) {
    Enumeration enumeration;
    TopExp::MapShapes(shape, kind, enumeration.map);
    enumeration.rmesh_index.assign(static_cast<size_t>(enumeration.map.Extent()), -1);
    for (int i = 1; i <= enumeration.map.Extent(); ++i) {
        // The one place seed_order's rule is spelled: degenerate edges are not
        // part of rmesh's enumeration, so they hold no index.
        if (kind == TopAbs_EDGE && BRep_Tool::Degenerated(TopoDS::Edge(enumeration.map.FindKey(i)))) {
            continue;
        }
        enumeration.rmesh_index[static_cast<size_t>(i) - 1] =
            static_cast<int32_t>(enumeration.count++);
    }
    return enumeration;
}

/// rmesh's index for a shape, or -1 if it is absent or unenumerated.
int32_t lookup(const Enumeration& enumeration, const TopoDS_Shape& shape) {
    const int index = enumeration.map.FindIndex(shape);
    if (index == 0) {
        return -1;
    }
    return enumeration.rmesh_index[static_cast<size_t>(index) - 1];
}

struct Streams {
    std::vector<uint32_t> records;
    uint32_t record_count = 0;

    void push(uint32_t source_kind, uint32_t source_index, uint32_t relation, uint32_t result_kind,
              const std::vector<uint32_t>& results) {
        records.push_back(source_kind);
        records.push_back(source_index);
        records.push_back(relation);
        records.push_back(result_kind);
        records.push_back(static_cast<uint32_t>(results.size()));
        records.insert(records.end(), results.begin(), results.end());
        ++record_count;
    }
};

/// Partition a result list by entity kind and translate to rmesh indices.
///
/// Results of a kind rmesh does not track (a vertex, a wire) are dropped, as is
/// anything not present in the result shape — a `Modified` list may legitimately
/// name splits that the operation then discarded.
void partition(const ShapeList& shapes, const Enumeration& result_faces,
               const Enumeration& result_edges, std::vector<uint32_t>& faces,
               std::vector<uint32_t>& edges) {
    for (ShapeList::Iterator it(shapes); it.More(); it.Next()) {
        const TopoDS_Shape& shape = it.Value();
        if (shape.ShapeType() == TopAbs_FACE) {
            const int32_t index = lookup(result_faces, shape);
            if (index >= 0) {
                faces.push_back(static_cast<uint32_t>(index));
            }
        } else if (shape.ShapeType() == TopAbs_EDGE) {
            const int32_t index = lookup(result_edges, shape);
            if (index >= 0) {
                edges.push_back(static_cast<uint32_t>(index));
            }
        }
    }
}

/// Emit every relation the algorithm knows about one source entity kind.
///
/// Templated rather than virtual because the two op families share nothing but
/// these three method names — `BRepAlgoAPI_BuilderAlgo` and
/// `BRepFilletAPI_MakeFillet` both inherit them from `BRepBuilderAPI_MakeShape`,
/// and for the boolean API they already account for the simplification history
/// that `SimplifyResult` merged in.
template <class Algo>
void emit_kind(Algo& algo, const Enumeration& source, uint32_t source_kind,
               const Enumeration& result_faces, const Enumeration& result_edges,
               Streams& streams) {
    for (int i = 1; i <= source.map.Extent(); ++i) {
        const int32_t source_index = source.rmesh_index[static_cast<size_t>(i) - 1];
        if (source_index < 0) {
            continue;
        }
        const TopoDS_Shape& shape = source.map.FindKey(i);
        const uint32_t index = static_cast<uint32_t>(source_index);

        std::vector<uint32_t> modified_faces;
        std::vector<uint32_t> modified_edges;
        partition(algo.Modified(shape), result_faces, result_edges, modified_faces, modified_edges);

        std::vector<uint32_t> generated_faces;
        std::vector<uint32_t> generated_edges;
        partition(algo.Generated(shape), result_faces, result_edges, generated_faces,
                  generated_edges);

        const bool deleted = algo.IsDeleted(shape);
        const bool silent = modified_faces.empty() && modified_edges.empty();

        // The untouched case: OCCT says nothing, so say it here (see the header
        // comment). Only for a source the operation neither modified nor
        // deleted, and only when it is genuinely still in the result.
        if (silent && !deleted) {
            const Enumeration& same_kind = source_kind == KIND_FACE ? result_faces : result_edges;
            const int32_t survivor = lookup(same_kind, shape);
            if (survivor >= 0) {
                (source_kind == KIND_FACE ? modified_faces : modified_edges)
                    .push_back(static_cast<uint32_t>(survivor));
            }
        }

        if (!modified_faces.empty()) {
            streams.push(source_kind, index, RELATION_MODIFIED, KIND_FACE, modified_faces);
        }
        if (!modified_edges.empty()) {
            streams.push(source_kind, index, RELATION_MODIFIED, KIND_EDGE, modified_edges);
        }
        if (!generated_faces.empty()) {
            streams.push(source_kind, index, RELATION_GENERATED, KIND_FACE, generated_faces);
        }
        if (!generated_edges.empty()) {
            streams.push(source_kind, index, RELATION_GENERATED, KIND_EDGE, generated_edges);
        }
        // Removal is reported only when nothing else was: a filleted edge is
        // both `IsDeleted` and the generator of the blend face, and the
        // generated relation is the informative half. Emitting both would let
        // the two disagree on the rmesh side about whether the entity survived.
        if (deleted && modified_faces.empty() && modified_edges.empty() &&
            generated_faces.empty() && generated_edges.empty()) {
            streams.push(source_kind, index, RELATION_REMOVED, source_kind, {});
        }
    }
}

/// Run the whole reporting pass for one operation and publish it to the globals.
template <class Algo>
void report(Algo& algo, const TopoDS_Shape& input, const TopoDS_Shape& result) {
    const Enumeration input_faces = enumerate(input, TopAbs_FACE);
    const Enumeration input_edges = enumerate(input, TopAbs_EDGE);
    const Enumeration result_faces = enumerate(result, TopAbs_FACE);
    const Enumeration result_edges = enumerate(result, TopAbs_EDGE);

    Streams streams;
    emit_kind(algo, input_faces, KIND_FACE, result_faces, result_edges, streams);
    emit_kind(algo, input_edges, KIND_EDGE, result_faces, result_edges, streams);

    g_hist_u32.clear();
    g_hist_u32.push_back(input_faces.count);
    g_hist_u32.push_back(result_faces.count);
    g_hist_u32.push_back(input_edges.count);
    g_hist_u32.push_back(result_edges.count);
    g_hist_u32.push_back(streams.record_count);
    g_hist_u32.insert(g_hist_u32.end(), streams.records.begin(), streams.records.end());
}

/// A boolean's two arguments are one history domain: the relations name
/// sub-shapes of either operand, so both must share one index space. A compound
/// of the two gives exactly that, and `MapShapes` over it visits `a` then `b` —
/// so an rmesh index below `a`'s count is `a`'s, and the rest are `b`'s, shifted.
TopoDS_Shape both(const TopoDS_Shape& a, const TopoDS_Shape& b) {
    TopoDS_Compound compound;
    BRep_Builder builder;
    builder.MakeCompound(compound);
    builder.Add(compound, a);
    builder.Add(compound, b);
    return compound;
}

void reset() {
    g_hist_u32.clear();
    g_hist_error.clear();
    g_hist_result = 0;
}

}  // namespace

extern "C" {

/// Boolean with provenance. `op_code` is 0=fuse, 1=cut, 2=common — the same
/// encoding `booleanPipeline`/`booleanFuzzy` use. A negative `fuzz` means "no
/// fuzzy value"; otherwise it is passed through, and `SimplifyResult` then
/// reuses it as the unifier's linear tolerance.
///
/// The refine is `SimplifyResult`, NOT a hand-rolled `ShapeUpgrade_UnifySameDomain`
/// like the generated boolean specs use. It performs the same unification and
/// then merges the unifier's history into the operation's, which is what makes a
/// refined boolean reportable at all; it additionally passes the fuzzy value as
/// the unifier's linear tolerance and the non-destructive flag as its safe-input
/// mode, neither of which the hand-rolled version does. It does force
/// `ConcatBSplines` on, where the hand-rolled call passes it off.
int32_t occt_rmesh_history_boolean(uint32_t a_id, uint32_t b_id, uint32_t op_code, double fuzz,
                                   uint32_t simplify) {
    reset();
    try {
        OcctKernel* kernel = occt_wasi_kernel_for_extensions();
        if (kernel == nullptr) {
            throw std::runtime_error("kernel not initialized");
        }
        const TopoDS_Shape& a = kernel->get(a_id);
        const TopoDS_Shape& b = kernel->get(b_id);

        ShapeList args;
        args.Append(a);
        ShapeList tools;
        tools.Append(b);

        auto run = [&](BRepAlgoAPI_BooleanOperation& op, const char* name) {
            op.SetArguments(args);
            op.SetTools(tools);
            if (fuzz >= 0.0) {
                op.SetFuzzyValue(fuzz);
            }
            op.Build();
            if (!op.IsDone() || op.HasErrors()) {
                throw std::runtime_error(std::string(name) + ": operation failed");
            }
            if (simplify != 0) {
                op.SimplifyResult(true, true);
            }
            const TopoDS_Shape result = op.Shape();
            report(op, both(a, b), result);
            g_hist_result = kernel->store(result);
        };

        switch (op_code) {
            case 0: {
                BRepAlgoAPI_Fuse op;
                run(op, "fuse");
                break;
            }
            case 1: {
                BRepAlgoAPI_Cut op;
                run(op, "cut");
                break;
            }
            case 2: {
                BRepAlgoAPI_Common op;
                run(op, "common");
                break;
            }
            default:
                throw std::runtime_error("unknown boolean op code");
        }
        return 0;
    } catch (const std::exception& e) {
        g_hist_error = std::string("history_boolean: ") + e.what();
        return -1;
    } catch (...) {
        g_hist_error = "history_boolean: unknown OCCT failure";
        return -1;
    }
}

/// Fillet (`kind` 0) or chamfer (`kind` 1) with provenance. `edge_ptr`/`edge_len`
/// are arena ids of edges of the solid, matching `occt_fillet`'s convention.
int32_t occt_rmesh_history_blend(uint32_t solid_id, const uint32_t* edge_ptr, uint32_t edge_len,
                                 double param, uint32_t kind) {
    reset();
    try {
        OcctKernel* kernel = occt_wasi_kernel_for_extensions();
        if (kernel == nullptr) {
            throw std::runtime_error("kernel not initialized");
        }
        const TopoDS_Shape& solid = kernel->get(solid_id);
        if (edge_len == 0) {
            throw std::runtime_error("no edges to blend");
        }

        auto run = [&](auto& maker, const char* name) {
            for (uint32_t i = 0; i < edge_len; ++i) {
                maker.Add(param, TopoDS::Edge(kernel->get(edge_ptr[i])));
            }
            maker.Build();
            if (!maker.IsDone()) {
                throw std::runtime_error(std::string(name) + ": operation failed");
            }
            const TopoDS_Shape result = maker.Shape();
            report(maker, solid, result);
            g_hist_result = kernel->store(result);
        };

        if (kind == 0) {
            BRepFilletAPI_MakeFillet maker(solid);
            run(maker, "fillet");
        } else if (kind == 1) {
            BRepFilletAPI_MakeChamfer maker(solid);
            run(maker, "chamfer");
        } else {
            throw std::runtime_error("unknown blend kind");
        }
        return 0;
    } catch (const std::exception& e) {
        g_hist_error = std::string("history_blend: ") + e.what();
        return -1;
    } catch (...) {
        g_hist_error = "history_blend: unknown OCCT failure";
        return -1;
    }
}

uint32_t occt_rmesh_history_result() {
    return g_hist_result;
}
int32_t occt_rmesh_history_u32() {
    return static_cast<int32_t>(reinterpret_cast<intptr_t>(g_hist_u32.data()));
}
uint32_t occt_rmesh_history_u32_len() {
    return static_cast<uint32_t>(g_hist_u32.size());
}
int32_t occt_rmesh_history_error() {
    return static_cast<int32_t>(reinterpret_cast<intptr_t>(g_hist_error.data()));
}
uint32_t occt_rmesh_history_error_len() {
    return static_cast<uint32_t>(g_hist_error.size());
}

}  // extern "C"
