// occt_to_brep_ir: IR-native BREP lift for the rmesh embedding.
//
// Walks a result TopoDS_Shape and emits rmesh's `BrepModel` arena layout as
// two flat streams (u32 topology, f64 geometry) — the kernel hands back the
// caller's OWN intermediate representation, no exchange format on the wire.
// Analytic geometry stays analytic: lines/circles/ellipses and
// planes/cylinders/cones/spheres/tori are emitted exactly; anything else
// fails loudly (documented gap, never approximated silently).
//
// Hand-written on purpose (kept OUT of the generated facade) so the fork's
// delta stays a few isolated files; it reaches the live kernel through the
// generated file's extension accessor `occt_wasi_kernel_for_extensions()`.
//
// Stream layout (sequential cursors; six-count header first):
//   u32: [n_vertices, n_edges, n_loops, n_faces, n_shells, n_solids]
//        edges:  per edge:  [start_vertex, end_vertex, curve_kind]
//        loops:  per loop:  [kind(0=edges,1=vertex),
//                            n_edges|vertex_idx, (edge_idx, same_sense)*]
//        faces:  per face:  [surface_kind, same_sense, outer_loop,
//                            n_inner, inner_loop*]
//        shells: per shell: [n_faces, face_idx*]
//        solids: per solid: [outer_shell, n_voids, void_shell*]
//   f64: vertices (3 per) ·
//        per edge: [t_start, t_end, curve payload]
//          line   (kind 0): origin(3) unit_direction(3)
//          circle (kind 1): center(3) axis(3) x_axis(3) radius
//          ellipse(kind 2): center(3) axis(3) x_axis(3) major minor
//        per face: surface payload
//          plane   (kind 0): origin(3) normal(3)
//          cylinder(kind 1): origin(3) axis(3) radius
//          cone    (kind 2): apex(3) axis(3) half_angle
//          sphere  (kind 3): center(3) radius
//          torus   (kind 4): center(3) axis(3) major minor

#include "occt_kernel.h"

#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepClass3d.hxx>
#include <BRepTools.hxx>
#include <BRepTools_WireExplorer.hxx>
#include <BRep_Tool.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <NCollection_IndexedMap.hxx>
#include <TopTools_ShapeMapHasher.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Solid.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopoDS_Wire.hxx>
#include <gp_Circ.hxx>
#include <gp_Cone.hxx>
#include <gp_Cylinder.hxx>
#include <gp_Elips.hxx>
#include <gp_Lin.hxx>
#include <gp_Pln.hxx>
#include <gp_Sphere.hxx>
#include <gp_Torus.hxx>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Defined in the generated wasi_exports.cpp (inside its extern "C" block).
extern "C" OcctKernel* occt_wasi_kernel_for_extensions();

namespace {

std::vector<uint32_t> g_ir_u32;
std::vector<double> g_ir_f64;
std::string g_ir_error;

void push_xyz(std::vector<double>& out, double x, double y, double z) {
    out.push_back(x);
    out.push_back(y);
    out.push_back(z);
}

void push_pnt(std::vector<double>& out, const gp_Pnt& p) {
    push_xyz(out, p.X(), p.Y(), p.Z());
}

void push_dir(std::vector<double>& out, const gp_Dir& d) {
    push_xyz(out, d.X(), d.Y(), d.Z());
}

struct IrBuilder {
    std::vector<uint32_t> u32;
    std::vector<double> f64;

    // Shared-entity dedup: TShape identity via indexed maps (OCCT 8 spelling;
    // the TopTools_IndexedMapOfShape alias is deprecated).
    using ShapeMap = NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher>;
    ShapeMap vertex_map;
    ShapeMap edge_map;
    ShapeMap face_map;
    // Emission-order indices (dense; -1 = not yet emitted).
    std::vector<int32_t> vertex_ir;
    std::vector<int32_t> edge_ir;
    std::vector<int32_t> face_ir;

    std::vector<double> vertex_f64;
    std::vector<uint32_t> edge_u32;
    std::vector<double> edge_f64;
    std::vector<uint32_t> loop_u32;
    uint32_t loop_count = 0;
    std::vector<uint32_t> face_u32;
    std::vector<double> face_f64;
    std::vector<uint32_t> shell_u32;
    uint32_t shell_count = 0;
    std::vector<uint32_t> solid_u32;
    uint32_t solid_count = 0;

    uint32_t vertex_count = 0;
    uint32_t edge_count = 0;
    uint32_t face_count = 0;

    uint32_t admit_vertex(const TopoDS_Vertex& vertex) {
        const int map_index = vertex_map.Add(vertex);
        if (map_index > static_cast<int>(vertex_ir.size())) {
            vertex_ir.resize(map_index, -1);
        }
        int32_t& slot = vertex_ir[map_index - 1];
        if (slot < 0) {
            slot = static_cast<int32_t>(vertex_count++);
            push_pnt(vertex_f64, BRep_Tool::Pnt(vertex));
        }
        return static_cast<uint32_t>(slot);
    }

    uint32_t admit_edge(const TopoDS_Edge& edge) {
        const int map_index = edge_map.Add(edge);
        if (map_index > static_cast<int>(edge_ir.size())) {
            edge_ir.resize(map_index, -1);
        }
        int32_t& slot = edge_ir[map_index - 1];
        if (slot >= 0) {
            return static_cast<uint32_t>(slot);
        }
        BRepAdaptor_Curve curve(edge);
        uint32_t kind = 0;
        // Vertices at the curve's parametric ends (orientation handled by the
        // loop's same_sense, so CumOri=false).
        TopoDS_Vertex first;
        TopoDS_Vertex last;
        TopExp::Vertices(edge, first, last, /*CumOri=*/false);
        if (first.IsNull() || last.IsNull()) {
            throw std::runtime_error("edge without vertices (closed edge unsupported here)");
        }
        const uint32_t start_vertex = admit_vertex(first);
        const uint32_t end_vertex = admit_vertex(last);
        std::vector<double> payload;
        payload.push_back(curve.FirstParameter());
        payload.push_back(curve.LastParameter());
        switch (curve.GetType()) {
            case GeomAbs_Line: {
                kind = 0;
                const gp_Lin line = curve.Line();
                push_pnt(payload, line.Location());
                push_dir(payload, line.Direction());
                break;
            }
            case GeomAbs_Circle: {
                kind = 1;
                const gp_Circ circle = curve.Circle();
                push_pnt(payload, circle.Location());
                push_dir(payload, circle.Axis().Direction());
                push_dir(payload, circle.XAxis().Direction());
                payload.push_back(circle.Radius());
                break;
            }
            case GeomAbs_Ellipse: {
                kind = 2;
                const gp_Elips ellipse = curve.Ellipse();
                push_pnt(payload, ellipse.Location());
                push_dir(payload, ellipse.Axis().Direction());
                push_dir(payload, ellipse.XAxis().Direction());
                payload.push_back(ellipse.MajorRadius());
                payload.push_back(ellipse.MinorRadius());
                break;
            }
            default:
                throw std::runtime_error(
                    "curve kind not liftable yet (line/circle/ellipse only)");
        }
        slot = static_cast<int32_t>(edge_count++);
        edge_u32.push_back(start_vertex);
        edge_u32.push_back(end_vertex);
        edge_u32.push_back(kind);
        edge_f64.insert(edge_f64.end(), payload.begin(), payload.end());
        return static_cast<uint32_t>(slot);
    }

    // Emit one wire as a loop; returns the loop index.
    uint32_t emit_loop(const TopoDS_Wire& wire, const TopoDS_Face& face) {
        std::vector<uint32_t> oriented;
        for (BRepTools_WireExplorer explorer(wire, face); explorer.More();
             explorer.Next()) {
            const TopoDS_Edge& edge = explorer.Current();
            if (BRep_Tool::Degenerated(edge)) {
                continue;  // no 3D curve; pole seams reduce to vertex loops
            }
            const uint32_t edge_index = admit_edge(edge);
            const uint32_t same_sense =
                edge.Orientation() == TopAbs_FORWARD ? 1 : 0;
            oriented.push_back(edge_index);
            oriented.push_back(same_sense);
        }
        const uint32_t loop_index = loop_count++;
        if (oriented.empty()) {
            // Entirely-degenerate wire (a cone/sphere pole): a vertex loop.
            TopExp_Explorer vertices(wire, TopAbs_VERTEX);
            if (!vertices.More()) {
                throw std::runtime_error("degenerate wire without a vertex");
            }
            const uint32_t vertex_index =
                admit_vertex(TopoDS::Vertex(vertices.Current()));
            loop_u32.push_back(1);
            loop_u32.push_back(vertex_index);
        } else {
            loop_u32.push_back(0);
            loop_u32.push_back(static_cast<uint32_t>(oriented.size() / 2));
            loop_u32.insert(loop_u32.end(), oriented.begin(), oriented.end());
        }
        return loop_index;
    }

    // Emit one face (surface + loops); returns the face index.
    uint32_t admit_face(const TopoDS_Face& face) {
        const int map_index = face_map.Add(face);
        if (map_index > static_cast<int>(face_ir.size())) {
            face_ir.resize(map_index, -1);
        }
        int32_t& slot = face_ir[map_index - 1];
        if (slot >= 0) {
            return static_cast<uint32_t>(slot);
        }
        BRepAdaptor_Surface surface(face);
        uint32_t surface_kind = 0;
        std::vector<double> payload;
        switch (surface.GetType()) {
            case GeomAbs_Plane: {
                surface_kind = 0;
                const gp_Pln plane = surface.Plane();
                push_pnt(payload, plane.Location());
                push_dir(payload, plane.Axis().Direction());
                break;
            }
            case GeomAbs_Cylinder: {
                surface_kind = 1;
                const gp_Cylinder cylinder = surface.Cylinder();
                push_pnt(payload, cylinder.Location());
                push_dir(payload, cylinder.Axis().Direction());
                payload.push_back(cylinder.Radius());
                break;
            }
            case GeomAbs_Cone: {
                surface_kind = 2;
                const gp_Cone cone = surface.Cone();
                push_pnt(payload, cone.Apex());
                push_dir(payload, cone.Axis().Direction());
                payload.push_back(cone.SemiAngle());
                break;
            }
            case GeomAbs_Sphere: {
                surface_kind = 3;
                const gp_Sphere sphere = surface.Sphere();
                push_pnt(payload, sphere.Location());
                payload.push_back(sphere.Radius());
                break;
            }
            case GeomAbs_Torus: {
                surface_kind = 4;
                const gp_Torus torus = surface.Torus();
                push_pnt(payload, torus.Location());
                push_dir(payload, torus.Axis().Direction());
                payload.push_back(torus.MajorRadius());
                payload.push_back(torus.MinorRadius());
                break;
            }
            default:
                throw std::runtime_error(
                    "surface kind not liftable yet "
                    "(plane/cylinder/cone/sphere/torus only)");
        }
        const TopoDS_Wire outer_wire = BRepTools::OuterWire(face);
        if (outer_wire.IsNull()) {
            throw std::runtime_error("face without an outer wire");
        }
        const uint32_t outer_loop = emit_loop(outer_wire, face);
        std::vector<uint32_t> inner_loops;
        for (TopExp_Explorer wires(face, TopAbs_WIRE); wires.More();
             wires.Next()) {
            const TopoDS_Wire wire = TopoDS::Wire(wires.Current());
            if (wire.IsSame(outer_wire)) {
                continue;
            }
            inner_loops.push_back(emit_loop(wire, face));
        }
        slot = static_cast<int32_t>(face_count++);
        face_u32.push_back(surface_kind);
        face_u32.push_back(face.Orientation() == TopAbs_FORWARD ? 1 : 0);
        face_u32.push_back(outer_loop);
        face_u32.push_back(static_cast<uint32_t>(inner_loops.size()));
        face_u32.insert(face_u32.end(), inner_loops.begin(), inner_loops.end());
        face_f64.insert(face_f64.end(), payload.begin(), payload.end());
        return static_cast<uint32_t>(slot);
    }

    uint32_t emit_shell(const TopoDS_Shell& shell) {
        std::vector<uint32_t> faces;
        for (TopExp_Explorer explorer(shell, TopAbs_FACE); explorer.More();
             explorer.Next()) {
            faces.push_back(admit_face(TopoDS::Face(explorer.Current())));
        }
        if (faces.empty()) {
            throw std::runtime_error("empty shell");
        }
        const uint32_t shell_index = shell_count++;
        shell_u32.push_back(static_cast<uint32_t>(faces.size()));
        shell_u32.insert(shell_u32.end(), faces.begin(), faces.end());
        return shell_index;
    }

    void emit_solid(const TopoDS_Solid& solid) {
        const TopoDS_Shell outer = BRepClass3d::OuterShell(solid);
        if (outer.IsNull()) {
            throw std::runtime_error("solid without an outer shell");
        }
        const uint32_t outer_index = emit_shell(outer);
        std::vector<uint32_t> voids;
        for (TopExp_Explorer shells(solid, TopAbs_SHELL); shells.More();
             shells.Next()) {
            const TopoDS_Shell shell = TopoDS::Shell(shells.Current());
            if (shell.IsSame(outer)) {
                continue;
            }
            voids.push_back(emit_shell(shell));
        }
        solid_count++;
        solid_u32.push_back(outer_index);
        solid_u32.push_back(static_cast<uint32_t>(voids.size()));
        solid_u32.insert(solid_u32.end(), voids.begin(), voids.end());
    }

    void finish() {
        u32.push_back(vertex_count);
        u32.push_back(edge_count);
        u32.push_back(loop_count);
        u32.push_back(face_count);
        u32.push_back(shell_count);
        u32.push_back(solid_count);
        u32.insert(u32.end(), edge_u32.begin(), edge_u32.end());
        u32.insert(u32.end(), loop_u32.begin(), loop_u32.end());
        u32.insert(u32.end(), face_u32.begin(), face_u32.end());
        u32.insert(u32.end(), shell_u32.begin(), shell_u32.end());
        u32.insert(u32.end(), solid_u32.begin(), solid_u32.end());
        f64.insert(f64.end(), vertex_f64.begin(), vertex_f64.end());
        f64.insert(f64.end(), edge_f64.begin(), edge_f64.end());
        f64.insert(f64.end(), face_f64.begin(), face_f64.end());
    }
};

}  // namespace

extern "C" {

int32_t occt_to_brep_ir(uint32_t shape_id) {
    g_ir_u32.clear();
    g_ir_f64.clear();
    g_ir_error.clear();
    try {
        OcctKernel* kernel = occt_wasi_kernel_for_extensions();
        if (kernel == nullptr) {
            throw std::runtime_error("kernel not initialized");
        }
        const TopoDS_Shape& shape = kernel->get(shape_id);
        IrBuilder builder;
        bool any_solid = false;
        for (TopExp_Explorer solids(shape, TopAbs_SOLID); solids.More();
             solids.Next()) {
            any_solid = true;
            builder.emit_solid(TopoDS::Solid(solids.Current()));
        }
        if (!any_solid) {
            throw std::runtime_error("shape contains no solid");
        }
        builder.finish();
        g_ir_u32 = std::move(builder.u32);
        g_ir_f64 = std::move(builder.f64);
        return 0;
    } catch (const std::exception& e) {
        g_ir_error = std::string("to_brep_ir: ") + e.what();
        return -1;
    } catch (...) {
        g_ir_error = "to_brep_ir: unknown OCCT failure";
        return -1;
    }
}

int32_t occt_rmesh_ir_u32() {
    return static_cast<int32_t>(reinterpret_cast<intptr_t>(g_ir_u32.data()));
}
uint32_t occt_rmesh_ir_u32_len() {
    return static_cast<uint32_t>(g_ir_u32.size());
}
int32_t occt_rmesh_ir_f64() {
    return static_cast<int32_t>(reinterpret_cast<intptr_t>(g_ir_f64.data()));
}
uint32_t occt_rmesh_ir_f64_len() {
    return static_cast<uint32_t>(g_ir_f64.size());
}
int32_t occt_rmesh_ir_error() {
    return static_cast<int32_t>(reinterpret_cast<intptr_t>(g_ir_error.data()));
}
uint32_t occt_rmesh_ir_error_len() {
    return static_cast<uint32_t>(g_ir_error.size());
}

}  // extern "C"
