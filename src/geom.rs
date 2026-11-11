//! Port of `bouncing-room/src/geometry.js`.
//!
//! Primitive meshes, generated at load time — no asset loading.
//!
//! Every mesh shares one interleaved layout so a single vertex setup works for
//! all three shader programs: `[px, py, pz, nx, ny, nz]` — stride 24 bytes,
//! 6 floats per vertex.
//!
//! Primitives are normalised so their bounding radius is exactly 1. A physics
//! body then only needs a scale factor, and its collision radius is that scale
//! (times a per-shape factor; see [`crate::sim::radius_factor`]).
//!
//! Faithfulness notes for anyone comparing against the JS:
//! - The generators keep `f64` intermediates exactly like the JS (plain number
//!   arrays), and only round to `f32` when packing a [`Vertex`] — that is what
//!   the JS `Float32Array` store does. Degenerate-triangle and pole-row branches
//!   depend on those bits.
//! - `same_point_xy` reproduces the JS `samePoint` quirk of comparing only the
//!   first two components, which is what makes the cone's 9-gon base drop one
//!   zero-area triangle (51 verts, not 54).

use std::collections::HashSet;
use std::f64::consts::{FRAC_PI_2, PI};

use bytemuck::{Pod, Zeroable};

/// Floats per vertex in the interleaved layout (position + normal).
pub const FLOATS_PER_VERT: usize = 6;

/// One vertex: position then normal, 24 bytes, `location 0` and `location 1`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// A mesh: an interleaved vertex list, optional triangle indices, and a
/// `GL_LINES`-style edge list for wireframes / outlines.
///
/// `indices == None` means the vertex list *is* the triangle list.
#[derive(Clone, Debug)]
pub struct Mesh {
    pub name: &'static str,
    pub vertices: Vec<Vertex>,
    pub indices: Option<Vec<u16>>,
    pub edges: Vec<u16>,
}

impl Mesh {
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Length of the triangle index buffer (0 when the mesh is unindexed).
    pub fn index_count(&self) -> usize {
        self.indices.as_ref().map_or(0, |v| v.len())
    }

    pub fn triangle_count(&self) -> usize {
        match &self.indices {
            Some(idx) => idx.len() / 3,
            None => self.vertices.len() / 3,
        }
    }

    /// Number of line segments in [`Mesh::edges`] (the list holds two indices
    /// per segment).
    pub fn edge_pair_count(&self) -> usize {
        self.edges.len() / 2
    }

    /// Largest vertex distance from the origin. Normalised primitives give 1.0.
    pub fn max_radius(&self) -> f32 {
        self.vertices
            .iter()
            .map(|v| Vec3d(v.position[0] as f64, v.position[1] as f64, v.position[2] as f64).len())
            .fold(0.0f64, f64::max) as f32
    }
}

/// A primitive ready to be drawn and simulated.
#[derive(Clone, Debug)]
pub struct ShapeDef {
    pub mesh: Mesh,
    pub color: [f32; 3],
    pub gloss: f32,
}

/// Names, colours and glosses of [`all_primitives`], in the same order.
/// `World` bodies refer to a primitive by index into these (`Body::shape`).
pub const PRIMITIVE_NAMES: [&str; 9] = [
    "sphere",
    "cube",
    "capsule",
    "torus",
    "cone",
    "cylinder",
    "icosahedron",
    "tetrahedron",
    "octahedron",
];

pub const PRIMITIVE_COLORS: [[f32; 3]; 9] = [
    [0.98, 0.42, 0.35],
    [1.0, 0.71, 0.33],
    [0.25, 0.82, 0.75],
    [0.42, 0.71, 1.0],
    [0.71, 0.55, 1.0],
    [0.66, 0.88, 0.37],
    [1.0, 0.48, 0.69],
    [0.95, 0.9, 0.82],
    [0.36, 0.63, 0.98],
];

pub const PRIMITIVE_GLOSS: [f32; 9] = [46.0, 26.0, 64.0, 92.0, 32.0, 32.0, 20.0, 15.0, 18.0];

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
struct Vec3d(f64, f64, f64);

impl Vec3d {
    fn new(a: [f64; 3]) -> Self {
        Vec3d(a[0], a[1], a[2])
    }
    fn to_array(self) -> [f64; 3] {
        [self.0, self.1, self.2]
    }
    fn len(self) -> f64 {
        // JS Math.hypot(x, y, z). Chained two-argument hypot matches to within an
        // ulp and cannot overflow/underflow.
        (self.0).hypot(self.1).hypot(self.2)
    }
    fn dot(self, o: Self) -> f64 {
        self.0 * o.0 + self.1 * o.1 + self.2 * o.2
    }
    fn cross(self, o: Self) -> Self {
        Vec3d(
            self.1 * o.2 - self.2 * o.1,
            self.2 * o.0 - self.0 * o.2,
            self.0 * o.1 - self.1 * o.0,
        )
    }
    fn sub(self, o: Self) -> Self {
        Vec3d(self.0 - o.0, self.1 - o.1, self.2 - o.2)
    }
    fn div(self, k: f64) -> Self {
        // The JS divides each component by the length; keep true division (not
        // multiply-by-reciprocal) so a "unit" normal has the same bits.
        Vec3d(self.0 / k, self.1 / k, self.2 / k)
    }
    fn neg(self) -> Self {
        Vec3d(-self.0, -self.1, -self.2)
    }
}

/// Unpacked builder: positions / normals / triangle indices / candidate edge
/// pairs, all still in `f64`, exactly like the plain JS arrays.
#[derive(Default, Debug)]
struct Builder {
    pos: Vec<[f64; 3]>,
    nrm: Vec<[f64; 3]>,
    idx: Vec<u32>,
    pairs: Vec<(u32, u32)>,
}

impl Builder {
    fn push_tri(&mut self, a: [f64; 3], b: [f64; 3], c: [f64; 3], n: [f64; 3]) {
        self.pos.extend_from_slice(&[a, b, c]);
        self.nrm.extend_from_slice(&[n, n, n]);
    }
}

/// Round to an integer key the way JS `Math.round` does (half up toward +inf),
/// computed in f64 from the f32 vertex component exactly as the JS reads the
/// packed `Float32Array` back out.
fn round_key(x: f32) -> i64 {
    ((x as f64 * 1e4) + 0.5).floor() as i64
}

fn key_of(vertices: &[Vertex], index: u32) -> [i64; 3] {
    let v = &vertices[index as usize];
    [
        round_key(v.position[0]),
        round_key(v.position[1]),
        round_key(v.position[2]),
    ]
}

/// Collapse `(vertexIndex, vertexIndex)` pairs into a de-duplicated `GL_LINES`
/// index list. Keying on position rather than index means meshes whose faces
/// duplicate their vertices (flat shading, UV seams) still get single, clean
/// wireframe lines instead of one line per adjacent face.
fn dedupe_edges(vertices: &[Vertex], pairs: &[(u32, u32)]) -> Vec<u16> {
    let mut seen: HashSet<([i64; 3], [i64; 3])> = HashSet::new();
    let mut lines = Vec::new();
    for &(ia, ib) in pairs {
        let ka = key_of(vertices, ia);
        let kb = key_of(vertices, ib);
        if ka == kb {
            continue; // same point (pole / seam) — zero-length line
        }
        let id = if ka < kb { (ka, kb) } else { (kb, ka) };
        if !seen.insert(id) {
            continue;
        }
        lines.push(ia as u16);
        lines.push(ib as u16);
    }
    lines
}

/// Pack the plain position/normal arrays into the shared interleaved form.
/// `normalize` rescales the mesh to a bounding radius of 1 (off for the room and
/// the helper quad/ring, which are sized by hand).
fn make_mesh(name: &'static str, b: Builder, normalize: bool) -> Mesh {
    let mut max_r = 0.0f64;
    if normalize {
        // True bounding *radius* (not the max coordinate): the physics body
        // treats every shape as a sphere of this size, so a tetrahedron and a
        // cube must agree on what "fits in radius 1" means.
        for p in &b.pos {
            max_r = f64::max(max_r, Vec3d::new(*p).len());
        }
    }
    let inv = if normalize && max_r > 1e-9 { 1.0 / max_r } else { 1.0 };

    let count = b.pos.len().min(b.nrm.len());
    let mut vertices = Vec::with_capacity(count);
    for i in 0..count {
        vertices.push(Vertex {
            position: [
                (b.pos[i][0] * inv) as f32,
                (b.pos[i][1] * inv) as f32,
                (b.pos[i][2] * inv) as f32,
            ],
            normal: [b.nrm[i][0] as f32, b.nrm[i][1] as f32, b.nrm[i][2] as f32],
        });
    }

    // Edges are keyed off the *packed* (and therefore normalised) positions,
    // as in the JS, which calls dedupeEdges on the finished vertex buffer.
    let edges = dedupe_edges(&vertices, &b.pairs);

    Mesh {
        name,
        vertices,
        // An empty index list means "draw the vertex list straight as triangles".
        indices: if b.idx.is_empty() {
            None
        } else {
            Some(b.idx.iter().map(|&i| i as u16).collect())
        },
        edges,
    }
}

// ---------------------------------------------------------------------------
// Surfaces of revolution: sphere, capsule, cone, cylinder
// ---------------------------------------------------------------------------

/// Sample a circular arc in (radius, y) space, used to build profiles.
fn arc(cy: f64, r: f64, from: f64, to: f64, steps: u32) -> Vec<(f64, f64)> {
    let mut pts = Vec::with_capacity(steps as usize + 1);
    for i in 0..=steps {
        let a = from + ((to - from) * i as f64) / steps as f64;
        pts.push((a.sin().abs() * r, cy + a.cos() * r));
    }
    pts
}

/// `profile` is a polyline in (radius, y) space walked top to bottom. Repeating a
/// point makes a sharp shoulder (cylinder caps); tangent-continuous points stay
/// smooth (capsule shoulders).
fn revolve(name: &'static str, profile: Vec<(f64, f64)>, segs: u32, flat: bool) -> Mesh {
    if flat {
        revolve_flat(name, &profile, segs)
    } else {
        revolve_smooth(name, &profile, segs)
    }
}

fn profile_segment_normals(profile: &[(f64, f64)]) -> Vec<Option<(f64, f64)>> {
    let mut out = Vec::with_capacity(profile.len().saturating_sub(1));
    for i in 0..profile.len().saturating_sub(1) {
        let (r0, y0) = profile[i];
        let (r1, y1) = profile[i + 1];
        let dr = r1 - r0;
        let dy = y1 - y0;
        let len = dr.hypot(dy);
        // Traversing the profile top to bottom, rotating the tangent by (-dy, dr)
        // gives the outward 2D normal.
        if len < 1e-9 {
            out.push(None);
        } else {
            out.push(Some((-dy / len, dr / len)));
        }
    }
    out
}

/// JS `samePoint` applied to a (radius, y) profile point.
fn same_point_rp(a: (f64, f64), b: (f64, f64)) -> bool {
    (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9
}

/// JS `samePoint` applied to a *3D* point. The JS helper only ever looks at the
/// first two components, so on a revolved ring this compares x and y and ignores
/// z. That quirk is load-bearing: it is what makes the cone collapse a pole quad
/// and then drop a zero-area triangle.
fn same_point_xy(a: [f64; 3], b: [f64; 3]) -> bool {
    (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9
}

fn revolve_smooth(name: &'static str, profile: &[(f64, f64)], segs: u32) -> Mesh {
    let rows = profile.len();
    let seg_n = profile_segment_normals(profile);

    // Per-profile-point normals: average the two adjacent segment normals unless
    // the shoulder is sharp, in which case keep the one valid normal.
    let mut point_n = Vec::with_capacity(rows);
    for i in 0..rows {
        let a = if i > 0 { seg_n[i - 1] } else { None };
        let b = seg_n.get(i).copied().unwrap_or(None);
        let (nr, ny) = match (a, b) {
            (Some(a), Some(b)) if a.0 * b.0 + a.1 * b.1 > 0.985 => (a.0 + b.0, a.1 + b.1),
            _ => {
                let use_n = a.or(b).unwrap_or((0.0, 1.0));
                (use_n.0, use_n.1)
            }
        };
        let mut len = nr.hypot(ny);
        if len == 0.0 {
            len = 1.0;
        }
        point_n.push((nr / len, ny / len));
    }

    let mut b = Builder::default();
    let s = segs as i64;
    let index = |row: usize, col: i64| -> u32 { (row as i64 * s + ((col % s) + s) % s) as u32 };

    for i in 0..rows {
        let (r, y) = profile[i];
        let (nr, ny) = point_n[i];
        for j in 0..segs {
            let t = (j as f64 / segs as f64) * PI * 2.0;
            let ct = t.cos();
            let st = t.sin();
            b.pos.push([r * ct, y, r * st]);
            b.nrm.push([nr * ct, ny, nr * st]);
        }
    }

    for i in 0..rows - 1 {
        if same_point_rp(profile[i], profile[i + 1]) {
            continue;
        }
        for j in 0..segs {
            let a = index(i, j as i64);
            let bb = index(i, j as i64 + 1);
            let c = index(i + 1, j as i64);
            let d = index(i + 1, j as i64 + 1);
            // The quad collapses to a triangle at a pole row: pick the live diagonal.
            if profile[i].0 == 0.0 {
                b.idx.extend_from_slice(&[a, d, c]);
            } else if profile[i + 1].0 == 0.0 {
                b.idx.extend_from_slice(&[a, bb, d]);
            } else {
                b.idx.extend_from_slice(&[a, bb, d, a, d, c]);
            }
            b.pairs.push((a, bb));
            b.pairs.push((a, c));
        }
    }
    make_mesh(name, b, true)
}

/// Faceted variant: every quad gets its own four vertices and one flat normal,
/// since a shared vertex cannot hold different normals per face.
fn revolve_flat(name: &'static str, profile: &[(f64, f64)], segs: u32) -> Mesh {
    let mut b = Builder::default();

    let at = |row: usize, col: u32| -> [f64; 3] {
        let (r, y) = profile[row];
        let t = (col as f64 / segs as f64) * PI * 2.0;
        [r * t.cos(), y, r * t.sin()]
    };

    for i in 0..profile.len() - 1 {
        if same_point_rp(profile[i], profile[i + 1]) {
            continue;
        }
        for j in 0..segs {
            let a = at(i, j);
            let bb = at(i, j + 1);
            let c = at(i + 1, j);
            let d = at(i + 1, j + 1);

            // A ring row whose first two corners (or last two) coincide is a
            // collapsed quad: it contributes one triangle, not two.
            let (tris, take) = if same_point_xy(a, bb) {
                ([(a, d, c), (a, d, c)], 1)
            } else if same_point_xy(c, d) {
                ([(a, bb, d), (a, bb, d)], 1)
            } else {
                ([(a, bb, d), (a, d, c)], 2)
            };

            for tri in tris.iter().take(take) {
                let (p0, p1, p2) = (Vec3d::new(tri.0), Vec3d::new(tri.1), Vec3d::new(tri.2));
                let u = p1.sub(p0);
                let v = p2.sub(p0);
                let mut n = u.cross(v);
                let len = n.len();
                if len < 1e-9 {
                    continue; // zero-area triangle, nothing to light
                }
                n = n.div(len);
                // Revolved solids are centred on the axis: point normals outward.
                let m = Vec3d(
                    (tri.0[0] + tri.1[0] + tri.2[0]) / 3.0,
                    (tri.0[1] + tri.1[1] + tri.2[1]) / 3.0,
                    (tri.0[2] + tri.1[2] + tri.2[2]) / 3.0,
                );
                if n.dot(m) < 0.0 {
                    n = n.neg();
                }
                let n = n.to_array();
                b.pos.push(tri.0);
                b.pos.push(tri.1);
                b.pos.push(tri.2);
                b.nrm.extend_from_slice(&[n, n, n]);
            }
        }
    }

    // Flat meshes have no index buffer, so the wireframe is derived from the
    // triangle soup: emit each triangle's outline and let makeMesh collapse the
    // edges two adjacent faces share.
    let tri_verts = b.pos.len() as u32;
    let mut t = 0u32;
    while t + 2 < tri_verts {
        b.pairs.push((t, t + 1));
        b.pairs.push((t + 1, t + 2));
        b.pairs.push((t + 2, t));
        t += 3;
    }
    make_mesh(name, b, true)
}

pub fn sphere_mesh() -> Mesh {
    revolve("sphere", arc(0.0, 1.0, 0.0, PI, 18), 30, false)
}

pub fn capsule_mesh() -> Mesh {
    let hh = 0.46;
    let r = 0.54;
    let mut profile = arc(hh, r, 0.0, FRAC_PI_2, 7);
    profile.push((r, -hh));
    profile.extend_from_slice(&arc(-hh, r, FRAC_PI_2, PI, 7));
    revolve("capsule", profile, 26, false)
}

pub fn cone_mesh() -> Mesh {
    let r = 0.86;
    let h = 1.5;
    revolve(
        "cone",
        vec![(0.0, h / 2.0), (r, -h / 2.0), (0.0, -h / 2.0)],
        9,
        true,
    )
}

pub fn cylinder_mesh() -> Mesh {
    let r = 0.7;
    let h = 1.15;
    let profile = vec![
        (0.0, h / 2.0),
        (r, h / 2.0),
        (r, h / 2.0),
        (r, -h / 2.0),
        (r, -h / 2.0),
        (0.0, -h / 2.0),
    ];
    revolve("cylinder", profile, 14, true)
}

pub fn torus_mesh() -> Mesh {
    let rr = 0.7;
    let tube = 0.3; // R + r == 1, so the bounding radius is already 1
    let major_steps: u32 = 22;
    let minor_steps: u32 = 14;
    let mut b = Builder::default();

    // Both directions wrap: the tube is a closed torus in u and v.
    let wrap = |k: i64, m: u32| -> u32 { (((k % m as i64) + m as i64) % m as i64) as u32 };
    let index = |i: i64, j: i64| -> u32 { wrap(i, major_steps) * minor_steps + wrap(j, minor_steps) };

    for i in 0..major_steps {
        let a = (i as f64 / major_steps as f64) * PI * 2.0;
        let ca = a.cos();
        let sa = a.sin();
        for j in 0..minor_steps {
            let bb = (j as f64 / minor_steps as f64) * PI * 2.0;
            let cb = bb.cos();
            let sb = bb.sin();
            b.nrm.push([cb * ca, sb, cb * sa]);
            b.pos.push([(rr + tube * cb) * ca, tube * sb, (rr + tube * cb) * sa]);
        }
    }
    for i in 0..major_steps as i64 {
        for j in 0..minor_steps as i64 {
            let a = index(i, j);
            let bb = index(i, j + 1);
            let c = index(i + 1, j);
            let d = index(i + 1, j + 1);
            b.idx.extend_from_slice(&[a, bb, d, a, d, c]);
            b.pairs.push((a, bb));
            b.pairs.push((a, c));
        }
    }
    make_mesh("torus", b, true)
}

// ---------------------------------------------------------------------------
// Flat-shaded polyhedra. Face winding is repaired on the fly: a convex solid
// centred on the origin has face normals pointing away from the centre, so any
// triangle whose normal points inward is flipped.
// ---------------------------------------------------------------------------

fn polyhedron(name: &'static str, verts: &[[f64; 3]], faces: &[[u32; 3]]) -> Mesh {
    let mut b = Builder::default();
    for f in faces {
        let a = verts[f[0] as usize];
        let bb = verts[f[1] as usize];
        let c = verts[f[2] as usize];
        let va = Vec3d::new(a);
        let vb = Vec3d::new(bb);
        let vc = Vec3d::new(c);
        let mut n = vb.sub(va).cross(vc.sub(va));
        let len = n.len();
        let len = if len == 0.0 { 1.0 } else { len };
        n = n.div(len);

        let cent = Vec3d(
            (a[0] + bb[0] + c[0]) / 3.0,
            (a[1] + bb[1] + c[1]) / 3.0,
            (a[2] + bb[2] + c[2]) / 3.0,
        );
        if n.dot(cent) < 0.0 {
            n = n.neg();
            b.push_tri(c, bb, a, n.to_array());
        } else {
            b.push_tri(a, bb, c, n.to_array());
        }
    }
    let tri_verts = b.pos.len() as u32;
    let mut t = 0u32;
    while t + 2 < tri_verts {
        b.pairs.push((t, t + 1));
        b.pairs.push((t + 1, t + 2));
        b.pairs.push((t + 2, t));
        t += 3;
    }
    make_mesh(name, b, true)
}

pub fn cube_mesh() -> Mesh {
    let v = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let faces = [
        [0, 1, 2],
        [0, 2, 3],
        [5, 4, 7],
        [5, 7, 6],
        [4, 0, 3],
        [4, 3, 7],
        [1, 5, 6],
        [1, 6, 2],
        [3, 2, 6],
        [3, 6, 7],
        [4, 5, 1],
        [4, 1, 0],
    ];
    polyhedron("cube", &v, &faces)
}

pub fn tetra_mesh() -> Mesh {
    let v = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ];
    polyhedron(
        "tetrahedron",
        &v,
        &[[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
    )
}

pub fn octa_mesh() -> Mesh {
    let v = [
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let faces = [
        [0, 2, 4],
        [0, 4, 3],
        [0, 3, 5],
        [0, 5, 2],
        [1, 4, 2],
        [1, 3, 4],
        [1, 5, 3],
        [1, 2, 5],
    ];
    polyhedron("octahedron", &v, &faces)
}

pub fn icosa_mesh() -> Mesh {
    let t = (1.0 + 5.0f64.sqrt()) / 2.0;
    let v = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    let faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    polyhedron("icosahedron", &v, &faces)
}

// ---------------------------------------------------------------------------
// Room and helper geometry
// ---------------------------------------------------------------------------

/// An axis-aligned box wound so that culling FRONT faces leaves the interior
/// visible: orbit outside and the near walls fall away for a cutaway view, while
/// a camera inside the room is fully enclosed. Normals are stored pointing *into*
/// the room because those are the surfaces being lit.
///
/// Not normalised: the room is sized by hand.
pub fn room_mesh(half_x: f32, half_y: f32, half_z: f32) -> Mesh {
    let hx = half_x as f64;
    let hz = half_z as f64;
    let y0 = 0.0;
    let y1 = half_y as f64;
    struct Face {
        pts: [[f64; 3]; 4],
        n: [f64; 3],
    }
    let faces = [
        // +z wall
        Face {
            pts: [[-hx, y0, hz], [hx, y0, hz], [hx, y1, hz], [-hx, y1, hz]],
            n: [0.0, 0.0, -1.0],
        },
        // -z wall
        Face {
            pts: [[hx, y0, -hz], [-hx, y0, -hz], [-hx, y1, -hz], [hx, y1, -hz]],
            n: [0.0, 0.0, 1.0],
        },
        // +x wall
        Face {
            pts: [[hx, y0, hz], [hx, y0, -hz], [hx, y1, -hz], [hx, y1, hz]],
            n: [-1.0, 0.0, 0.0],
        },
        // -x wall
        Face {
            pts: [[-hx, y0, -hz], [-hx, y0, hz], [-hx, y1, hz], [-hx, y1, -hz]],
            n: [1.0, 0.0, 0.0],
        },
        // floor
        Face {
            pts: [[-hx, y0, -hz], [hx, y0, -hz], [hx, y0, hz], [-hx, y0, hz]],
            n: [0.0, 1.0, 0.0],
        },
        // ceiling
        Face {
            pts: [[-hx, y1, hz], [hx, y1, hz], [hx, y1, -hz], [-hx, y1, -hz]],
            n: [0.0, -1.0, 0.0],
        },
    ];
    let mut b = Builder::default();
    let mut v = 0u32;
    for f in faces.iter() {
        for p in f.pts.iter() {
            b.pos.push(*p);
            b.nrm.push(f.n);
        }
        b.idx.extend_from_slice(&[v, v + 1, v + 2, v, v + 2, v + 3]);
        b.pairs.push((v, v + 1));
        b.pairs.push((v + 1, v + 2));
        b.pairs.push((v + 2, v + 3));
        b.pairs.push((v + 3, v));
        v += 4;
    }
    make_mesh("room", b, false)
}

/// Unit quad spanning [-1,1] in XZ, used for shadow blobs and the ceiling panel.
pub fn quad_mesh() -> Mesh {
    let corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let mut b = Builder::default();
    for c in corners.iter() {
        b.pos.push([c[0], 0.0, c[1]]);
        b.nrm.push([0.0, 1.0, 0.0]);
    }
    b.idx.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    make_mesh("quad", b, false)
}

/// Unit circle in XZ as `GL_LINES` pairs; impact rings are drawn from this.
pub fn ring_mesh(segments: u32) -> Mesh {
    let segments = segments.max(1);
    let mut b = Builder::default();
    for i in 0..segments {
        let a = (i as f64 / segments as f64) * PI * 2.0;
        b.pos.push([a.cos(), 0.0, a.sin()]);
        b.nrm.push([0.0, 1.0, 0.0]);
    }
    for i in 0..segments {
        b.pairs.push((i, (i + 1) % segments));
    }
    make_mesh("ring", b, false)
}

pub fn all_primitives() -> Vec<ShapeDef> {
    let meshes = [
        sphere_mesh(),
        cube_mesh(),
        capsule_mesh(),
        torus_mesh(),
        cone_mesh(),
        cylinder_mesh(),
        icosa_mesh(),
        tetra_mesh(),
        octa_mesh(),
    ];
    meshes
        .into_iter()
        .enumerate()
        .map(|(i, mesh)| ShapeDef {
            mesh,
            color: PRIMITIVE_COLORS[i],
            gloss: PRIMITIVE_GLOSS[i],
        })
        .collect()
}
