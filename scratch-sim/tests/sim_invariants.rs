//! Invariant tests for the native port, ported from the JS oracle
//! `bouncing-room/scripts/check.mjs`.
//!
//! These talk to the crate through its public API only — geometry counts, mesh
//! sanity, and 3600 ticks of a full room of bodies — so they keep biting if a
//! contact model detail changes. Run with `--nocapture` to get the mesh table the
//! integrator can diff against the JS output.

use glam::{Mat4, Vec3};

use scratch_sim::sim::{radius_factor, MAX_SPIN, SpawnOpts, World};
use scratch_sim::{all_primitives, quad_mesh, ring_mesh, room_mesh, Mesh};

/// Room constants from the port contract.
const HALF_X: f32 = 7.2;
const HALF_Y: f32 = 5.4;
const HALF_Z: f32 = 7.2;

fn new_world() -> World {
    World::new(HALF_X, HALF_Y, HALF_Z)
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// Exact mesh structure, taken from the JS build's own output
/// (`node scripts/check.mjs`). If a generator changes shape, this fails.
const EXPECTED: [(&str, usize, Option<usize>, usize, usize); 9] = [
    // name           verts  triangle indices  edge pairs  triangles
    ("sphere", 570, Some(3150), 1050, 1050),
    ("cube", 36, None, 18, 12),
    ("capsule", 442, Some(2262), 754, 754),
    ("torus", 308, Some(1848), 616, 616),
    ("cone", 51, None, 27, 17),
    ("cylinder", 168, None, 84, 56),
    ("icosahedron", 60, None, 30, 20),
    ("tetrahedron", 12, None, 6, 4),
    ("octahedron", 24, None, 12, 8),
];

/// The number the JS check prints in its "tris-idx" column:
/// `mesh.indices ? mesh.indices.length : mesh.vertices.length / 3 * 3`.
/// For an unindexed (flat / triangle-soup) mesh that collapses to the *float*
/// count, which is why the contract quotes "cube 36/216" or "cone 51" with no
/// index buffer. Reproduced here so the two tables diff cleanly; the real
/// structure is asserted separately.
fn js_tris_idx_column(m: &Mesh) -> usize {
    match &m.indices {
        Some(v) => v.len(),
        None => m.vertices.len() * 6,
    }
}

/// Prints the same columns, in the same widths, as `scripts/check.mjs`.
fn print_row(m: &Mesh) {
    println!(
        "  {:<13} {:>5} verts {:>6} tris-idx {:>5} edges  ({} tris, {})",
        m.name,
        m.vertex_count(),
        js_tris_idx_column(m),
        m.edge_pair_count(),
        m.triangle_count(),
        if m.indices.is_some() { "indexed" } else { "triangle soup" },
    );
}

#[test]
fn mesh_counts_match_the_js_exactly() {
    println!("  name             verts    tris-idx   edges");
    for p in all_primitives() {
        print_row(&p.mesh);
    }
    print_row(&room_mesh(HALF_X, HALF_Y, HALF_Z));
    print_row(&quad_mesh());
    print_row(&ring_mesh(40));

    let prims = all_primitives();
    assert_eq!(prims.len(), 9, "nine primitives");
    for (i, p) in prims.iter().enumerate() {
        let (name, verts, indices, edges, tris) = EXPECTED[i];
        assert_eq!(p.mesh.name, name, "primitive {i} must be in JS order");
        assert_eq!(p.mesh.vertex_count(), verts, "{name}: vertex count");
        assert_eq!(p.mesh.index_count(), indices.unwrap_or(0), "{name}: index count");
        assert_eq!(
            p.mesh.indices.is_some(),
            indices.is_some(),
            "{name}: indexed vs triangle-soup (None means the vertex list is the triangle list)"
        );
        assert_eq!(p.mesh.edge_pair_count(), edges, "{name}: unique wireframe edges");
        assert_eq!(p.mesh.triangle_count(), tris, "{name}: triangles");
    }

    // The column the contract's table quotes, reproduced exactly.
    let js_column: Vec<(&str, usize)> = all_primitives()
        .iter()
        .map(|p| (p.mesh.name, js_tris_idx_column(&p.mesh)))
        .collect();
    let expected_column = [
        ("sphere", 3150),
        ("cube", 216),
        ("capsule", 2262),
        ("torus", 1848),
        ("cone", 306),
        ("cylinder", 1008),
        ("icosahedron", 360),
        ("tetrahedron", 72),
        ("octahedron", 144),
    ];
    assert_eq!(js_column.as_slice(), expected_column.as_slice(), "JS 'tris-idx' column");
}

#[test]
fn every_primitive_has_a_unit_bounding_radius() {
    for p in all_primitives() {
        let r = p.mesh.max_radius();
        assert!(
            (r - 1.0).abs() < 1e-6,
            "{}: bounding radius is 1 (got {r})",
            p.mesh.name
        );
    }
    // The room and the helpers are sized by hand, not normalised.
    let room = room_mesh(HALF_X, HALF_Y, HALF_Z);
    assert!(room.max_radius() > HALF_X, "room keeps its authored size");
    let ring = ring_mesh(40);
    assert!((ring.max_radius() - 1.0).abs() < 1e-6, "ring is a unit circle");
}

#[test]
fn normals_are_unit_length_and_not_degenerate() {
    for p in all_primitives() {
        let mut bad = 0;
        let mut max_dot = f32::NEG_INFINITY;
        for v in &p.mesh.vertices {
            let len = Vec3::from(v.normal).length();
            if (len - 1.0).abs() > 1e-4 {
                bad += 1;
            }
            max_dot = max_dot.max(Vec3::from(v.normal).dot(Vec3::from(v.position)));
        }
        assert_eq!(bad, 0, "{}: all normals unit length", p.mesh.name);
        assert!(
            max_dot > 0.3,
            "{}: normals not degenerate (max dot {max_dot})",
            p.mesh.name
        );
        assert_eq!(p.mesh.edges.len() % 2, 0, "{}: edge list is pairs", p.mesh.name);
    }
}

#[test]
fn normals_point_outward_for_star_convex_solids() {
    for p in all_primitives() {
        // A tube's inner half faces the axis, so it is checked separately.
        if p.mesh.name == "torus" {
            continue;
        }
        let mut min_dot = f32::INFINITY;
        for v in &p.mesh.vertices {
            let dot = Vec3::from(v.normal).dot(Vec3::from(v.position));
            min_dot = min_dot.min(dot);
        }
        assert!(
            min_dot > -1e-6,
            "{}: normals face outward (min dot {min_dot})",
            p.mesh.name
        );
    }
}

#[test]
fn torus_normals_point_out_of_the_tube_centre_line() {
    let torus = all_primitives().into_iter().find(|p| p.mesh.name == "torus").unwrap().mesh;
    let r_major = 0.7;
    let mut worst = 0.0f32;
    for v in &torus.vertices {
        let p = Vec3::from(v.position);
        let n = Vec3::from(v.normal);
        // Correct normal for a tube: radially out from the centre ring, not the origin.
        let rl = p.x.hypot(p.z).max(1e-9);
        let centre = Vec3::new(p.x / rl * r_major, 0.0, p.z / rl * r_major);
        let d = (p - centre).normalize();
        worst = worst.max(n.dot(d) - 1.0).abs();
    }
    assert!(worst < 1e-4, "torus: normals follow the tube centre line (err {worst})");
}

#[test]
fn indices_are_in_range_and_16_bit_safe() {
    let mut meshes: Vec<Mesh> = all_primitives().into_iter().map(|p| p.mesh).collect();
    meshes.push(room_mesh(HALF_X, HALF_Y, HALF_Z));
    meshes.push(quad_mesh());
    meshes.push(ring_mesh(40));
    for m in meshes {
        let n = m.vertex_count();
        if let Some(idx) = &m.indices {
            assert_eq!(idx.len() % 3, 0, "{}: index count divisible by 3", m.name);
            let worst = *idx.iter().max().unwrap();
            assert!((worst as usize) < n, "{}: index {worst} within vertex count {n}", m.name);
        }
        assert!(n < 65535, "{}: fits 16-bit indices ({n} verts)", m.name);
        assert!(m.edges.len() < 60000, "{}: edges fit 16-bit indices", m.name);
    }
}

#[test]
fn room_is_wound_inside_out() {
    let room = room_mesh(HALF_X, HALF_Y, HALF_Z);
    assert_eq!(room.indices.as_ref().unwrap().len(), 36, "6 quads -> 36 indices");
    assert_eq!(room.edge_pair_count(), 12, "12 unique box edges");

    // Normals are stored pointing *into* the room, because those are the
    // surfaces being lit.
    let centre = Vec3::new(0.0, HALF_Y * 0.5, 0.0);
    let mut inward = 0;
    for v in &room.vertices {
        let outward = Vec3::from(v.position) - centre;
        if Vec3::from(v.normal).dot(-outward) > 0.0 {
            inward += 1;
        }
    }
    assert_eq!(room.vertex_count(), 24, "24 room verts");
    assert_eq!(inward, 24, "all room normals point inward");

    // The box is not normalised: it is sized by hand.
    let max_abs = room
        .vertices
        .iter()
        .map(|v| v.position[0].abs().max(v.position[2].abs()))
        .fold(0.0f32, f32::max);
    assert!((max_abs - HALF_X).abs() < 1e-6, "room spans +/- half X/Z");
}

#[test]
fn ring_and_quad_helpers() {
    let ring = ring_mesh(40);
    assert_eq!(ring.vertex_count(), 40, "40 ring verts");
    assert_eq!(ring.edges.len(), 80, "40 line pairs");
    assert_eq!(ring.indices, None, "ring is a line list, not triangles");
    assert_eq!(ring_mesh(8).edge_pair_count(), 8, "segment count follows the request");

    let quad = quad_mesh();
    assert_eq!(quad.vertex_count(), 4, "quad has 4 corners");
    assert_eq!(quad.index_count(), 6, "quad is two triangles");
    assert_eq!(quad.edges.len(), 0, "quad asks for no wireframe");
    for v in &quad.vertices {
        assert_eq!(v.normal, [0.0, 1.0, 0.0], "quad faces up");
        assert!(v.position[0].abs() <= 1.0 && v.position[2].abs() <= 1.0, "unit quad in XZ");
        assert_eq!(v.position[1], 0.0);
    }
}

#[test]
fn radius_factors_match_the_contract() {
    let expected = [
        ("sphere", 1.0f32),
        ("cube", 0.82),
        ("capsule", 0.9),
        ("torus", 0.86),
        ("cone", 0.82),
        ("cylinder", 0.85),
        ("icosahedron", 0.95),
        ("tetrahedron", 0.74),
        ("octahedron", 0.86),
    ];
    for (name, want) in expected {
        assert_eq!(radius_factor(name), want, "radius factor for {name}");
    }
    // Unknown names fall back like the JS `?? 0.85`.
    assert_eq!(radius_factor("dodecahedron"), 0.85);

    // Collision radius is scale * factor, against a mesh of bounding radius 1.
    let mut w = new_world();
    for (i, p) in all_primitives().iter().enumerate() {
        let idx = w.spawn(i, SpawnOpts::with_scale(0.5));
        let b = w.bodies[idx];
        assert_eq!(b.shape, i);
        assert!(
            (b.radius - 0.5 * radius_factor(p.mesh.name)).abs() < 1e-6,
            "{}: radius from scale * factor",
            p.mesh.name
        );
        assert!((p.mesh.max_radius() - 1.0).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

/// 3600 steps of a full room of bodies: nothing NaNs, nothing escapes, nothing
/// runs away, and every rotation stays an honest rotation.
#[test]
fn long_run_holds_the_room_together() {
    let mut w = new_world();
    w.set_seed(0xC0FF_EE00_0000_0001);
    for i in 0..18 {
        w.record(i % 9, SpawnOpts::default());
    }
    assert_eq!(w.bodies.len(), 18);
    w.settle();

    let dt = 1.0f32 / 60.0;
    let mut max_speed = 0.0f32;
    let mut max_spin = 0.0f32;
    let mut escapes = 0;
    let mut non_finite = 0;
    let mut bad_rotation = 0;
    let mut impacts_seen = 0usize;

    for _step in 0..3600 {
        w.step(dt);
        impacts_seen += w.impacts.len();
        for b in &w.bodies {
            for c in [
                b.pos.x,
                b.pos.y,
                b.pos.z,
                b.vel.x,
                b.vel.y,
                b.vel.z,
                b.ang_vel.x,
                b.ang_vel.y,
                b.ang_vel.z,
            ] {
                if !c.is_finite() {
                    non_finite += 1;
                }
            }
            let slack = b.scale * 1.05;
            if b.pos.x.abs() > HALF_X + slack || b.pos.z.abs() > HALF_Z + slack {
                escapes += 1;
            }
            if b.pos.y < -slack || b.pos.y > HALF_Y + slack {
                escapes += 1;
            }
            max_speed = max_speed.max(b.vel.length());
            max_spin = max_spin.max(b.ang_vel.length());

            // Rotation must stay orthonormal: every column is a unit vector and
            // they stay mutually perpendicular.
            let cols = b.rot.to_cols_array_2d();
            for c in 0..3 {
                let v = Vec3::new(cols[c][0], cols[c][1], cols[c][2]);
                if (v.length() - 1.0).abs() > 1e-3 {
                    bad_rotation += 1;
                }
            }
            let c0 = Vec3::from_array([cols[0][0], cols[0][1], cols[0][2]]);
            let c1 = Vec3::from_array([cols[1][0], cols[1][1], cols[1][2]]);
            let c2 = Vec3::from_array([cols[2][0], cols[2][1], cols[2][2]]);
            if c0.dot(c1).abs() > 1e-3 || c0.dot(c2).abs() > 1e-3 || c1.dot(c2).abs() > 1e-3 {
                bad_rotation += 1;
            }
        }
    }

    let avg_y = w.bodies.iter().map(|b| b.pos.y).sum::<f32>() / w.bodies.len() as f32;
    let resting = w.bodies.iter().filter(|b| b.vel.length() < 0.4).count();
    println!(
        "  after 60s: avg height {avg_y:.2}, {resting} bodies at rest, \
         max speed {max_speed:.1}, max |omega| {max_spin:.2}"
    );

    assert_eq!(non_finite, 0, "no NaN over 3600 steps");
    assert_eq!(bad_rotation, 0, "rotation columns stay orthonormal");
    assert_eq!(escapes, 0, "no body left the room");
    assert!(max_speed < 200.0, "speeds stay bounded (max {max_speed})");
    // Spin is only ever produced by contacts, which clamp it.
    assert!(max_spin <= MAX_SPIN + 0.01, "spin stays under MAX_SPIN (max {max_spin})");
    assert!(impacts_seen > 0, "impacts were recorded");
    // Gravity should leave bodies resting near the floor, not jittering forever.
    assert!(avg_y < HALF_Y * 0.55, "bodies fall to the lower half (avg y {avg_y})");
    // The JS build settles all 18 to a stop in the same 60s; anything close to
    // that here means contacts are dissipating rather than jittering.
    assert!(resting >= 16, "the pile settles ({resting}/18 at rest)");
}

/// A long spin must not shear the orientation matrix: the JS keeps a
/// Gram-Schmidt pass after every exponential-map step for exactly this.
#[test]
fn fast_spin_keeps_the_rotation_matrix_orthonormal() {
    let mut w = new_world();
    w.gravity = 0.0;
    w.drag = 0.0;
    w.collide = false;
    let i = w.spawn(0, SpawnOpts::with_scale(0.3));
    let r = w.bodies[i].radius;
    w.bodies[i].pos = Vec3::new(0.0, HALF_Y - r - 0.2, 0.0);
    w.bodies[i].vel = Vec3::ZERO;
    w.bodies[i].ang_vel = Vec3::new(9.0, -7.0, 5.0); // near MAX_SPIN, tumbling hard
    for _ in 0..3600 {
        w.step(1.0 / 60.0);
    }
    let cols = w.bodies[i].rot.to_cols_array_2d();
    let cols: [Vec3; 3] = [
        Vec3::from_array([cols[0][0], cols[0][1], cols[0][2]]),
        Vec3::from_array([cols[1][0], cols[1][1], cols[1][2]]),
        Vec3::from_array([cols[2][0], cols[2][1], cols[2][2]]),
    ];
    for c in cols {
        assert!((c.length() - 1.0).abs() < 1e-4, "column stays unit length: {}", c.length());
    }
    assert!(cols[0].dot(cols[1]).abs() < 1e-4);
    assert!(cols[0].dot(cols[2]).abs() < 1e-4);
    assert!(cols[1].dot(cols[2]).abs() < 1e-4);
    // det(R) = +1, i.e. still a rotation, not a reflection or a shear.
    let det = cols[0].dot(cols[1].cross(cols[2]));
    assert!((det - 1.0).abs() < 1e-4, "still a proper rotation (det {det})");
}

/// Every angular-velocity step ends with a Gram-Schmidt pass, so an orientation
/// that has picked up drift (shear / scale) is pulled straight again on the next
/// tick rather than shearing further. This is the only test that proves the
/// re-orthonormalisation actually runs: multiplying by a fresh rotation matrix
/// would look fine on its own for thousands of steps.
#[test]
fn a_drifted_orientation_is_pulled_back_to_a_rotation() {
    let mut w = new_world();
    w.gravity = 0.0;
    w.drag = 0.0;
    w.collide = false;
    let i = w.spawn(0, SpawnOpts::with_scale(0.3));
    // Deliberately not a rotation: ~2-4% shear and scale, plus a 3% cross-term.
    w.bodies[i].rot = Mat4::from_cols_array_2d(&[
        [1.02, 0.01, 0.0, 0.0],
        [0.03, 0.97, 0.0, 0.0],
        [0.0, 0.05, 1.04, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    // A spin is required: the exponential map is a no-op below 1e-9 rad.
    w.bodies[i].ang_vel = Vec3::new(0.3, -0.2, 0.5);

    w.step(1.0 / 60.0);

    let c = w.bodies[i].rot.to_cols_array_2d();
    let cols: [Vec3; 3] = [
        Vec3::new(c[0][0], c[0][1], c[0][2]),
        Vec3::new(c[1][0], c[1][1], c[1][2]),
        Vec3::new(c[2][0], c[2][1], c[2][2]),
    ];
    println!(
        "  re-orthonormalised column lengths: {:?}",
        [cols[0].length(), cols[1].length(), cols[2].length()]
    );
    for v in cols {
        assert!(
            (v.length() - 1.0).abs() < 1e-5,
            "the shear is gone: column length {} (was off by ~2e-2)",
            v.length()
        );
    }
    assert!(cols[0].dot(cols[1]).abs() < 1e-5, "columns orthogonal");
    assert!(cols[0].dot(cols[2]).abs() < 1e-5, "columns orthogonal");
    assert!(cols[1].dot(cols[2]).abs() < 1e-5, "columns orthogonal");
    assert!((cols[0].dot(cols[1].cross(cols[2])) - 1.0).abs() < 1e-5, "still a proper rotation");
    // The Gram-Schmidt pass only touches the 3x3 part, as in the JS.
    assert_eq!(w.bodies[i].rot.to_cols_array_2d()[3], [0.0, 0.0, 0.0, 1.0]);
}

/// Contacts ease a body toward rolling spin, which can overshoot the spin cap
/// hard; `clampSpin` is what keeps a fast slide sane.
#[test]
fn contact_spin_is_capped_at_max_spin() {
    let mut w = new_world();
    w.drag = 0.0;
    w.restitution = 0.0;
    let b = w.record(0, SpawnOpts::with_scale(0.6));
    let r = w.bodies[b].radius;
    w.bodies[b].pos = Vec3::new(0.0, r, 0.0);
    // Fast enough that the rolling target omega = |v| / r is far over MAX_SPIN,
    // so the clamp has to bite.
    w.bodies[b].vel = Vec3::new(30.0, 0.0, 0.0);
    w.bodies[b].ang_vel = Vec3::ZERO;
    assert!(
        30.0 / r > MAX_SPIN * 3.0,
        "the rolling target really is over the cap (|v|/r = {}",
        30.0 / r
    );

    let mut observed = 0.0f32;
    for _ in 0..60 {
        w.step(1.0 / 60.0);
        observed = observed.max(w.bodies[b].ang_vel.length());
    }
    println!("  slide-to-roll spin peaked at {observed} (cap {MAX_SPIN})");
    assert!(
        observed >= MAX_SPIN - 0.01,
        "the cap is doing real work here (peak {observed})"
    );
    assert!(
        observed <= MAX_SPIN + 1e-3,
        "contact spin never exceeds MAX_SPIN (peak {observed})"
    );
    // The body is dragged toward rolling, so |omega| * r closes in on |v|.
    let body = w.bodies[b];
    assert!(
        body.ang_vel.length() * body.radius > body.vel.length() * 0.5,
        "friction is spinning it up as it slows (|v| {}, |omega| {})",
        body.vel.length(),
        body.ang_vel.length()
    );
}

/// Contacts must remove energy, never add it: bounce apexes have to fall.
#[test]
fn bounce_apexes_never_rise() {
    let mut w = new_world();
    w.drag = 0.0;
    w.restitution = 0.8;
    let ball = w.record(0, SpawnOpts::with_scale(0.6));
    w.bodies[ball].pos = Vec3::new(0.0, 5.0, 0.0);
    w.bodies[ball].vel = Vec3::ZERO;
    w.bodies[ball].ang_vel = Vec3::ZERO;

    let dt = 1.0f32 / 60.0;
    let mut apexes: Vec<f32> = Vec::new();
    let mut prev_y = 5.0f32;
    let mut rising = false;
    for _ in 0..600 {
        w.step(dt);
        let y = w.bodies[ball].pos.y;
        if rising && y < prev_y {
            apexes.push(y);
            rising = false;
        }
        if !rising && y > prev_y {
            rising = true;
        }
        prev_y = y;
    }
    println!("  bounce apexes: {apexes:?}");
    assert!(apexes.len() >= 2, "ball bounces more than once ({} apexes)", apexes.len());
    for i in 1..apexes.len() {
        assert!(
            apexes[i] <= apexes[i - 1] + 1e-3,
            "bounce apexes never rise: {:?}",
            apexes
        );
    }
}

/// Zero restitution must come to rest rather than bounce forever.
#[test]
fn zero_restitution_body_settles_on_the_floor() {
    let mut w = new_world();
    w.restitution = 0.0;
    let b = w.record(0, SpawnOpts::with_scale(0.6));
    w.bodies[b].pos = Vec3::new(0.0, 5.0, 0.0);
    w.bodies[b].vel = Vec3::ZERO;
    w.bodies[b].ang_vel = Vec3::ZERO;
    for _ in 0..900 {
        w.step(1.0 / 60.0);
    }
    let body = w.bodies[b];
    println!(
        "  settled at y {:.3} (radius {:.3}), |v| {:.4}, |omega| {:.4}",
        body.pos.y,
        body.radius,
        body.vel.length(),
        body.ang_vel.length()
    );
    assert!(
        (body.pos.y - body.radius).abs() < 0.05,
        "zero-bounce body settles on the floor (y {}, r {})",
        body.pos.y,
        body.radius
    );
    assert!(body.vel.length() < 0.2, "spindle-free zero-bounce body stops (|v| {})", body.vel.length());
    assert!(body.ang_vel.length() < 0.2, "and stays unspun (|omega| {})", body.ang_vel.length());
    // It should not have drifted sideways either.
    assert!(body.pos.x.abs() < 0.05 && body.pos.z.abs() < 0.05, "no random creep");
}

/// Spin resting against the floor must convert into travel, not vanish: a
/// topspinning ball in contact rolls away.
#[test]
fn spin_on_the_floor_becomes_travel() {
    let mut w = new_world();
    w.restitution = 0.0;
    w.drag = 0.0;
    let b = w.record(0, SpawnOpts::with_scale(0.6));
    let r = w.bodies[b].radius;
    w.bodies[b].pos = Vec3::new(0.0, r, 0.0);
    w.bodies[b].vel = Vec3::ZERO;
    // Topspin about +z drives +x travel (rolling needs omega = (n x v)/r).
    w.bodies[b].ang_vel = Vec3::new(0.0, 0.0, -6.0);
    for _ in 0..120 {
        w.step(1.0 / 60.0);
    }
    let body = w.bodies[b];
    println!(
        "  spin->travel: vx {:.2}, |omega| {:.2}, x {:.2}",
        body.vel.x,
        body.ang_vel.length(),
        body.pos.x
    );
    assert!(body.vel.x > 0.5, "spin becomes travel (vx {})", body.vel.x);
    assert!(body.vel.x > 0.0, "topspin drives +x, not -x (vx {})", body.vel.x);
    assert!(
        body.ang_vel.length() < 6.0,
        "and the spin is bled down (|omega| {})",
        body.ang_vel.length()
    );
    assert!(body.pos.x > 0.0, "it rolls forward (x {})", body.pos.x);
}

/// A sliding ball has to be dragged into *rolling* spin by floor friction, in
/// the correct direction (moving +x means negative spin about +z).
///
/// This is the invariant that catches friction applied to tangential velocity
/// instead of to the slip at the contact point.
#[test]
fn a_sliding_ball_is_dragged_into_rolling() {
    let mut w = new_world();
    w.drag = 0.0;
    w.restitution = 0.2;
    let b = w.record(0, SpawnOpts::with_scale(0.6));
    w.bodies[b].pos = Vec3::new(0.0, 0.7, 0.0);
    w.bodies[b].vel = Vec3::new(4.0, 0.0, 0.0);
    w.bodies[b].ang_vel = Vec3::ZERO;
    for _ in 0..60 {
        w.step(1.0 / 60.0);
    }
    let body = w.bodies[b];
    let slide = body.vel.length();
    let surface = body.ang_vel.length() * body.radius;
    println!(
        "  slide test: |v| {:.2}, omega*r {:.2}, omega_z {:.2}, x {:.2}",
        slide, surface, body.ang_vel.z, body.pos.x
    );
    assert!(body.ang_vel.z < -0.5, "+x slide spins the ball the right way (omega_z {})", body.ang_vel.z);
    assert!(slide > 2.5, "rolling contact keeps the ball moving (|v| {slide})");
    assert!(
        (surface - slide).abs() < slide * 0.5,
        "friction pulls the ball into rolling, not overspin (surface {surface} vs |v| {slide})"
    );
    assert!(
        body.pos.x > 2.5 && body.pos.x < HALF_X - body.radius + 0.01,
        "ball slides forward and stays in the room (x {})",
        body.pos.x
    );
    // Rolling without slipping also means the transverse spin components stay put.
    assert!(body.ang_vel.x.abs() < 0.5 && body.ang_vel.y.abs() < 0.5, "no bogus spin axes");
}

/// A truly head-on wall hit has no tangential slip and therefore no moment arm.
/// Gravity is off so the approach stays purely normal.
#[test]
fn head_on_wall_hit_adds_no_spin_and_reverses_outward() {
    let mut w = new_world();
    w.drag = 0.0;
    w.gravity = 0.0;
    let b = w.record(0, SpawnOpts::with_scale(0.6));
    w.bodies[b].pos = Vec3::new(-HALF_X + 2.0, HALF_Y - 1.0, 0.0);
    w.bodies[b].vel = Vec3::new(-12.0, 0.0, 0.0);
    w.bodies[b].ang_vel = Vec3::ZERO;
    for _ in 0..30 {
        w.step(1.0 / 60.0); // 0.5s: long enough to reach the wall
    }
    let body = w.bodies[b];
    println!(
        "  head-on: |omega| {:.4}, vx {:.2}, y {:.2}",
        body.ang_vel.length(),
        body.vel.x,
        body.pos.y
    );
    assert!(body.ang_vel.length() < 0.3, "head-on bounce adds no spin (|omega| {})", body.ang_vel.length());
    assert!(body.vel.x > 0.0, "head-on bounce reverses outward (vx {})", body.vel.x);
    // A radial impulse has no moment arm, and nothing should have knocked the
    // body out of its straight line either.
    assert!(body.vel.y.abs() < 1e-3 && body.vel.z.abs() < 1e-3, "pure normal rebound");
    assert!((body.pos.y - (HALF_Y - 1.0)).abs() < 1e-3, "no gravity means no sag");
}

/// The pair normal impulse has to push the two bodies *apart* — body a along
/// `-n`, body b along `+n`. Getting that backwards still separates them
/// positionally (the positional correction does that part), so it needs this
/// direct velocity check to catch it.
#[test]
fn a_head_on_pair_collision_reverses_both_bodies_outward() {
    let mut w = new_world();
    w.gravity = 0.0;
    w.drag = 0.0;
    let a = w.record(0, SpawnOpts::with_scale(0.7));
    let b = w.record(0, SpawnOpts::with_scale(0.7));
    let y = HALF_Y - 0.8;
    w.bodies[a].pos = Vec3::new(-0.5, y, 0.0);
    w.bodies[b].pos = Vec3::new(0.5, y, 0.0);
    w.bodies[a].vel = Vec3::new(6.0, 0.0, 0.0);
    w.bodies[b].vel = Vec3::new(-6.0, 0.0, 0.0);
    w.bodies[a].ang_vel = Vec3::ZERO;
    w.bodies[b].ang_vel = Vec3::ZERO;

    let sum = w.bodies[a].radius + w.bodies[b].radius;
    let gap_before = (w.bodies[b].pos - w.bodies[a].pos).length();
    assert!(gap_before < sum, "set the pair up overlapping (gap {gap_before}, sum {sum})");

    // They overlap and approach from the very first substep.
    w.step(1.0 / 60.0);
    let (ba, bb) = (w.bodies[a], w.bodies[b]);
    println!(
        "  pair impulse: a.vx {:.2} -> {:.2}, b.vx {:.2} -> {:.2}",
        6.0, ba.vel.x, -6.0, bb.vel.x
    );
    assert!(ba.vel.x < 0.0, "body a is pushed along -n (vx {})", ba.vel.x);
    assert!(bb.vel.x > 0.0, "body b is pushed along +n (vx {})", bb.vel.x);
    // Equal masses, so the reversal is symmetric and neither keeps driving
    // into the other.
    assert!(
        ba.vel.x.abs() < 6.0 && bb.vel.x.abs() < 6.0,
        "the collision is inelastic, it does not speed them up (a.vx {}, b.vx {})",
        ba.vel.x,
        bb.vel.x
    );
    // A radial impulse has no moment arm: nothing here should spin them.
    assert!(
        ba.ang_vel.length() < 1e-3 && bb.ang_vel.length() < 1e-3,
        "a head-on pair hit adds no spin (|omega| {}, {})",
        ba.ang_vel.length(),
        bb.ang_vel.length()
    );

    // And they keep separating rather than staying buried.
    let mut prev_gap = (bb.pos - ba.pos).length();
    for _ in 0..30 {
        w.step(1.0 / 60.0);
        let gap = (w.bodies[b].pos - w.bodies[a].pos).length();
        assert!(gap > prev_gap - 1e-3, "the pair stops closing (gap {gap})");
        prev_gap = gap;
    }
    assert!(prev_gap >= sum - 0.02, "they end up apart (gap {prev_gap}, sum {sum})");
}

/// Pin for the pair friction torque, including a JS quirk kept on purpose.
///
/// A tangential slip at a body-to-body contact has to spin *both* bodies
/// (spin comes only from the tangential part of the impulse) and has to drag
/// body a along the slip while holding body b back against it.
///
/// The JS computes one shared `cross(n, J)` and adds it, with a positive
/// coefficient, to *both* bodies — even though body b's moment arm is
/// `-n * radius`, which would mathematically flip the sign. That is ported
/// verbatim (the JS numbers are the oracle) and asserted here so a refactor
/// cannot silently "fix" it. If the contract is changed on purpose, change this
/// test on purpose too, and say so in the changelog.
#[test]
fn pair_friction_spins_both_bodies_and_pins_the_js_torque_sign() {
    let mut w = new_world();
    w.gravity = 0.0;
    w.drag = 0.0;
    let a = w.record(0, SpawnOpts::with_scale(0.7));
    let b = w.record(0, SpawnOpts::with_scale(0.7));
    // n points from a to b along +x. The slip is created by spin, not by
    // tangential velocity: like two wheels pressed together, both spinning about
    // +z at 5 rad/s, so the contact points slide along -y at 0.7*5 + 0.7*5 = 7.
    // The bodies close along x slowly (below the pair resting speed, so the
    // contact is dead rather than bouncy) and stay at the same height, which
    // keeps n exactly along x and `vn` negative — friction only bites on an
    // approaching contact, as in the JS.
    w.bodies[a].pos = Vec3::new(-0.35, 4.0, 0.0);
    w.bodies[b].pos = Vec3::new(0.35, 4.0, 0.0);
    w.bodies[a].vel = Vec3::new(0.2, 0.0, 0.0);
    w.bodies[b].vel = Vec3::new(-0.2, 0.0, 0.0);
    w.bodies[a].ang_vel = Vec3::new(0.0, 0.0, 5.0);
    w.bodies[b].ang_vel = Vec3::new(0.0, 0.0, 5.0);

    // One substep exactly (dt = 1/120 gives ceil(1) = 1), so the contact is
    // resolved once and the analytic numbers below are directly comparable.
    w.step(1.0 / 120.0);
    let (ba, bb) = (w.bodies[a], w.bodies[b]);
    println!(
        "  pair friction: a v {} omega {} | b v {} omega {}",
        ba.vel,
        ba.ang_vel,
        bb.vel,
        bb.ang_vel
    );

    // Sanity: the contact really happened, and nothing was reported as a ring
    // (the approach is below the 2.4 body-to-body reporting threshold).
    assert!(w.impacts.is_empty(), "a soft pair jostle reports no ring");

    // Equal masses, so each body's tangential delta-v is slip * pair_friction / 2
    // = 7 * 0.12 / 2 = 0.42: a dragged along the slip (-y), b held back against
    // it (+y).
    assert!(
        (ba.vel.y + 0.42).abs() < 0.02,
        "a is dragged along the slip (v.y {}, want -0.42)",
        ba.vel.y
    );
    assert!(
        (bb.vel.y - 0.42).abs() < 0.02,
        "b is held back against the slip (v.y {}, want +0.42)",
        bb.vel.y
    );
    // Tangential friction conserves momentum: equal and opposite delta-v.
    assert!((ba.vel.y + bb.vel.y).abs() < 0.02, "no net momentum added");
    // The normal impulse removed the approach and nothing else.
    assert!(
        ba.vel.x.abs() < 0.02 && bb.vel.x.abs() < 0.02,
        "approach removed (a.vx {}, b.vx {})",
        ba.vel.x,
        bb.vel.x
    );
    assert!(ba.pos.y < 4.01 && bb.pos.y > 3.99, "no vertical nonsense");

    // Spin came from the tangential impulse only: |delta omega| = 1.25 * slip *
    // pair_friction / radius = 1.25 * 7 * 0.12 / 0.7 = 1.5 about -z, landing
    // inside the spin cap.
    let want = 5.0 - 1.5;
    assert!(
        (ba.ang_vel.z - want).abs() < 0.02,
        "a is spun down by the slip (omega.z {}, want {want})",
        ba.ang_vel.z
    );
    assert!(
        (bb.ang_vel.z - want).abs() < 0.02,
        "b takes the SAME-signed delta, exactly as the JS does (omega.z {}, want {want})",
        bb.ang_vel.z
    );
    assert!(ba.ang_vel.length() <= MAX_SPIN && bb.ang_vel.length() <= MAX_SPIN);
    // The impulse ran along y with n along x, so only the z spin axis is legal.
    for body in [ba, bb] {
        assert!(
            body.ang_vel.x.abs() < 1e-4 && body.ang_vel.y.abs() < 1e-4,
            "no spin about axes the impulse cannot reach: {}",
            body.ang_vel
        );
    }
}

/// Bodies pushed together have to separate, never sink into each other.
#[test]
fn an_overlapping_pair_separates() {
    let mut w = new_world();
    let a = w.record(0, SpawnOpts::with_scale(0.7));
    let b = w.record(0, SpawnOpts::with_scale(0.7));
    w.bodies[a].pos = Vec3::new(0.0, 0.7, 0.0);
    w.bodies[b].pos = Vec3::new(0.2, 0.7, 0.0);
    w.bodies[a].vel = Vec3::ZERO;
    w.bodies[b].vel = Vec3::ZERO;

    let sum = w.bodies[a].radius + w.bodies[b].radius;
    let mut worst_overlap = 0.0f32;
    let dt = 1.0f32 / 60.0;
    for step in 0..240 {
        w.step(dt);
        let gap = (w.bodies[b].pos - w.bodies[a].pos).length();
        worst_overlap = worst_overlap.max(sum - gap);
        if step == 0 {
            assert!(gap < sum, "the pair starts interpenetrating (gap {gap}, sum {sum})");
        }
    }
    let gap = (w.bodies[b].pos - w.bodies[a].pos).length();
    println!("  pair separation: gap {:.3}, want {sum:.3}", gap);
    assert!(gap >= sum - 0.05, "overlapping pair separates (gap {gap}, want {sum})");
    // By the end of the run they must be moving apart or resting apart, not
    // still buried in each other.
    assert!(
        worst_overlap < sum,
        "the pair never stays buried (worst overlap {worst_overlap})"
    );
}

/// `settle` is the "never leave bodies interpenetrating" control; `clamp_to_room`
/// is the position-level counterpart.
#[test]
fn settle_pulls_apart_every_overlapping_pair() {
    let mut w = new_world();
    w.set_seed(0x5177_1E00_0000_0007);
    for i in 0..12 {
        w.spawn(i % 9, SpawnOpts::default());
    }
    // Deliberately pile them into one corner before settling.
    for (i, b) in w.bodies.clone().iter().enumerate() {
        let idx = i;
        w.bodies[idx].pos = Vec3::new(
            -HALF_X + b.radius + 0.3 + (i as f32) * 0.05,
            b.radius + 0.2 + (i as f32) * 0.1,
            -HALF_Z + b.radius + 0.3,
        );
    }
    w.settle();

    let mut worst = 0.0f32;
    for i in 0..w.bodies.len() {
        for j in i + 1..w.bodies.len() {
            let a = w.bodies[i];
            let b = w.bodies[j];
            worst = worst.max(a.radius + b.radius - (b.pos - a.pos).length());
        }
    }
    assert!(worst < 0.02, "settled pile is not interpenetrating (worst overlap {worst})");

    // And settle keeps everything inside the room.
    for b in &w.bodies {
        assert!(b.pos.x >= -HALF_X + b.radius - 1e-3 && b.pos.x <= HALF_X - b.radius + 1e-3);
        assert!(b.pos.z >= -HALF_Z + b.radius - 1e-3 && b.pos.z <= HALF_Z - b.radius + 1e-3);
        assert!(b.pos.y >= b.radius - 1e-3 && b.pos.y <= HALF_Y - b.radius + 1e-3);
    }

    // clamp_to_room on its own is a hard position clamp.
    w.bodies[0].pos = Vec3::new(1e6, -1e6, 1e6);
    w.clamp_to_room();
    let b = w.bodies[0];
    assert!(b.pos.x <= HALF_X - b.radius + 1e-6 && b.pos.y >= b.radius - 1e-6 && b.pos.z <= HALF_Z - b.radius + 1e-6);
}

/// The "kick" control has to stay inside the same safety budget as everything
/// else: no NaN, no escapes, bounded speeds.
#[test]
fn kicks_do_not_break_the_room() {
    let mut w = new_world();
    w.set_seed(0x1C1C_0FFE_0000_0002);
    for i in 0..18 {
        w.record(i % 9, SpawnOpts::default());
    }
    let dt = 1.0f32 / 60.0;
    let mut max_speed = 0.0f32;
    for round in 0..8 {
        w.kick(8.0);
        for _ in 0..150 {
            w.step(dt);
        }
        let mut escapes = 0;
        for b in &w.bodies {
            assert!(
                [b.pos.x, b.pos.y, b.pos.z, b.vel.x, b.vel.y, b.vel.z, b.ang_vel.x, b.ang_vel.y, b.ang_vel.z]
                    .iter()
                    .all(|v| v.is_finite()),
                "kick round {round} produced a non-finite value"
            );
            let slack = b.scale * 1.05;
            if b.pos.x.abs() > HALF_X + slack
                || b.pos.z.abs() > HALF_Z + slack
                || b.pos.y < -slack
                || b.pos.y > HALF_Y + slack
            {
                escapes += 1;
            }
            max_speed = max_speed.max(b.vel.length());
        }
        assert_eq!(escapes, 0, "kick round {round}: bodies stayed in the room");
    }
    println!("  after 8 kicks: max speed {max_speed:.1}");
    assert!(max_speed < 200.0, "speeds stay bounded under kicking (max {max_speed})");
}

/// Turning pair collisions off must not destabilise the room (the collide flag
/// is a UI control in the JS app).
#[test]
fn pair_collisions_can_be_switched_off() {
    let mut w = new_world();
    w.set_seed(0xDECA_0000_0000_0003);
    w.collide = false;
    for i in 0..18 {
        w.record(i % 9, SpawnOpts::default());
    }
    let dt = 1.0f32 / 60.0;
    let mut overlaps = 0;
    for _ in 0..600 {
        w.step(dt);
        for i in 0..w.bodies.len() {
            for j in i + 1..w.bodies.len() {
                let a = w.bodies[i];
                let b = w.bodies[j];
                if (b.pos - a.pos).length() + 1e-6 < a.radius + b.radius {
                    overlaps += 1; // expected: bodies pass through each other
                }
            }
        }
    }
    assert!(overlaps > 0, "with collide off, bodies do interpenetrate");
    for b in &w.bodies {
        assert!(b.pos.x.abs() <= HALF_X && b.pos.y >= -1e-3 && b.pos.y <= HALF_Y + 1e-3 && b.pos.z.abs() <= HALF_Z);
    }
}
