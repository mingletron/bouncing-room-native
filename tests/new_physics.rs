//! Tests for what the simulation gained after the port: hull colliders, the grid
//! broadphase, picking, holding and snapshots. They use the public API only,
//! like `sim_invariants.rs`, and they are about behaviour a person would see —
//! a box lying flat, a crowd staying in the room — not about internal numbers.

use std::sync::Arc;

use glam::{Mat4, Vec3};

use bouncing_room_native::geom::all_primitives;
use bouncing_room_native::hull::Hull;
use bouncing_room_native::sim::{Body, SpawnOpts, World, BRUTE_FORCE_MAX};

const CUBE: usize = 1;
const TETRA: usize = 7;

fn hull_for(shape: usize) -> Arc<Hull> {
    let mesh = &all_primitives()[shape].mesh;
    let points: Vec<Vec3> = mesh.vertices.iter().map(|v| Vec3::from(v.position)).collect();
    Arc::new(Hull::from_points(&points).expect("the primitive encloses volume"))
}

fn world_with_hull(shape: usize) -> World {
    let mut world = World::default();
    world.colliders = vec![None; 9];
    world.colliders[shape] = Some(hull_for(shape));
    world
}

fn body(shape: usize, scale: f32, pos: Vec3, rot: Mat4) -> Body {
    Body {
        shape,
        scale,
        radius: scale,
        mass: scale * scale * scale * 4.0 + 0.2,
        pos,
        vel: Vec3::ZERO,
        rot,
        ang_vel: Vec3::ZERO,
    }
}

fn run(world: &mut World, seconds: f32) {
    for _ in 0..(seconds * 60.0) as usize {
        world.step(1.0 / 60.0);
    }
}

#[test]
fn a_dropped_cube_comes_to_rest_flat_on_a_face() {
    let mut world = world_with_hull(CUBE);
    // Tilted, so it has to find a face rather than start on one.
    let tilt = Mat4::from_rotation_z(0.5) * Mat4::from_rotation_x(0.3);
    world.bodies.push(body(CUBE, 0.8, Vec3::new(0.0, 3.0, 0.0), tilt));
    run(&mut world, 12.0);

    let b = &world.bodies[0];
    // The cube is normalised to a unit bounding radius, so its half edge is
    // scale / sqrt(3): that is how high its centre sits when it lies on a face,
    // and well below the scale * 0.82 a sphere collider would hold it at.
    let half_edge = 0.8 / 3.0f32.sqrt();
    assert!(
        (b.pos.y - half_edge).abs() < 0.03,
        "centre at {}, a face-down cube sits at {half_edge}",
        b.pos.y
    );
    assert!(b.vel.length() < 0.4, "still moving: {:?}", b.vel);
    assert!(b.ang_vel.length() < 0.6, "still turning: {:?}", b.ang_vel);
    // One of its axes points straight up.
    let up = [b.rot.x_axis, b.rot.y_axis, b.rot.z_axis]
        .iter()
        .map(|axis| axis.truncate().dot(Vec3::Y).abs())
        .fold(0.0f32, f32::max);
    assert!(up > 0.995, "not lying on a face: best axis alignment {up}");
}

#[test]
fn a_cube_without_a_hull_still_rests_as_a_sphere() {
    // The inherited model, untouched: the same drop with no collider registered
    // leaves the centre at the tuned sphere radius, not at the half edge.
    let mut world = World::default();
    let mut b = body(CUBE, 0.8, Vec3::new(0.0, 3.0, 0.0), Mat4::IDENTITY);
    b.radius = 0.8 * 0.82;
    world.bodies.push(b);
    run(&mut world, 12.0);
    assert!((world.bodies[0].pos.y - 0.8 * 0.82).abs() < 0.02, "{}", world.bodies[0].pos.y);
}

#[test]
fn the_hull_switch_turns_every_body_back_into_a_sphere() {
    let mut world = world_with_hull(CUBE);
    world.hulls = false;
    world.bodies.push(body(CUBE, 0.8, Vec3::new(0.0, 3.0, 0.0), Mat4::IDENTITY));
    run(&mut world, 12.0);
    // radius == scale here, so as a sphere it rests a full scale off the floor.
    assert!((world.bodies[0].pos.y - 0.8).abs() < 0.02, "{}", world.bodies[0].pos.y);
}

#[test]
fn hulls_and_spheres_thrown_together_stay_in_the_room_and_calm_down() {
    let mut world = World::default();
    world.set_seed(0xB0B);
    world.colliders = vec![None; 9];
    world.colliders[CUBE] = Some(hull_for(CUBE));
    world.colliders[TETRA] = Some(hull_for(TETRA));
    for i in 0..30 {
        world.record([0, CUBE, TETRA][i % 3], SpawnOpts::default());
    }
    world.kick(9.0);
    let mut fastest = 0.0f32;
    for _ in 0..(20.0 * 60.0) as usize {
        world.step(1.0 / 60.0);
        for b in &world.bodies {
            assert!(b.pos.is_finite() && b.vel.is_finite() && b.ang_vel.is_finite());
            fastest = fastest.max(b.vel.length());
        }
    }
    // Nothing gained energy it should not have: a kick of 9 plus a 5 m fall is
    // well under this, and a solver that pumps energy in blows straight past it.
    assert!(fastest < 40.0, "something reached {fastest} units/s");
    let (hx, hy, hz) = (world.half_x, world.half_y, world.half_z);
    for b in &world.bodies {
        assert!(b.pos.x.abs() <= hx + 0.05 && b.pos.z.abs() <= hz + 0.05, "{:?}", b.pos);
        assert!(b.pos.y >= -0.05 && b.pos.y <= hy + 0.05, "{:?}", b.pos);
    }
    let moving = world.bodies.iter().filter(|b| b.vel.length() > 1.5).count();
    assert!(moving <= 3, "{moving} of 30 bodies are still flying after 20 s");
}

#[test]
fn a_crowd_past_the_brute_force_limit_does_not_tunnel_or_pile_into_itself() {
    let count = BRUTE_FORCE_MAX * 8;
    let mut world = World::default();
    world.set_seed(42);
    for i in 0..count {
        world.spawn(i % 9, SpawnOpts::with_scale(0.2));
    }
    world.settle();
    run(&mut world, 6.0);

    let (hx, hy, hz) = (world.half_x, world.half_y, world.half_z);
    for b in &world.bodies {
        assert!(b.pos.x.abs() <= hx + 0.02 && b.pos.z.abs() <= hz + 0.02, "{:?}", b.pos);
        assert!(b.pos.y >= -0.02 && b.pos.y <= hy + 0.02, "{:?}", b.pos);
    }
    // If the grid missed pairs, bodies would sink into each other. Checked the
    // slow way, against every pair, which is exactly what the grid replaces.
    let mut deep = 0;
    for i in 0..count {
        for j in i + 1..count {
            let (a, b) = (&world.bodies[i], &world.bodies[j]);
            let gap = a.pos.distance(b.pos) - (a.radius + b.radius);
            if gap < -0.25 * a.radius.min(b.radius) {
                deep += 1;
            }
        }
    }
    assert!(deep <= count / 100, "{deep} pairs are deeply interpenetrating");
}

#[test]
fn a_ray_picks_the_nearest_body_it_passes_through() {
    let mut world = World::default();
    world.bodies.push(body(0, 0.5, Vec3::new(0.0, 1.0, -4.0), Mat4::IDENTITY));
    world.bodies.push(body(0, 0.5, Vec3::new(0.0, 1.0, -8.0), Mat4::IDENTITY));
    let origin = Vec3::new(0.0, 1.0, 0.0);
    let (index, t) = world.pick(origin, Vec3::NEG_Z).expect("the ray runs through both");
    assert_eq!(index, 0, "the nearer one");
    assert!((t - (4.0 - 0.5 * 0.95)).abs() < 1e-4, "hit at {t}");
    assert!(world.pick(origin, Vec3::Z).is_none(), "nothing behind the ray's origin");
    assert!(world.pick(origin, Vec3::X).is_none());
}

#[test]
fn a_held_body_follows_its_target_and_keeps_its_speed_when_let_go() {
    let mut world = World::default();
    world.bodies.push(body(0, 0.5, Vec3::new(0.0, 1.0, 0.0), Mat4::IDENTITY));
    // Sweep the target sideways, holding every frame, as the pointer would.
    let mut target = Vec3::new(0.0, 2.5, 0.0);
    for _ in 0..60 {
        target.x += 0.05;
        world.hold(0, target);
        world.step(1.0 / 60.0);
    }
    let b = world.bodies[0];
    assert!(b.pos.distance(target) < 0.35, "{:?} is not near {target:?}", b.pos);
    assert!(b.vel.x > 2.0, "it is moving with the sweep: {:?}", b.vel);
    // Let go: no more `hold`. It carries on in the direction it was thrown.
    let x = b.pos.x;
    world.step(1.0 / 60.0);
    assert!(world.bodies[0].pos.x > x);

    // A target outside the room is pulled back inside it, not chased through
    // the wall.
    world.hold(0, Vec3::new(100.0, 2.0, 0.0));
    run(&mut world, 0.0);
    for _ in 0..120 {
        world.hold(0, Vec3::new(100.0, 2.0, 0.0));
        world.step(1.0 / 60.0);
    }
    assert!(world.bodies[0].pos.x <= world.half_x - 0.5 + 0.02);
    // Holding a body that is not there is a no-op, not a panic.
    world.hold(99, Vec3::ZERO);
}

#[test]
fn a_restored_snapshot_replays_identically() {
    let mut world = World::default();
    world.set_seed(99);
    for i in 0..12 {
        world.record(i % 9, SpawnOpts::default());
    }
    run(&mut world, 1.0);
    let saved = world.snapshot();
    assert_eq!(saved.body_count(), 12);

    world.kick(8.0); // draws from the generator
    run(&mut world, 1.0);
    let first: Vec<Vec3> = world.bodies.iter().map(|b| b.pos).collect();

    world.restore(&saved);
    world.kick(8.0);
    run(&mut world, 1.0);
    let second: Vec<Vec3> = world.bodies.iter().map(|b| b.pos).collect();
    assert_eq!(first, second, "the generator's state came back with the bodies");
}
