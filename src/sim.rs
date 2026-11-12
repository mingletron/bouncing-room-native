//! Port of `bouncing-room/src/physics.js`.
//!
//! Bouncing-body simulation. Every shape is approximated by a bounding sphere,
//! which keeps the room/corner maths honest while still reading as a pile of
//! tumbling solids. Each shape carries a tuned radius factor so contacts land
//! where the eye expects rather than at the extremes of the bounding sphere.
//!
//! The two behaviours this port must not get wrong (and which the integration
//! tests exist to catch):
//! - friction acts on the **slip at the contact point**
//!   (`vel + ang_vel.cross(r_contact)`, with `r_contact = -n * radius` at a wall
//!   and `+/-n * radius` for a pair), not on tangential velocity — that is what
//!   makes bodies roll instead of skid;
//! - the pair normal impulse pushes the two bodies **apart** (body a along `-n`,
//!   body b along `+n`).
//!
//! Spin only ever comes from the tangential part of an impulse: a radial impulse
//! runs along the centre line, so it has no moment arm.

use std::sync::Arc;

use glam::{Mat3, Mat4, Vec3};

use crate::geom::PRIMITIVE_NAMES;
use crate::hull::{Contact, Hull, Placed};

/// Ceiling on angular speed (rad/s); contacts clamp toward it after any change.
pub const MAX_SPIN: f32 = 9.0;

/// Below this approach speed a *wall* contact stops bouncing (absorbs the last
/// of the bounce) so bodies do not chatter forever on the floor.
pub const WALL_REST_SPEED: f32 = 0.55;
/// Same for a body-to-body contact.
pub const PAIR_REST_SPEED: f32 = 0.5;
/// Approach speed above which a wall contact reports an impact ring.
pub const WALL_IMPACT_SPEED: f32 = 1.1;
/// Body rings need a harder hit than wall ones: a settled pile jostles
/// constantly, and their contacts are usually edge-on to the camera.
pub const PAIR_IMPACT_SPEED: f32 = 2.4;

/// Default tuning, straight from the JS `World` constructor.
pub const GRAVITY: f32 = -17.0;
pub const RESTITUTION: f32 = 0.84;
pub const DRAG: f32 = 0.06;
pub const WALL_FRICTION: f32 = 0.16;
pub const PAIR_FRICTION: f32 = 0.12;
/// How hard a contact pulls a body toward rolling (0 = frictionless skid,
/// 1 = instant grip).
pub const SPIN_COUPLING: f32 = 0.45;

/// Strength of the "kick" control.
pub const DEFAULT_KICK: f32 = 8.0;

/// Room constants the demo runs with (x/z in [-half, half], y in [0, half]).
pub const ROOM_HALF_X: f32 = 7.2;
pub const ROOM_HALF_Y: f32 = 5.4;
pub const ROOM_HALF_Z: f32 = 7.2;

/// Substepping: at most this many ticks per `step`, each no longer than
/// [`SUBSTEP_TARGET`] seconds.
pub const MAX_SUBSTEPS: usize = 4;
pub const SUBSTEP_TARGET: f64 = 1.0 / 120.0;

/// Up to this many bodies, every pair is tested against every other, in the
/// order the JS oracle tests them — so a room the size the browser build could
/// show behaves exactly as it always has. Past it the pair loop goes through a
/// spatial grid instead, which is what makes thousands of bodies affordable.
pub const BRUTE_FORCE_MAX: usize = 64;

/// How hard a held body is pulled to the pointer: its velocity is this many
/// times its distance from where the pointer wants it, per second.
pub const HOLD_STIFFNESS: f32 = 14.0;
/// Fastest a held body is allowed to chase the pointer, so a flick across the
/// window is a throw rather than a tunnel through the wall.
pub const HOLD_MAX_SPEED: f32 = 34.0;

/// Friction for hull contacts is a Coulomb limit rather than the fraction of
/// slip the sphere contacts remove; this turns one dial into the other.
const HULL_FRICTION_SCALE: f32 = 3.5;
/// Spin a hull keeps per contact. A sphere can roll forever; a faceted body
/// rocking on an edge has to lose that energy somewhere or it never lies down.
const HULL_CONTACT_SPIN_KEEP: f32 = 0.985;

/// Impact rings older than this many entries get dropped.
pub const MAX_IMPACTS: usize = 48;

/// Effective collision radius as a fraction of the normalised bounding radius.
pub const RADIUS_FACTOR_TABLE: [(&str, f32); 9] = [
    ("sphere", 1.0),
    ("cube", 0.82),
    ("capsule", 0.9),
    ("torus", 0.86),
    ("cone", 0.82),
    ("cylinder", 0.85),
    ("icosahedron", 0.95),
    ("tetrahedron", 0.74),
    ("octahedron", 0.86),
];

/// Fallback factor for an unknown shape name (the JS `?? 0.85`).
pub const DEFAULT_RADIUS_FACTOR: f32 = 0.85;

pub fn radius_factor(name: &str) -> f32 {
    RADIUS_FACTOR_TABLE
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, f)| *f)
        .unwrap_or(DEFAULT_RADIUS_FACTOR)
}

/// Factor for a primitive referenced by index into [`PRIMITIVE_NAMES`].
pub fn radius_factor_for_shape(shape: usize) -> f32 {
    radius_factor(PRIMITIVE_NAMES.get(shape).copied().unwrap_or(""))
}

/// One simulated solid. `shape` indexes the primitive list (see
/// [`crate::geom::PRIMITIVE_NAMES`]); the mesh is drawn with `rot * scale`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Body {
    pub shape: usize,
    pub scale: f32,
    /// Collision radius: `scale * radius_factor(shape)`.
    pub radius: f32,
    pub mass: f32,
    pub pos: Vec3,
    pub vel: Vec3,
    /// Orientation as an orthonormal rotation matrix (translation column zero).
    pub rot: Mat4,
    /// Angular velocity in world axes, rad/s.
    pub ang_vel: Vec3,
}

/// A contact worth drawing a ring for.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Impact {
    pub pos: Vec3,
    pub n: Vec3,
    pub color: [f32; 3],
    pub base_radius: f32,
    /// Approach speed along the contact normal, in units/s, *before* the bounce.
    /// A ring does not need it — a ring only exists at all because this speed
    /// cleared a threshold — but "how hard" is exactly what a sound is built from,
    /// and by the time the frame is drawn the bodies have already changed velocity.
    /// So it is recorded here, at the contact, and nothing downstream recomputes it.
    pub speed: f32,
    /// The shape this contact is credited to, which sets a sound's timbre. The wall
    /// contacts use the body that hit; the pair contacts use body `a`, the same body
    /// whose palette colour the ring is drawn in, so the ring and the click agree.
    pub shape: usize,
    pub age: f32,
    pub life: f32,
}

/// Per-spawn overrides; everything else is randomised like the JS `spawn`.
#[derive(Copy, Clone, Debug, Default)]
pub struct SpawnOpts {
    pub scale: Option<f32>,
}

impl SpawnOpts {
    pub fn with_scale(scale: f32) -> Self {
        SpawnOpts { scale: Some(scale) }
    }
}

/// Small deterministic PRNG (splitmix64) so the port is reproducible without
/// pulling in a dependency. Stand-in for `Math.random`.
#[derive(Copy, Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0)
    }

    /// JS `rand(lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

/// Everything a run needs to be put back exactly where it was: the bodies, the
/// rings in flight, and the generator's position in its stream. The tuning is
/// deliberately not here — rewinding the room should not undo a slider.
#[derive(Clone, Debug)]
pub struct Snapshot {
    bodies: Vec<Body>,
    impacts: Vec<Impact>,
    rng: Rng,
}

impl Snapshot {
    /// Roughly what holding this costs, for a history that budgets by memory.
    pub fn bytes(&self) -> usize {
        self.bodies.len() * std::mem::size_of::<Body>()
            + self.impacts.len() * std::mem::size_of::<Impact>()
            + std::mem::size_of::<Self>()
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }
}

/// The contact parameters, copied out so `bodies` and `impacts` can be borrowed
/// at the same time.
#[derive(Copy, Clone, Debug)]
struct Tuning {
    restitution: f32,
    wall_friction: f32,
    pair_friction: f32,
    spin_coupling: f32,
}

#[derive(Clone, Debug)]
pub struct World {
    pub half_x: f32,
    pub half_y: f32,
    pub half_z: f32,
    pub bodies: Vec<Body>,
    pub impacts: Vec<Impact>,

    pub gravity: f32,
    pub restitution: f32,
    pub drag: f32,
    pub collide: bool,
    pub wall_friction: f32,
    pub pair_friction: f32,
    /// How hard a contact pulls a body toward rolling (0 = frictionless skid,
    /// 1 = instant grip).
    pub spin_coupling: f32,
    /// Impact ring colours, indexed by `Body::shape`. The JS body carries its
    /// `primitive`; the contract's `Body` carries only a shape index, so the
    /// palette lives on the world.
    pub colors: Vec<[f32; 3]>,
    /// Convex-hull colliders, indexed by `Body::shape` like `colors`. A shape
    /// with no entry (or `None`) collides as its bounding sphere, which is every
    /// shape unless the caller registers a hull for it.
    pub colliders: Vec<Option<Arc<Hull>>>,
    /// Master switch for `colliders`: off, every body is a sphere again.
    pub hulls: bool,

    rng: Rng,
}

impl Default for World {
    fn default() -> Self {
        World::new(ROOM_HALF_X, ROOM_HALF_Y, ROOM_HALF_Z)
    }
}

impl World {
    pub fn new(half_x: f32, half_y: f32, half_z: f32) -> Self {
        World {
            half_x,
            half_y,
            half_z,
            bodies: Vec::new(),
            impacts: Vec::new(),
            gravity: GRAVITY,
            restitution: RESTITUTION,
            drag: DRAG,
            collide: true,
            wall_friction: WALL_FRICTION,
            pair_friction: PAIR_FRICTION,
            spin_coupling: SPIN_COUPLING,
            colors: crate::geom::PRIMITIVE_COLORS.to_vec(),
            colliders: Vec::new(),
            hulls: true,
            rng: Rng(0x5EED_1234_ABCD_EF00),
        }
    }

    /// Make a run reproducible (tests, replay).
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Rng(seed);
    }

    pub fn body(&self, index: usize) -> Option<&Body> {
        self.bodies.get(index)
    }

    pub fn body_mut(&mut self, index: usize) -> Option<&mut Body> {
        self.bodies.get_mut(index)
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            bodies: self.bodies.clone(),
            impacts: self.impacts.clone(),
            rng: self.rng,
        }
    }

    pub fn restore(&mut self, snapshot: &Snapshot) {
        self.bodies.clone_from(&snapshot.bodies);
        self.impacts.clone_from(&snapshot.impacts);
        self.rng = snapshot.rng;
    }

    /// The first body a ray hits, and how far along the ray. `dir` must be unit
    /// length. Bodies are tested as spheres the size they are *drawn*, not the
    /// size they collide at, because what is being picked is the picture.
    pub fn pick(&self, origin: Vec3, dir: Vec3) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
        for (i, b) in self.bodies.iter().enumerate() {
            let radius = b.scale.max(b.radius) * 0.95;
            let to = b.pos - origin;
            let along = to.dot(dir);
            let off2 = to.length_squared() - along * along;
            if off2 > radius * radius {
                continue;
            }
            let t = along - (radius * radius - off2).sqrt();
            if t > 0.0 && best.is_none_or(|(_, bt)| t < bt) {
                best = Some((i, t));
            }
        }
        best
    }

    /// Pull a held body toward `target`: its velocity becomes whatever closes
    /// the gap, capped. Nothing is done on release — the body simply keeps the
    /// velocity it was last given, which is what makes letting go a throw.
    pub fn hold(&mut self, index: usize, target: Vec3) {
        let (hx, hy, hz) = (self.half_x, self.half_y, self.half_z);
        let Some(b) = self.bodies.get_mut(index) else {
            return;
        };
        let r = b.radius;
        let target = Vec3::new(
            clamp_f(target.x, -hx + r, hx - r),
            clamp_f(target.y, r, hy - r),
            clamp_f(target.z, -hz + r, hz - r),
        );
        let mut vel = (target - b.pos) * HOLD_STIFFNESS;
        let speed = vel.length();
        if speed > HOLD_MAX_SPEED {
            vel *= HOLD_MAX_SPEED / speed;
        }
        b.vel = vel;
        // A held thing should not also be spinning itself out of the hand.
        b.ang_vel *= 0.9;
    }

    /// Random-ish spawn; returns the new body's index.
    pub fn spawn(&mut self, primitive: usize, opts: SpawnOpts) -> usize {
        let (half_x, half_y, half_z) = (self.half_x, self.half_y, self.half_z);
        let scale = opts.scale.unwrap_or_else(|| self.rng.range(0.42, 0.78));
        let factor = radius_factor_for_shape(primitive);
        let r = scale * factor;

        let x = self.rng.range(-half_x + r + 0.2, half_x - r - 0.2);
        let z = self.rng.range(-half_z + r + 0.2, half_z - r - 0.2);
        let y = self.rng.range(r + 0.15, half_y - r - 0.15);

        // Random orientation: a random axis through a random angle.
        let axis = Vec3::new(
            self.rng.range(-1.0, 1.0),
            self.rng.range(-1.0, 1.0),
            self.rng.range(-1.0, 1.0),
        );
        let angle = self.rng.range(0.0, std::f32::consts::PI * 2.0);
        let rot = axis_angle(axis, angle);

        let ang_vel = Vec3::new(
            self.rng.range(-4.0, 4.0),
            self.rng.range(-4.0, 4.0),
            self.rng.range(-4.0, 4.0),
        );

        self.bodies.push(Body {
            shape: primitive,
            scale,
            radius: r,
            mass: scale * scale * scale * 4.0 + 0.2,
            pos: Vec3::new(x, y, z),
            vel: Vec3::new(
                self.rng.range(-5.0, 5.0),
                self.rng.range(-2.0, 5.0),
                self.rng.range(-5.0, 5.0),
            ),
            rot,
            ang_vel,
        });
        self.bodies.len() - 1
    }

    /// Nudge every body with a random impulse — the "kick" control.
    pub fn kick(&mut self, strength: f32) {
        for b in self.bodies.iter_mut() {
            b.vel.x += self.rng.range(-strength, strength);
            b.vel.y += self.rng.range(strength * 0.35, strength * 1.5);
            b.vel.z += self.rng.range(-strength, strength);
            b.ang_vel.x += self.rng.range(-3.0, 3.0);
            b.ang_vel.y += self.rng.range(-3.0, 3.0);
            b.ang_vel.z += self.rng.range(-3.0, 3.0);
        }
    }

    /// Separate any overlapping pair so a respawn or a drag never leaves bodies
    /// interpenetrating.
    pub fn settle(&mut self) {
        if self.bodies.len() > BRUTE_FORCE_MAX {
            self.settle_with_grid();
            return;
        }
        for _pass in 0..24 {
            let mut worst = 0.0f32;
            let n = self.bodies.len();
            for i in 0..n {
                let (left, right) = self.bodies.split_at_mut(i + 1);
                for b in right.iter_mut() {
                    // Fresh copy per pair: the JS holds a live reference, so a
                    // body pushed by an earlier pair this pass must be seen at
                    // its new position by the next one.
                    let a = left[i];
                    let d = b.pos - a.pos;
                    let sum = a.radius + b.radius;
                    let dist = d.length();
                    if dist >= sum {
                        continue;
                    }
                    let overlap = sum - dist;
                    worst = worst.max(overlap);
                    let nvec = if dist > 1e-6 {
                        d / dist
                    } else {
                        Vec3::new(
                            self.rng.range(-1.0, 1.0),
                            self.rng.range(-1.0, 1.0),
                            self.rng.range(-1.0, 1.0),
                        )
                    };
                    let l = nvec.length();
                    let l = if l == 0.0 { 1.0 } else { l };
                    let push = overlap * 0.5 + 0.01;
                    left[i].pos -= (nvec / l) * push;
                    b.pos += (nvec / l) * push;
                }
            }
            self.clamp_to_room();
            if worst < 1e-3 {
                break;
            }
        }
    }

    /// [`World::settle`] for a crowd: the same push-apart rule, with the pairs
    /// found through the grid instead of by testing every body against every
    /// other, 24 times over.
    fn settle_with_grid(&mut self) {
        for _pass in 0..24 {
            let mut worst = 0.0f32;
            let pairs = grid_pairs(&self.bodies, &|b: &Body| b.radius, self.half_x, self.half_y, self.half_z);
            for (i, j) in pairs {
                let (left, right) = self.bodies.split_at_mut(j as usize);
                let (a, b) = (&mut left[i as usize], &mut right[0]);
                let d = b.pos - a.pos;
                let sum = a.radius + b.radius;
                let dist = d.length();
                if dist >= sum {
                    continue;
                }
                let overlap = sum - dist;
                worst = worst.max(overlap);
                let nvec = if dist > 1e-6 {
                    d / dist
                } else {
                    Vec3::new(
                        self.rng.range(-1.0, 1.0),
                        self.rng.range(-1.0, 1.0),
                        self.rng.range(-1.0, 1.0),
                    )
                    .normalize_or(Vec3::Y)
                };
                let push = overlap * 0.5 + 0.01;
                a.pos -= nvec * push;
                b.pos += nvec * push;
            }
            self.clamp_to_room();
            if worst < 1e-3 {
                break;
            }
        }
    }

    pub fn clamp_to_room(&mut self) {
        let (hx, hy, hz) = (self.half_x, self.half_y, self.half_z);
        for b in self.bodies.iter_mut() {
            let r = b.radius;
            b.pos.x = clamp_f(b.pos.x, -hx + r, hx - r);
            b.pos.z = clamp_f(b.pos.z, -hz + r, hz - r);
            b.pos.y = clamp_f(b.pos.y, r, hy - r);
        }
    }

    /// Spawn, then separate the pile.
    pub fn record(&mut self, primitive: usize, opts: SpawnOpts) -> usize {
        let i = self.spawn(primitive, opts);
        self.settle();
        i
    }

    /// One physics tick. Substeps keep a fast body from tunnelling through a
    /// wall between frames.
    pub fn step(&mut self, dt: f32) {
        let substeps = substep_count(dt as f64);
        let h = dt / substeps as f32;
        for _ in 0..substeps {
            self.integrate(h);
        }
        self.age_impacts(dt);
    }

    fn integrate(&mut self, h: f32) {
        let g = self.gravity;
        let damp = (-self.drag * h).exp();
        let t = Tuning {
            restitution: self.restitution,
            wall_friction: self.wall_friction,
            pair_friction: self.pair_friction,
            spin_coupling: self.spin_coupling,
        };
        let (hx, hy, hz) = (self.half_x, self.half_y, self.half_z);
        let collide = self.collide;

        let hulls = self.hulls;

        // Destructure so `bodies`, `impacts`, `colors` and `rng` can all be
        // borrowed independently inside the loops below.
        let World {
            bodies,
            impacts,
            colors,
            colliders,
            rng,
            ..
        } = self;
        let colliders = colliders.as_slice();

        for b in bodies.iter_mut() {
            b.vel.y += g * h;
            b.vel *= damp;
            b.ang_vel *= damp;

            b.pos += b.vel * h;

            integrate_rotation(&mut b.rot, b.ang_vel * h);
        }

        let palette = colors.as_slice();
        for b in bodies.iter_mut() {
            match hull_in(hulls, colliders, b.shape) {
                Some(hull) => collide_hull_with_walls(b, hull, t, hx, hy, hz, palette, impacts),
                None => collide_with_walls(b, t, hx, hy, hz, palette, impacts),
            }
        }

        if collide {
            let n = bodies.len();
            if n <= BRUTE_FORCE_MAX {
                for i in 0..n {
                    let (left, right) = bodies.split_at_mut(i + 1);
                    // Copy `a` out so the rest of the slice stays free to mutate.
                    // Nothing else touches bodies[i] while this inner loop runs, so
                    // carrying the running updates in the local matches the JS live
                    // reference (unlike `settle`, where each pass re-reads it).
                    let mut a = left[i];
                    for b in right.iter_mut() {
                        resolve_pair(&mut a, b, t, palette, hulls, colliders, rng, impacts);
                    }
                    left[i] = a;
                }
            } else {
                // The same contacts, in the same order (ascending i, then j), but
                // only for pairs the grid says are near each other.
                let bound = |b: &Body| bound_radius(b, hulls, colliders);
                for (i, j) in grid_pairs(bodies, &bound, hx, hy, hz) {
                    let (left, right) = bodies.split_at_mut(j as usize);
                    resolve_pair(
                        &mut left[i as usize],
                        &mut right[0],
                        t,
                        palette,
                        hulls,
                        colliders,
                        rng,
                        impacts,
                    );
                }
            }
        }
    }

    fn age_impacts(&mut self, dt: f32) {
        let mut i = self.impacts.len();
        while i > 0 {
            i -= 1;
            self.impacts[i].age += dt;
            if self.impacts[i].age >= self.impacts[i].life {
                // `remove`, not `swap_remove`: the JS splices, which keeps the
                // ring order the renderer draws in.
                self.impacts.remove(i);
            }
        }
        if self.impacts.len() > MAX_IMPACTS {
            let drop = self.impacts.len() - MAX_IMPACTS;
            self.impacts.drain(0..drop);
        }
    }
}

fn hull_in(enabled: bool, colliders: &[Option<Arc<Hull>>], shape: usize) -> Option<&Hull> {
    if !enabled {
        return None;
    }
    colliders.get(shape).and_then(|h| h.as_deref())
}

/// The radius of the sphere that is guaranteed to contain a body's collider:
/// its tuned radius when it collides as a sphere, and its full scale when it
/// collides as a hull, whose corners reach out to the mesh's bounding sphere.
fn bound_radius(b: &Body, hulls: bool, colliders: &[Option<Arc<Hull>>]) -> f32 {
    match hull_in(hulls, colliders, b.shape) {
        Some(_) => b.scale.max(b.radius),
        None => b.radius,
    }
}

fn placed<'a>(hull: &'a Hull, b: &Body) -> Placed<'a> {
    Placed {
        hull,
        pos: b.pos,
        rot: Mat3::from_mat4(b.rot),
        scale: b.scale,
    }
}

/// One pair of bodies for one substep: nothing if their bounding spheres are
/// apart, a nudge if they are exactly concentric, and otherwise the contact —
/// the inherited sphere rule when both are spheres, the hull rule when either
/// has one.
#[allow(clippy::too_many_arguments)]
fn resolve_pair(
    a: &mut Body,
    b: &mut Body,
    t: Tuning,
    palette: &[[f32; 3]],
    hulls: bool,
    colliders: &[Option<Arc<Hull>>],
    rng: &mut Rng,
    impacts: &mut Vec<Impact>,
) {
    let hull_a = hull_in(hulls, colliders, a.shape);
    let hull_b = hull_in(hulls, colliders, b.shape);
    let either = hull_a.is_some() || hull_b.is_some();

    let d = b.pos - a.pos;
    let sum = if either {
        bound_radius(a, hulls, colliders) + bound_radius(b, hulls, colliders)
    } else {
        a.radius + b.radius
    };
    if d.length_squared() >= sum * sum {
        return;
    }
    if d.length() < 1e-6 {
        // Perfectly concentric pair: pick any direction to split them.
        b.pos += Vec3::new(rng.range(-0.02, 0.02), 0.03, rng.range(-0.02, 0.02));
        return;
    }
    let impact = if either {
        let contact = match (hull_a, hull_b) {
            (Some(ha), Some(hb)) => placed(ha, a).against_hull(&placed(hb, b)),
            (Some(ha), None) => placed(ha, a).against_sphere(b.pos, b.radius),
            // The hull is `b`, so the contact comes back pointing b -> a.
            (None, Some(hb)) => placed(hb, b).against_sphere(a.pos, a.radius).map(|c| Contact {
                n: -c.n,
                depth: c.depth,
                point: a.pos - c.n * a.radius,
            }),
            (None, None) => None,
        };
        contact.and_then(|c| collide_rigid_pair(a, b, c, t, palette))
    } else {
        collide_pair(a, b, t, palette)
    };
    if let Some(impact) = impact {
        impacts.push(impact);
    }
}

/// Every pair of bodies whose bounding spheres are close enough to matter, as
/// `(i, j)` with `i < j`, in ascending order.
///
/// A uniform grid with cells at least as wide as the largest body's diameter:
/// two bodies can only touch if they are in the same cell or neighbouring ones,
/// so each body is tested against 27 cells' worth of others instead of all of
/// them. The small margin covers the position corrections earlier contacts in
/// the same substep make after this list was built.
fn grid_pairs(
    bodies: &[Body],
    bound: &dyn Fn(&Body) -> f32,
    hx: f32,
    hy: f32,
    hz: f32,
) -> Vec<(u32, u32)> {
    const MARGIN: f32 = 0.05;
    let radii: Vec<f32> = bodies.iter().map(bound).collect();
    let largest = radii.iter().copied().fold(0.05f32, f32::max);
    let cell = largest * 2.0 + MARGIN;
    let dims = [
        ((2.0 * hx / cell).ceil() as usize).clamp(1, 256),
        ((hy / cell).ceil() as usize).clamp(1, 256),
        ((2.0 * hz / cell).ceil() as usize).clamp(1, 256),
    ];
    let coord = |v: f32, lo: f32, n: usize| -> usize {
        (((v - lo) / cell).floor().max(0.0) as usize).min(n - 1)
    };
    let cell_of = |b: &Body| -> [usize; 3] {
        [
            coord(b.pos.x, -hx, dims[0]),
            coord(b.pos.y, 0.0, dims[1]),
            coord(b.pos.z, -hz, dims[2]),
        ]
    };
    let flat = |c: [usize; 3]| (c[1] * dims[2] + c[2]) * dims[0] + c[0];

    // Counting sort of body indices into cells: `starts[c]..starts[c + 1]`
    // indexes `sorted` for cell `c`.
    let cells: Vec<usize> = bodies.iter().map(|b| flat(cell_of(b))).collect();
    let mut starts = vec![0u32; dims[0] * dims[1] * dims[2] + 1];
    for &c in &cells {
        starts[c + 1] += 1;
    }
    for c in 1..starts.len() {
        starts[c] += starts[c - 1];
    }
    let mut cursor = starts.clone();
    let mut sorted = vec![0u32; bodies.len()];
    for (i, &c) in cells.iter().enumerate() {
        sorted[cursor[c] as usize] = i as u32;
        cursor[c] += 1;
    }

    let mut pairs = Vec::new();
    for (i, a) in bodies.iter().enumerate() {
        let from = pairs.len();
        let c = cell_of(a);
        for y in c[1].saturating_sub(1)..=(c[1] + 1).min(dims[1] - 1) {
            for z in c[2].saturating_sub(1)..=(c[2] + 1).min(dims[2] - 1) {
                for x in c[0].saturating_sub(1)..=(c[0] + 1).min(dims[0] - 1) {
                    let cell_index = flat([x, y, z]);
                    for &j in &sorted[starts[cell_index] as usize..starts[cell_index + 1] as usize] {
                        if j as usize <= i {
                            continue;
                        }
                        let reach = radii[i] + radii[j as usize] + MARGIN;
                        if a.pos.distance_squared(bodies[j as usize].pos) < reach * reach {
                            pairs.push((i as u32, j));
                        }
                    }
                }
            }
        }
        pairs[from..].sort_unstable();
    }
    pairs
}

/// JS `clamp`. Note the JS order: below `lo` wins even if `lo > hi`.
fn clamp_f(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// `dt / (1/120)`, rounded to 1e-6 before the ceiling.
///
/// The JS passes f64 dts, where `1/60` divides into `1/120` exactly. Here `dt`
/// arrives as an `f32`, so `1/60f32` is a hair *more* than two half-frames and a
/// bare `ceil` would silently add a third substep — which would apply the wall
/// contacts a third time per frame and change the whole simulation. Snapping
/// first keeps the JS substep count; genuinely fractional cases (1/50 -> 3) are
/// unaffected.
fn substep_count(dt: f64) -> usize {
    let raw = dt / SUBSTEP_TARGET;
    let snapped = (raw * 1e6).round() / 1e6;
    let steps = snapped.ceil().max(1.0) as usize;
    steps.min(MAX_SUBSTEPS)
}

/// Rotation matrix for a rotation of `angle` radians about `axis`.
///
/// Mirrors the JS `axisAngle`: the axis is normalised, and a zero axis leaves
/// length 1 (so the vector is used as-is, i.e. all zeros).
fn axis_angle(axis: Vec3, angle: f32) -> Mat4 {
    let len = axis.length();
    let len = if len == 0.0 { 1.0 } else { len };
    let a = axis / len;
    Mat4::from_axis_angle(a, angle)
}

/// Re-orthonormalise the 3x3 part of a rotation matrix. Error creeps in every
/// time we integrate angular velocity, so the Gram-Schmidt pass keeps the shape
/// from shearing over a long run. The translation column is left alone, as in
/// the JS (which only touches m[0..3], m[4..7], m[8..11]).
pub fn orthonormalise(m: &mut Mat4) {
    let c = m.to_cols_array_2d();

    let mut c0 = Vec3::new(c[0][0], c[0][1], c[0][2]);
    let mut l = c0.length();
    if l == 0.0 {
        l = 1.0;
    }
    c0 /= l;

    let c1v = Vec3::new(c[1][0], c[1][1], c[1][2]);
    let d = c1v.dot(c0);
    let mut c1 = c1v - c0 * d;
    let mut l = c1.length();
    if l == 0.0 {
        l = 1.0;
    }
    c1 /= l;

    // column 2 = column0 x column1
    let c2 = c0.cross(c1);

    *m = Mat4::from_cols_array_2d(&[
        [c0.x, c0.y, c0.z, 0.0],
        [c1.x, c1.y, c1.z, 0.0],
        [c2.x, c2.y, c2.z, 0.0],
        c[3],
    ]);
}

/// Fold an angular-velocity step (`w`, radians, already scaled by dt) into an
/// existing rotation matrix using the small-angle exponential map, then keep the
/// result orthonormal.
pub fn integrate_rotation(m: &mut Mat4, w: Vec3) {
    let angle = w.length();
    if angle < 1e-9 {
        return;
    }
    let step = axis_angle(w / angle, angle);
    // JS multiply(step, m): apply m first, then the new infinitesimal rotation.
    *m = step * *m;
    orthonormalise(m);
}

fn clamp_spin(b: &mut Body) {
    let len = b.ang_vel.length();
    if len > MAX_SPIN {
        b.ang_vel *= MAX_SPIN / len;
    }
}

fn impact_color(colors: &[[f32; 3]], shape: usize) -> [f32; 3] {
    // A world configured without a colour for this shape still has to report a
    // sensible ring colour, so fall back to the built-in palette.
    colors
        .get(shape)
        .or_else(|| crate::geom::PRIMITIVE_COLORS.get(shape))
        .copied()
        .unwrap_or([1.0, 1.0, 1.0])
}

fn collide_with_walls(
    b: &mut Body,
    t: Tuning,
    hx: f32,
    hy: f32,
    hz: f32,
    colors: &[[f32; 3]],
    impacts: &mut Vec<Impact>,
) {
    let r = b.radius;
    let [x, y, z] = b.pos.to_array();

    if x < -hx + r {
        resolve_wall(b, Vec3::X, -hx + r - x, t, colors, impacts);
    } else if x > hx - r {
        resolve_wall(b, Vec3::NEG_X, x - (hx - r), t, colors, impacts);
    }

    if y < r {
        resolve_wall(b, Vec3::Y, r - y, t, colors, impacts);
    } else if y > hy - r {
        resolve_wall(b, Vec3::NEG_Y, y - (hy - r), t, colors, impacts);
    }

    if z < -hz + r {
        resolve_wall(b, Vec3::Z, -hz + r - z, t, colors, impacts);
    } else if z > hz - r {
        resolve_wall(b, Vec3::NEG_Z, z - (hz - r), t, colors, impacts);
    }
}

/// `depth` is how far the body has crossed the boundary, `n` points back into
/// the room.
fn resolve_wall(
    b: &mut Body,
    n: Vec3,
    depth: f32,
    t: Tuning,
    colors: &[[f32; 3]],
    impacts: &mut Vec<Impact>,
) {
    if depth > 0.0 {
        b.pos += n * depth;
    }
    let vn = b.vel.dot(n);
    if vn >= 0.0 {
        return;
    }

    let approach = -vn;
    // Below the resting speed the contact simply absorbs the last of the
    // bounce; otherwise bodies chatter forever on the floor.
    let e = if approach < WALL_REST_SPEED {
        0.0
    } else {
        t.restitution
    };
    let dv = (1.0 + e) * approach;

    b.vel += n * dv;

    // Friction acts on the *slip* at the contact point, not on the body's
    // tangential velocity: v + omega x r_c, with the contact point at -n * r.
    // A body already rolling has zero slip and is therefore left alone, which is
    // what lets things roll across the floor instead of skidding to a stop.
    let w = b.ang_vel;
    let rc = -n * b.radius;
    let cpv = b.vel + w.cross(rc);
    let cpn = cpv.dot(n);
    let s = cpv - n * cpn;

    b.vel -= s * t.wall_friction;

    // Rolling contact needs omega = (n x v) / r; ease toward it by the same grip.
    let grip = t.spin_coupling;
    let inv_r = 1.0 / b.radius.max(0.05);
    let o = n.cross(b.vel) * inv_r;
    b.ang_vel += (o - w) * grip;
    clamp_spin(b);

    if approach > WALL_IMPACT_SPEED {
        impacts.push(Impact {
            pos: b.pos - n * b.radius,
            n,
            color: impact_color(colors, b.shape),
            base_radius: b.radius * 0.55,
            speed: approach,
            shape: b.shape,
            age: 0.0,
            life: 0.5,
        });
    }
}

/// Contact between two bodies. `n` points from `a` to `b`.
fn collide_pair(
    a: &mut Body,
    b: &mut Body,
    t: Tuning,
    colors: &[[f32; 3]],
) -> Option<Impact> {
    let d = b.pos - a.pos;
    let sum = a.radius + b.radius;
    let dist2 = d.length_squared();
    if dist2 >= sum * sum {
        return None;
    }

    let dist = dist2.sqrt();
    debug_assert!(dist >= 1e-6, "concentric pair handled by the caller");
    let n = d / dist;
    let overlap = sum - dist;

    let inv_a = 1.0 / a.mass;
    let inv_b = 1.0 / b.mass;
    let inv_sum = inv_a + inv_b;

    let corr = (overlap / inv_sum) * 0.7;
    a.pos -= n * inv_a * corr;
    b.pos += n * inv_b * corr;

    let rv = b.vel - a.vel;
    let vn = rv.dot(n);
    if vn >= 0.0 {
        return None;
    }

    let approach = -vn;
    let e = if approach < PAIR_REST_SPEED {
        0.0
    } else {
        t.restitution
    };
    let jimp = (1.0 + e) * approach / inv_sum;

    // The impulse pushes the pair apart: a along -n, b along +n.
    a.vel -= n * jimp * inv_a;
    b.vel += n * jimp * inv_b;

    // Tangential friction works on the slip between the two contact points,
    // spin included, so a pair already rolling against each other keeps rolling.
    // Only this tangential part can spin them: the normal impulse runs along the
    // centre line, so its moment arm is zero.
    let wa = a.ang_vel;
    let wb = b.ang_vel;
    let ra = n * a.radius;
    let rb = -n * b.radius;
    let s = (b.vel + wb.cross(rb)) - (a.vel + wa.cross(ra));
    let sn = s.dot(n);
    let tvec = s - n * sn;
    let slip = tvec.length();

    if slip > 1e-6 {
        let fr = t.pair_friction;
        // Mass-weighted so fr means "remove this fraction of the slip" whatever
        // the pair's masses. a is dragged along the slip, b against it.
        let p = (slip * fr) / inv_sum;
        let j = tvec / slip * p;
        a.vel += j * inv_a;
        b.vel -= j * inv_b;

        // Angular impulse cross(r, J) over a solid-sphere inertia of 0.4*m*r^2.
        // The radius folds into `k`, leaving one shared cross(n, J).
        //
        // Copied verbatim from the JS, quirk included: `b`'s moment arm is
        // `-n * b.radius`, so its torque should carry a minus sign the JS does
        // not apply (both bodies get `+cross(n, J)`). Ported as-is rather than
        // "fixed" — the JS numbers are the oracle. Flagged in the report.
        let c = n.cross(j);
        let k_a = (2.5 * inv_a) / a.radius.max(0.05);
        let k_b = (2.5 * inv_b) / b.radius.max(0.05);
        a.ang_vel += c * k_a;
        b.ang_vel += c * k_b;
        clamp_spin(a);
        clamp_spin(b);
    }

    // Body-to-body rings need a harder hit than wall ones.
    if approach > PAIR_IMPACT_SPEED {
        return Some(Impact {
            pos: a.pos + n * a.radius,
            n: -n,
            color: impact_color(colors, a.shape),
            base_radius: a.radius.min(b.radius) * 0.5,
            speed: approach,
            shape: a.shape,
            age: 0.0,
            life: 0.42,
        });
    }
    None
}

/// `1 / I` for a body, treating it as a solid sphere of its own radius
/// (`I = 0.4 m r^2`). The sphere contacts above fold the same assumption into
/// their constants; the hull contacts need it spelled out because their contact
/// points are not on the centre line, so a normal impulse has a moment arm.
fn inverse_inertia(b: &Body) -> f32 {
    let r = b.scale.max(b.radius).max(0.05);
    1.0 / (0.4 * b.mass * r * r)
}

/// A hull against the six walls. For each wall the body's bounding sphere
/// reaches, the hull's deepest vertices along that wall's normal are found; if
/// they are through the wall, their centroid is the contact point.
#[allow(clippy::too_many_arguments)]
fn collide_hull_with_walls(
    b: &mut Body,
    hull: &Hull,
    t: Tuning,
    hx: f32,
    hy: f32,
    hz: f32,
    colors: &[[f32; 3]],
    impacts: &mut Vec<Impact>,
) {
    let reach = b.scale.max(b.radius);
    // (inward normal, the value of n.dot(x) on the wall)
    let walls = [
        (Vec3::X, -hx),
        (Vec3::NEG_X, -hx),
        (Vec3::Y, 0.0),
        (Vec3::NEG_Y, -hy),
        (Vec3::Z, -hz),
        (Vec3::NEG_Z, -hz),
    ];
    for (n, offset) in walls {
        if n.dot(b.pos) - reach > offset {
            continue;
        }
        let (low, point) = placed(hull, b).lowest_along(n);
        let depth = offset - low;
        if depth <= 0.0 {
            continue;
        }
        b.pos += n * depth;
        let point = point + n * depth;
        let r = point - b.pos;
        let inv_m = 1.0 / b.mass;
        let inv_i = inverse_inertia(b);

        let vn = (b.vel + b.ang_vel.cross(r)).dot(n);
        if vn >= 0.0 {
            continue;
        }
        let approach = -vn;
        let e = if approach < WALL_REST_SPEED { 0.0 } else { t.restitution };
        let rn = r.cross(n);
        let j = (1.0 + e) * approach / (inv_m + inv_i * rn.length_squared());
        b.vel += n * (j * inv_m);
        b.ang_vel += rn * (j * inv_i);

        // Coulomb friction on the slip at the contact point, capped by the
        // normal impulse — so a hull landing on a corner is tripped by it.
        let vc = b.vel + b.ang_vel.cross(r);
        let slip = vc - n * vc.dot(n);
        let speed = slip.length();
        if speed > 1e-6 {
            let dir = slip / speed;
            let rt = r.cross(dir);
            let jt = (speed / (inv_m + inv_i * rt.length_squared()))
                .min(t.wall_friction * HULL_FRICTION_SCALE * j);
            b.vel -= dir * (jt * inv_m);
            b.ang_vel -= rt * (jt * inv_i);
        }
        b.ang_vel *= HULL_CONTACT_SPIN_KEEP;
        clamp_spin(b);

        if approach > WALL_IMPACT_SPEED {
            impacts.push(Impact {
                pos: point,
                n,
                color: impact_color(colors, b.shape),
                base_radius: reach * 0.45,
                speed: approach,
                shape: b.shape,
                age: 0.0,
                life: 0.5,
            });
        }
    }
}

/// Contact between two bodies at least one of which is a hull. The same
/// impulse as [`collide_pair`], except that the contact point is wherever the
/// hull test found it rather than on the line between the centres — so the
/// normal impulse can turn the bodies as well as part them.
fn collide_rigid_pair(
    a: &mut Body,
    b: &mut Body,
    contact: Contact,
    t: Tuning,
    colors: &[[f32; 3]],
) -> Option<Impact> {
    let Contact { n, depth, point } = contact;
    if n == Vec3::ZERO {
        return None;
    }
    let inv_a = 1.0 / a.mass;
    let inv_b = 1.0 / b.mass;
    let inv_sum = inv_a + inv_b;
    let (ii_a, ii_b) = (inverse_inertia(a), inverse_inertia(b));

    let corr = (depth / inv_sum) * 0.7;
    a.pos -= n * inv_a * corr;
    b.pos += n * inv_b * corr;

    let ra = point - a.pos;
    let rb = point - b.pos;
    let rv = (b.vel + b.ang_vel.cross(rb)) - (a.vel + a.ang_vel.cross(ra));
    let vn = rv.dot(n);
    if vn >= 0.0 {
        return None;
    }
    let approach = -vn;
    let e = if approach < PAIR_REST_SPEED { 0.0 } else { t.restitution };
    let (ran, rbn) = (ra.cross(n), rb.cross(n));
    let j = (1.0 + e) * approach
        / (inv_sum + ii_a * ran.length_squared() + ii_b * rbn.length_squared());
    a.vel -= n * (j * inv_a);
    a.ang_vel -= ran * (j * ii_a);
    b.vel += n * (j * inv_b);
    b.ang_vel += rbn * (j * ii_b);

    let rv = (b.vel + b.ang_vel.cross(rb)) - (a.vel + a.ang_vel.cross(ra));
    let slip = rv - n * rv.dot(n);
    let speed = slip.length();
    if speed > 1e-6 {
        let dir = slip / speed;
        let (rat, rbt) = (ra.cross(dir), rb.cross(dir));
        let jt = (speed / (inv_sum + ii_a * rat.length_squared() + ii_b * rbt.length_squared()))
            .min(t.pair_friction * HULL_FRICTION_SCALE * j);
        a.vel += dir * (jt * inv_a);
        a.ang_vel += rat * (jt * ii_a);
        b.vel -= dir * (jt * inv_b);
        b.ang_vel -= rbt * (jt * ii_b);
    }
    a.ang_vel *= HULL_CONTACT_SPIN_KEEP;
    b.ang_vel *= HULL_CONTACT_SPIN_KEEP;
    clamp_spin(a);
    clamp_spin(b);

    if approach > PAIR_IMPACT_SPEED {
        return Some(Impact {
            pos: point,
            n: -n,
            color: impact_color(colors, a.shape),
            base_radius: a.radius.min(b.radius) * 0.5,
            speed: approach,
            shape: a.shape,
            age: 0.0,
            life: 0.42,
        });
    }
    None
}
