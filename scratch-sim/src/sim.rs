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

use glam::{Mat4, Vec3};

use crate::geom::PRIMITIVE_NAMES;

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

        // Destructure so `bodies`, `impacts`, `colors` and `rng` can all be
        // borrowed independently inside the loops below.
        let World {
            bodies,
            impacts,
            colors,
            rng,
            ..
        } = self;

        for b in bodies.iter_mut() {
            b.vel.y += g * h;
            b.vel *= damp;
            b.ang_vel *= damp;

            b.pos += b.vel * h;

            integrate_rotation(&mut b.rot, b.ang_vel * h);
        }

        let palette = colors.as_slice();
        for b in bodies.iter_mut() {
            collide_with_walls(b, t, hx, hy, hz, palette, impacts);
        }

        if collide {
            let n = bodies.len();
            for i in 0..n {
                let (left, right) = bodies.split_at_mut(i + 1);
                // Copy `a` out so the rest of the slice stays free to mutate.
                // Nothing else touches bodies[i] while this inner loop runs, so
                // carrying the running updates in the local matches the JS live
                // reference (unlike `settle`, where each pass re-reads it).
                let mut a = left[i];
                for b in right.iter_mut() {
                    let d = b.pos - a.pos;
                    let sum = a.radius + b.radius;
                    if d.length_squared() >= sum * sum {
                        continue;
                    }
                    if d.length() < 1e-6 {
                        // Perfectly concentric pair: pick any direction to split them.
                        b.pos += Vec3::new(
                            rng.range(-0.02, 0.02),
                            0.03,
                            rng.range(-0.02, 0.02),
                        );
                        continue;
                    }
                    if let Some(impact) = collide_pair(&mut a, b, t, palette) {
                        impacts.push(impact);
                    }
                }
                left[i] = a;
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
            age: 0.0,
            life: 0.42,
        });
    }
    None
}
