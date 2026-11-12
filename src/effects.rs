//! Motion trails and impact sparks: decoration, kept out of the simulation.
//!
//! Neither feeds back into the physics. They read the world after it has
//! stepped and turn what they see into coloured line segments for the
//! renderer's trail pipeline, with a generator of their own so that switching
//! them on cannot change what a seed replays.

use std::collections::VecDeque;

use glam::Vec3;

use crate::gpu_types::LineVertex;
use crate::sim::World;

/// Positions remembered per body. At sixty frames a second this is a third of a
/// second of travel.
const TRAIL_POINTS: usize = 20;
/// Only the first this-many bodies get a trail. A few thousand trails are a
/// blur, and they cost a vertex buffer upload every frame.
const TRAIL_MAX_BODIES: usize = 500;
/// A body slower than this (units/s) draws no trail: a settled pile should not
/// shimmer.
const TRAIL_MIN_SPEED: f32 = 1.2;
const SPARKS_MAX: usize = 2000;
const SPARKS_PER_IMPACT: usize = 9;
const SPARK_LIFE: f32 = 0.55;

#[derive(Copy, Clone, Debug)]
struct Spark {
    pos: Vec3,
    vel: Vec3,
    color: [f32; 3],
    /// Seconds left.
    life: f32,
}

#[derive(Debug, Default)]
pub struct Effects {
    trails: Vec<VecDeque<Vec3>>,
    sparks: Vec<Spark>,
    rng: u64,
    lines: Vec<LineVertex>,
}

impl Effects {
    /// Forget everything in flight — for a reset, a rewind, or any other jump
    /// that would otherwise draw a streak from where a body was to where it
    /// now suddenly is.
    pub fn clear(&mut self) {
        self.trails.clear();
        self.sparks.clear();
    }

    pub fn spark_count(&self) -> usize {
        self.sparks.len()
    }

    fn next_f32(&mut self) -> f32 {
        // splitmix64, the same generator the simulation uses, on its own state.
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 * (1.0 / 16_777_216.0)
    }

    /// Call once after each `World::step(dt)`.
    pub fn after_step(&mut self, world: &World, dt: f32, trails: bool, sparks: bool) {
        if trails {
            let tracked = world.bodies.len().min(TRAIL_MAX_BODIES);
            if self.trails.len() != tracked {
                // The body list changed length, so index i is no longer the body
                // it was: every trail would jump. Start them all again.
                self.trails.clear();
                self.trails.resize_with(tracked, VecDeque::new);
            }
            for (trail, body) in self.trails.iter_mut().zip(&world.bodies) {
                if trail.len() == TRAIL_POINTS {
                    trail.pop_front();
                }
                trail.push_back(body.pos);
            }
        } else {
            self.trails.clear();
        }

        if sparks {
            // An impact is new on the step that made it: it was pushed with age
            // 0 and has been aged by exactly this step's `dt` since.
            let fresh: Vec<(Vec3, Vec3, [f32; 3])> = world
                .impacts
                .iter()
                .filter(|hit| hit.age <= dt * 1.001)
                .map(|hit| (hit.pos, hit.n, hit.color))
                .collect();
            for (pos, n, color) in fresh {
                for _ in 0..SPARKS_PER_IMPACT {
                    if self.sparks.len() >= SPARKS_MAX {
                        break;
                    }
                    // Mostly along the surface, a little away from it.
                    let scatter = Vec3::new(
                        self.next_f32() - 0.5,
                        self.next_f32() - 0.5,
                        self.next_f32() - 0.5,
                    ) * 2.0;
                    let along = scatter - n * scatter.dot(n);
                    let vel = along * (2.0 + self.next_f32() * 3.5)
                        + n * (1.0 + self.next_f32() * 3.0);
                    let life = SPARK_LIFE * (0.5 + 0.5 * self.next_f32());
                    self.sparks.push(Spark {
                        pos: pos + n * 0.02,
                        vel,
                        color: [
                            (color[0] * 0.5 + 0.5).min(1.0),
                            (color[1] * 0.5 + 0.5).min(1.0),
                            (color[2] * 0.5 + 0.5).min(1.0),
                        ],
                        life,
                    });
                }
            }
            let gravity = world.gravity * 0.6;
            for spark in &mut self.sparks {
                spark.vel.y += gravity * dt;
                spark.pos += spark.vel * dt;
                spark.life -= dt;
            }
            self.sparks.retain(|s| s.life > 0.0 && s.pos.y > 0.0);
        } else {
            self.sparks.clear();
        }
    }

    /// This frame's segments, two vertices each.
    pub fn lines(&mut self, world: &World) -> &[LineVertex] {
        self.lines.clear();
        for (trail, body) in self.trails.iter().zip(&world.bodies) {
            if trail.len() < 2 || body.vel.length() < TRAIL_MIN_SPEED {
                continue;
            }
            let c = world
                .colors
                .get(body.shape)
                .copied()
                .unwrap_or([1.0, 1.0, 1.0]);
            let last = (trail.len() - 1) as f32;
            for (i, (a, b)) in trail.iter().zip(trail.iter().skip(1)).enumerate() {
                // Faint at the old end, strongest next to the body.
                let fade = |k: f32| 0.55 * (k / last) * (k / last);
                self.lines.push(LineVertex {
                    pos: a.to_array(),
                    color: [c[0], c[1], c[2], fade(i as f32)],
                });
                self.lines.push(LineVertex {
                    pos: b.to_array(),
                    color: [c[0], c[1], c[2], fade(i as f32 + 1.0)],
                });
            }
        }
        for spark in &self.sparks {
            let alpha = (spark.life / SPARK_LIFE).clamp(0.0, 1.0);
            let tail = spark.pos - spark.vel * 0.035;
            let c = spark.color;
            self.lines.push(LineVertex {
                pos: spark.pos.to_array(),
                color: [c[0], c[1], c[2], alpha],
            });
            self.lines.push(LineVertex {
                pos: tail.to_array(),
                color: [c[0], c[1], c[2], 0.0],
            });
        }
        &self.lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{Impact, SpawnOpts};

    #[test]
    fn a_fresh_impact_throws_sparks_once_and_they_die_out() {
        let mut world = World::default();
        let mut fx = Effects::default();
        let dt = 1.0 / 60.0;
        world.impacts.push(Impact {
            pos: Vec3::new(0.0, 1.0, 0.0),
            n: Vec3::Y,
            color: [1.0, 0.2, 0.2],
            base_radius: 0.3,
            speed: 3.0,
            shape: 0,
            age: dt,
            life: 0.5,
        });
        fx.after_step(&world, dt, false, true);
        assert_eq!(fx.spark_count(), SPARKS_PER_IMPACT);
        // The same impact, a frame older, is not new any more.
        world.impacts[0].age += dt;
        fx.after_step(&world, dt, false, true);
        assert_eq!(fx.spark_count(), SPARKS_PER_IMPACT);
        assert_eq!(fx.lines(&world).len(), SPARKS_PER_IMPACT * 2);
        for _ in 0..120 {
            fx.after_step(&world, dt, false, true);
        }
        assert_eq!(fx.spark_count(), 0, "every spark outlives its life by now");
    }

    #[test]
    fn trails_restart_when_the_body_list_changes_length() {
        let mut world = World::default();
        world.set_seed(7);
        world.spawn(0, SpawnOpts::default());
        let mut fx = Effects::default();
        for _ in 0..5 {
            world.step(1.0 / 60.0);
            fx.after_step(&world, 1.0 / 60.0, true, false);
        }
        assert_eq!(fx.trails[0].len(), 5);
        world.spawn(1, SpawnOpts::default());
        fx.after_step(&world, 1.0 / 60.0, true, false);
        assert_eq!(fx.trails.len(), 2);
        assert_eq!(fx.trails[0].len(), 1, "index 0 started over rather than jumping");
    }
}
