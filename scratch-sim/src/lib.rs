//! Native port of the simulation and geometry half of the WebGL demo
//! `bouncing-room` (see `docs/PORT_CONTRACT.md`).
//!
//! Two modules:
//! - [`geom`] ports `src/geometry.js` (primitives, room, quad, ring).
//! - [`sim`] ports `src/physics.js` (`World`, `Body`, `Impact`, contacts).
//!
//! The public types are re-exported here so the integrator can `use
//! scratch_sim::{Mesh, Vertex, World, ..}`.

pub mod geom;
pub mod sim;

pub use geom::{
    FLOATS_PER_VERT, Mesh, PRIMITIVE_COLORS, PRIMITIVE_GLOSS, PRIMITIVE_NAMES, ShapeDef, Vertex,
    all_primitives, cone_mesh, cube_mesh, cylinder_mesh, icosa_mesh, octa_mesh, quad_mesh,
    ring_mesh, room_mesh, sphere_mesh, tetra_mesh, torus_mesh,
};
pub use sim::{
    Body, DEFAULT_KICK, Impact, MAX_SPIN, PAIR_FRICTION, PAIR_IMPACT_SPEED, PAIR_REST_SPEED,
    RADIUS_FACTOR_TABLE, SpawnOpts, WALL_FRICTION, WALL_IMPACT_SPEED, WALL_REST_SPEED, World,
    radius_factor,
};
