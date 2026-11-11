//! Native port of the `bouncing-room` WebGL demo: a room full of bouncing
//! primitives, drawn with wgpu (Metal on macOS, Vulkan on Linux) and controlled from an egui HUD.
//!
//! The split mirrors the browser build:
//!
//! * [`geom`] and [`sim`] are pure logic — mesh generation and the physics — and
//!   are covered by `tests/sim_invariants.rs` with no GPU involved.
//! * [`gfx`] is the render pass, [`gpu_types`] the CPU/GPU shared layouts,
//!   [`camera`] the orbit camera, [`import`] geometry read in from disk.
//! * [`hull`] is the convex-hull collider imports use, [`effects`] the trails
//!   and sparks, and [`record`] the GIF recorder — none of which the browser
//!   build had.
//! * [`app`] glues them to a window: winit events, the surface, the egui HUD,
//!   and the frame loop.

pub mod app;
pub mod camera;
pub mod effects;
pub mod geom;
pub mod gfx;
pub mod gpu_types;
pub mod hull;
pub mod import;
pub mod record;
pub mod sim;

pub use camera::OrbitCamera;
pub use geom::{Mesh, ShapeDef, Vertex};
pub use gfx::{Disc, Gfx, GfxOptions, Ring, RoomDraw, Scene};
pub use sim::{Body, World};
