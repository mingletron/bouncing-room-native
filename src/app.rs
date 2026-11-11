//! The window layer: winit events, the surface, the egui HUD, and the frame
//! loop. It is the port's equivalent of the browser build's `main.js` plus
//! `controls.js`, with the DOM controls replaced by egui.
//!
//! Two render passes per frame on the same surface texture: the 3D scene draws
//! into an MSAA attachment that resolves onto the surface, then egui paints over
//! it with `LoadOp::Load` and no depth buffer of its own. The first belongs to
//! [`Gfx`]; the second to `egui_wgpu::Renderer`, whose pipeline is built against
//! the *same* colour format — hence the `config.format = gfx.surface_format`
//! assignment below, which is a correctness requirement rather than tidiness.
//!
//! Two things are worth knowing before editing the physics plumbing here:
//!
//! * The HUD is egui, so keyboard shortcuts are read from egui's own input
//!   instead of from winit. That is what preserves the browser build's rule that
//!   a focused input field keeps its keys — see [`hud`].
//! * The probe in `src/bin/probe.rs` deliberately builds its *own* scene rather
//!   than calling these helpers. If the scene under test and the scene being
//!   tested were the same code, a wrong constant would cancel out in both.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::camera::OrbitCamera;
use crate::geom::{self, ShapeDef};
use crate::gfx::{Disc, Gfx, GfxOptions, RenderTarget, Ring, RoomDraw, Scene};
use crate::gpu_types::Light;
use crate::sim::{self, SpawnOpts, World};

use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, Touch, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

/// Bodies in the room at startup, as in the browser build.
const INITIAL_COUNT: usize = 14;
/// The browser build's HUD asks for a kick of 9, not the simulation module's
/// default of 8.
const KICK: f32 = 9.0;
/// Segments in an impact ring — the same count the WebGL build asked for.
const RING_SEGMENTS: u32 = 44;
/// The seed `World::new` builds its own generator from, restated here so the
/// HUD's seed box can open on the value the first room really came from. Set
/// explicitly in [`App::new`] rather than left to chance: if the simulation's
/// default ever moves, the box moves with it.
const DEFAULT_SEED: u64 = 0x5EED_1234_ABCD_EF00;
/// Simulated time advanced by one frame-step. A sixtieth of a second, which
/// [`World::step`](crate::sim::World::step) splits into two 1/120 substeps — so
/// a scrubbed step is indistinguishable from a frame of a running sim.
const FRAME_STEP: f32 = 1.0 / 60.0;
/// Most frame-steps queued at once, so key auto-repeat scrubbing cannot outrun
/// the display with a growing backlog of simulated time.
const MAX_QUEUED_STEPS: u32 = 8;

/// Scene colours, copied from the WebGL build's `scene` object. The probe keeps
/// its own copy on purpose; see the module note.
const BACKGROUND: [f32; 3] = [0.052, 0.058, 0.072];
const WALL: [f32; 3] = [0.2, 0.22, 0.25];
const FLOOR: [f32; 3] = [0.13, 0.14, 0.16];
const CEIL: [f32; 3] = [0.16, 0.17, 0.19];
const EDGE: [f32; 4] = [0.55, 0.62, 0.72, 0.4];
const GRID_SPACING: f32 = 1.2;
const BLOB_COLOR: [f32; 3] = [0.015, 0.017, 0.022];
const BLOB_INNER: f32 = 0.15;
const PANEL: Disc = Disc {
    center: glam::Vec3::new(0.0, sim::ROOM_HALF_Y - 0.002, 0.0),
    normal: glam::Vec3::new(0.0, -1.0, 0.0),
    radius: 1.7,
    alpha: 0.42,
    inner: 0.45,
    color: [1.0, 0.94, 0.82],
};

#[derive(Clone, Debug)]
struct Settings {
    paused: bool,
    wireframe: bool,
    time_scale: f32,
    count: usize,
    /// The seed the bodies now in the room were spawned from. Re-applied by
    /// [`App::reset`], which is what makes `r` replay a layout instead of rolling
    /// a new one off the end of the generator's stream.
    seed: u64,
    /// What is in the seed box. Deliberately separate from `seed`: the draft is
    /// whatever the user has typed, and only a successful parse reaches `seed` —
    /// an unparsable box must not be able to change the world.
    seed_text: String,
    /// The whole egui overlay is off, which is how a clean picture of the room
    /// gets made. `h` is the way back, and it still works with nothing painted —
    /// see the note in [`hud`] about where that check sits.
    hud_hidden: bool,
    /// Which primitive the next "Spawn one" adds.
    spawn_shape: usize,
    /// How big that one is — unless `spawn_random` leaves it to the sim, which
    /// draws its own scale from 0.42 to 0.78.
    spawn_size: f32,
    spawn_random: bool,
}

/// Everything that only exists once there is a window and a device. It is one
/// struct so that a lost or outdated surface can be torn down and rebuilt
/// without half the fields lingering.
/// The instance is deliberately *not* kept: wgpu documents that an `Instance`
/// does not have to outlive what it made, and the window handle the surface came
/// from belongs to `App::window` anyway.
struct Gpu {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gfx: Gfx,
    egui_renderer: egui_wgpu::Renderer,
    /// GPU mesh handles, index-aligned with `App::shapes`.
    shape_mesh: Vec<usize>,
    room_mesh: usize,
    quad_mesh: usize,
    ring_mesh: usize,
}

pub struct App {
    settings: Settings,
    world: World,
    shapes: Vec<ShapeDef>,
    camera: OrbitCamera,
    /// Simulated time driving the orbiting accent light.
    accent_t: f32,
    /// Frame-steps queued by the HUD, paid by the next frame's physics block.
    /// Queued rather than stepped inside [`hud`] so that there is exactly one
    /// place in the app where the world advances, and so a scrubbed step is
    /// [`FRAME_STEP`] of *simulated* time regardless of when the key was struck.
    step_requests: u32,

    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,

    last: Option<Instant>,
    fps: f32,
    stat_timer: f32,
    readout: String,
    adapter_label: String,

    /// Orbit-camera pointer state. egui claims a press on one of its widgets
    /// before we see it, so a slider drag never also spins the room.
    dragging: bool,
    /// Set while the right button holds the orbit centre to the pointer.
    panning: bool,
    last_cursor: Option<(f32, f32)>,
    touches: HashMap<u64, (f32, f32)>,
    pinch_span: Option<f32>,

    /// Set while the window is fully covered or minimized. Nothing can be
    /// presented to it then, so the redraw loop stops rather than spinning.
    occluded: bool,
    /// The last presentation failure reported, so a persistent one is named once
    /// instead of once per frame.
    present_note: Option<String>,
}

/// Start the app. Called from `main`.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    // `Wait` plus a `request_redraw` every `about_to_wait` gives a continuously
    // animating frame that is still paced by the window system, instead of the
    // busy spin `ControlFlow::Poll` gives.
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}

impl App {
    fn new() -> Self {
        let mut world = World::default();
        // The browser build used `Math.random`; the port's world carries its own
        // fixed seed, so the same run looks the same twice unless told otherwise.
        // That seed is now a control rather than a constant: it is applied here
        // and re-applied on every reset, and the HUD lets you edit it.
        world.set_seed(DEFAULT_SEED);
        for i in 0..INITIAL_COUNT {
            world.record(i % geom::PRIMITIVE_NAMES.len(), SpawnOpts::default());
        }
        world.settle();

        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());

        App {
            settings: Settings {
                paused: false,
                wireframe: false,
                time_scale: 1.0,
                count: INITIAL_COUNT,
                seed: DEFAULT_SEED,
                // Hex, because the seed box is also how a layout gets handed to
                // `probe --seed`, and that flag reads a `0x` prefix as hex.
                seed_text: format!("{DEFAULT_SEED:#x}"),
                hud_hidden: false,
                // The starting room cycles through the primitives in order, so
                // the picker opens on the one that spawn would add next and the
                // first click continues the cycle instead of breaking it.
                spawn_shape: INITIAL_COUNT % geom::PRIMITIVE_NAMES.len(),
                spawn_size: 0.55,
                spawn_random: true,
            },
            world,
            shapes: geom::all_primitives(),
            camera: OrbitCamera::default(),
            accent_t: 0.0,
            step_requests: 0,
            window: None,
            gpu: None,
            egui_ctx,
            egui_state: None,
            last: None,
            fps: 60.0,
            stat_timer: 1.0,
            readout: "booting…".to_owned(),
            adapter_label: String::new(),
            dragging: false,
            panning: false,
            last_cursor: None,
            touches: HashMap::new(),
            pinch_span: None,
            occluded: false,
            present_note: None,
        }
    }

    /// Create the window, the device and the egui renderer. Split out of
    /// `resumed` because a suspended-then-resumed cycle needs it again with the
    /// window already in place.
    fn build_gpu(&mut self, event_loop: &ActiveEventLoop, window: Arc<Window>) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            // winit owns the display handle, and wgpu 30 wants it on the
            // instance before a surface can be made from one of its windows.
            display: Some(Box::new(event_loop.owned_display_handle())),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("this window cannot be presented to");

        let mut gfx = match Gfx::new(&instance, Some(&surface), GfxOptions::default()) {
            Ok(gfx) => gfx,
            Err(e) => {
                // The browser build shows a fatal message rather than throwing;
                // a panic with the same message is the honest native equivalent.
                panic!("no usable GPU: {e}");
            }
        };
        self.adapter_label = gfx.adapter_name();

        let shape_mesh = self
            .shapes
            .iter()
            .map(|s| gfx.upload_mesh(&s.mesh))
            .collect();
        let room_mesh = gfx.upload_mesh(&geom::room_mesh(
            sim::ROOM_HALF_X,
            sim::ROOM_HALF_Y,
            sim::ROOM_HALF_Z,
        ));
        let quad_mesh = gfx.upload_mesh(&geom::quad_mesh());
        let ring_mesh = gfx.upload_mesh(&geom::ring_mesh(RING_SEGMENTS));

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&gfx.adapter(), size.width.max(1), size.height.max(1))
            .expect("the adapter cannot present this window's surface format");
        // The pipelines were built against the format `Gfx` chose from the same
        // capability list. Left at the adapter's own preference the two can
        // disagree — most often an sRGB format the pipelines never asked for —
        // and every draw in the frame would then be a validation error.
        config.format = gfx.surface_format;
        surface.configure(&gfx.device, &config);

        let egui_renderer = egui_wgpu::Renderer::new(
            &gfx.device,
            config.format,
            // Defaults: no MSAA for the UI pass, and no depth buffer — egui's
            // pipeline has no depth stage at all, which is why it can join a
            // pass that has one.
            egui_wgpu::RendererOptions::default(),
        );

        self.egui_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(gfx.device.limits().max_texture_dimension_2d as usize),
        ));

        self.gpu = Some(Gpu {
            surface,
            config,
            gfx,
            egui_renderer,
            shape_mesh,
            room_mesh,
            quad_mesh,
            ring_mesh,
        });
    }

    /// Grow or shrink the room to `count` bodies, settling afterwards so a pile
    /// does not start inside itself. The browser build's count slider did the
    /// same thing.
    fn set_count(settings: &mut Settings, world: &mut World, count: usize) {
        settings.count = count;
        while world.bodies.len() < count {
            let pick = world.bodies.len() % geom::PRIMITIVE_NAMES.len();
            world.record(pick, SpawnOpts::default());
        }
        while world.bodies.len() > count {
            world.bodies.pop();
        }
        world.settle();
    }

    /// Replace every body, leaving the world's tuned gravity, bounce and
    /// friction as the HUD set them — which is why this is not "make a new
    /// World", and why a naive reset quietly throws away the user's settings.
    ///
    /// The seed goes back on the generator first. Without that, `r` would keep
    /// walking down the same stream and give a different pile every press, and
    /// the number in the seed box would describe nothing.
    fn reset(settings: &mut Settings, world: &mut World) {
        world.set_seed(settings.seed);
        world.impacts.clear();
        world.bodies.clear();
        for i in 0..settings.count {
            world.record(i % geom::PRIMITIVE_NAMES.len(), SpawnOpts::default());
        }
        world.settle();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let window = match &self.window {
            Some(existing) => existing.clone(),
            None => {
                let created = event_loop
                    .create_window(
                        WindowAttributes::default()
                            .with_title("Bouncing room — native port")
                            .with_inner_size(LogicalSize::new(900.0, 600.0))
                            .with_min_inner_size(LogicalSize::new(360.0, 240.0)),
                    )
                    .expect("could not create a window");
                Arc::new(created)
            }
        };
        self.window = Some(window.clone());
        self.build_gpu(event_loop, window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // egui sees every event first. `consumed` means it claimed one
        // exclusively, which is exactly what stops a drag that began on a slider
        // from also orbiting the camera.
        let consumed = if let (Some(state), Some(window)) = (&mut self.egui_state, &self.window) {
            state.on_window_event(window.as_ref(), &event).consumed
        } else {
            false
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.on_resize(size),
            // egui tracks the scale factor itself; the window is resized as a
            // consequence of the change, and that lands in `Resized`.
            WindowEvent::ScaleFactorChanged { .. } => {}
            // winit 0.30 reports mouse buttons as `Left`/`Right` rather than a
            // "primary" abstraction, and the left button is what orbits — as in
            // the browser build. A press that egui claimed is not a drag.
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if !consumed => {
                self.dragging = true;
                self.camera.dragging = true;
            }
            // The right button grabs the room rather than the view. It does not
            // set `camera.dragging`, because auto-orbit around a point you have
            // just chosen is a feature rather than a fight — switching it off is
            // the HUD's checkbox, not a side effect of a drag.
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } if !consumed => {
                self.panning = true;
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                ..
            } => {
                self.dragging = false;
                self.camera.dragging = false;
                self.panning = false;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32);
                if let Some((lx, ly)) = self.last_cursor.take() {
                    if self.dragging {
                        self.camera.orbit(x - lx, y - ly);
                    } else if self.panning {
                        // Read before the pan: `units_per_pixel` looks at the
                        // camera's distance, and the pan is about to change the
                        // camera.
                        let upp = self.units_per_pixel();
                        self.camera.pan(x - lx, y - ly, upp);
                        App::clamp_target(&mut self.camera);
                    }
                }
                self.last_cursor = Some((x, y));
            }
            // One wheel path is enough: on macOS a two-finger scroll arrives
            // here too, as a `PixelDelta` rather than a `LineDelta`, and
            // [`App::dolly`] normalises both.
            WindowEvent::MouseWheel { delta, .. } if !consumed => {
                self.dolly(delta);
            }
            WindowEvent::Touch(touch) if !consumed => self.on_touch(touch),
            // A window that is minimized or completely behind other windows
            // cannot be presented to. winit says so here, which is what lets the
            // redraw loop park instead of burning a GPU every frame on a picture
            // nobody can see.
            WindowEvent::Occluded(is_occluded) => {
                self.occluded = is_occluded;
            }
            WindowEvent::RedrawRequested => self.draw(),
            // A focus loss mid-drag would otherwise leave the camera stuck to a
            // pointer that is no longer ours — grabbing the room included.
            WindowEvent::Focused(false) => {
                self.dragging = false;
                self.camera.dragging = false;
                self.panning = false;
                self.last_cursor = None;
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // The room is always moving (bodies, the accent light, the auto-orbit),
        // so there is no idle state to wait for — unless the window cannot be
        // presented to at all, in which case parking is the whole point.
        if !self.occluded {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // `resumed` rebuilds them; holding a surface across a suspend is how
        // these apps get killed on platforms that do suspend.
        self.gpu = None;
        self.egui_state = None;
    }
}

impl App {
    /// Release the textures egui asked to drop, leaving the delta empty.
    ///
    /// Every way out of a frame has to do this, including the frames where the
    /// surface was not drawable: `TexturesDelta` debug-asserts on drop that the
    /// work it queued was handled. egui frees through wgpu's `destroy`, which
    /// leaves an already-submitted command buffer valid.
    fn drain_frees(
        gpu: &mut Gpu,
        delta: &mut egui::epaint::textures::TexturesDelta,
    ) {
        for id in std::mem::take(&mut delta.free) {
            gpu.egui_renderer.free_texture(&id);
        }
    }

    fn on_resize(&mut self, size: PhysicalSize<u32>) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        // A minimized window reports 0×0, and wgpu rejects a zero-sized surface.
        // The real size arrives again when the window is restored, so the last
        // good configuration simply stays in place.
        if size.width == 0 || size.height == 0 {
            return;
        }
        gpu.config.width = size.width;
        gpu.config.height = size.height;
        gpu.surface.configure(&gpu.gfx.device, &gpu.config);
        // `Gfx` notices the new extent on the next frame and rebuilds its MSAA
        // and depth attachments, so there is nothing else to do.
    }

    fn dolly(&mut self, delta: MouseScrollDelta) {
        let units = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(point) => (point.y as f32 / 40.0).clamp(-4.0, 4.0),
        };
        if units != 0.0 {
            self.camera.dolly((-units * 0.12).exp());
        }
    }

    /// How many world units one vertical pixel spans at the current orbit
    /// distance, which is what makes a right-drag grab feel like a grab at both
    /// zoom limits: the room is `2 * distance * tan(fov/2)` units tall on screen.
    fn units_per_pixel(&self) -> f32 {
        let height = self
            .window
            .as_ref()
            .map_or(600.0, |w| w.inner_size().height.max(1) as f32);
        (2.0 * self.camera.distance * (crate::camera::FOV_Y * 0.5).tan()) / height
    }

    /// Keep the orbit centre inside the room.
    ///
    /// An unbounded pan slides the target off to somewhere no picture shows, after
    /// which orbiting sweeps the camera through a huge arc around a point nobody
    /// can see. The room box is the honest limit — everything worth looking at is
    /// inside it, and the interior-culling trick already lets the camera sit
    /// outside the walls — and it is applied here rather than in `camera.rs`
    /// because the room's own constants live in `sim`.
    fn clamp_target(camera: &mut OrbitCamera) {
        let (hx, hy, hz) = (sim::ROOM_HALF_X, sim::ROOM_HALF_Y, sim::ROOM_HALF_Z);
        camera.target = camera
            .target
            .clamp(glam::Vec3::new(-hx, 0.0, -hz), glam::Vec3::new(hx, hy, hz));
    }

    /// One-finger drag orbits; two fingers pinch to dolly. The browser build's
    /// `controls.js` did the same thing with `Touch` events.
    fn on_touch(&mut self, touch: Touch) {
        let (x, y) = (touch.location.x as f32, touch.location.y as f32);
        match touch.phase {
            TouchPhase::Started => {
                self.touches.insert(touch.id, (x, y));
                if self.touches.len() == 2 {
                    self.pinch_span = self.touch_span();
                }
            }
            TouchPhase::Moved => {
                match self.touches.get(&touch.id).copied() {
                    Some((lx, ly)) => {
                        if self.touches.len() == 1 {
                            self.camera.orbit(x - lx, y - ly);
                        }
                    }
                    None => {}
                }
                self.touches.insert(touch.id, (x, y));
                if let (Some(previous), Some(now)) = (self.pinch_span, self.touch_span()) {
                    if previous > 1.0 && now > 1.0 {
                        self.camera.dolly((previous / now).clamp(0.5, 2.0));
                    }
                    self.pinch_span = Some(now);
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.touches.remove(&touch.id);
                if self.touches.len() < 2 {
                    self.pinch_span = None;
                }
            }
        }
    }

    fn touch_span(&self) -> Option<f32> {
        let mut points = self.touches.values().copied().take(2);
        let a = points.next()?;
        let b = points.next()?;
        Some(((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt())
    }

    /// The three lights at simulated time `t`: a warm key in the ceiling
    /// fixture, a cool fill from a low corner, and a coloured accent orbiting
    /// the room so the shading never settles.
    fn lights_at(t: f32) -> [Light; 3] {
        let a = t * 0.55;
        let (hx, hy, hz) = (sim::ROOM_HALF_X, sim::ROOM_HALF_Y, sim::ROOM_HALF_Z);
        [
            Light::new(glam::Vec3::new(0.0, hy - 0.65, 0.0), glam::Vec3::new(1.45, 1.32, 1.12)),
            Light::new(
                glam::Vec3::new(-hx * 0.6, 1.1, hz * 0.6),
                glam::Vec3::new(0.26, 0.38, 0.58),
            ),
            Light::new(
                glam::Vec3::new(
                    a.cos() * hx * 0.55,
                    2.4 + (a * 1.7).sin() * 1.3,
                    a.sin() * hz * 0.55,
                ),
                glam::Vec3::new(0.72, 0.24, 0.55),
            ),
        ]
    }

    /// Ground shadows, splayed outward from the key light rather than stacked
    /// under each body — the browser build's `blobsFor`.
    fn blobs_for(world: &World, lights: &[Light; 3]) -> Vec<Disc> {
        let key = lights[0].pos;
        world
            .bodies
            .iter()
            .map(|b| {
                let h = (b.pos.y - b.radius).max(0.0);
                let (dx, dz) = (b.pos.x - key.x, b.pos.z - key.z);
                let dl = (dx * dx + dz * dz).sqrt();
                let (ox, oz) = if dl > 1e-4 {
                    let spread = (h * 0.34).min(1.6);
                    (dx / dl * spread, dz / dl * spread)
                } else {
                    (0.0, 0.0)
                };
                Disc {
                    center: glam::Vec3::new(
                        (b.pos.x + ox)
                            .clamp(-sim::ROOM_HALF_X + 0.05, sim::ROOM_HALF_X - 0.05),
                        0.0,
                        (b.pos.z + oz)
                            .clamp(-sim::ROOM_HALF_Z + 0.05, sim::ROOM_HALF_Z - 0.05),
                    ),
                    normal: glam::Vec3::Y,
                    radius: b.scale * (1.05 + h * 0.12),
                    alpha: 0.48 / (1.0 + h * 0.42),
                    inner: BLOB_INNER,
                    color: BLOB_COLOR,
                }
            })
            .collect()
    }

    fn rings_for(world: &World) -> Vec<Ring> {
        world
            .impacts
            .iter()
            .map(|hit| Ring {
                pos: hit.pos,
                n: hit.n,
                color: hit.color,
                base_radius: hit.base_radius,
                t: (hit.age / hit.life).clamp(0.0, 1.0),
            })
            .collect()
    }

    fn draw(&mut self) {
        let App {
            settings,
            world,
            shapes,
            camera,
            accent_t,
            step_requests,
            window,
            gpu,
            egui_ctx,
            egui_state,
            last,
            fps,
            stat_timer,
            readout,
            adapter_label,
            present_note,
            ..
        } = self;
        let (Some(window), Some(gpu)) = (window.clone(), gpu.as_mut()) else {
            return;
        };
        let Some(egui_state) = egui_state.as_mut() else {
            return;
        };

        // --- time and physics ------------------------------------------------
        let now = Instant::now();
        let raw_dt = last.map_or(1.0 / 60.0, |t| now.duration_since(t).as_secs_f32());
        *last = Some(now);
        // Clamped to a twentieth of a second exactly as the browser build does:
        // a backgrounded tab or a stopped debugger must not fling every body
        // through a wall on the next frame.
        let raw_dt = raw_dt.min(1.0 / 20.0);

        if settings.paused {
            // Scrubbing: each queued request advances exactly one frame of
            // simulated time, lights and all. Anything still queued from a
            // released pause key is dropped rather than banked up.
            for _ in 0..*step_requests {
                world.step(FRAME_STEP);
                *accent_t += FRAME_STEP;
            }
        } else {
            world.step(raw_dt * settings.time_scale);
            *accent_t += raw_dt;
        }
        *step_requests = 0;
        // The camera keeps orbiting even while the physics is paused, as in the
        // browser build.
        camera.update(raw_dt);

        *fps = fps.mul_add(0.9, (1.0 / raw_dt.max(1e-4)) * 0.1);
        *stat_timer += raw_dt;
        if *stat_timer > 0.25 {
            *stat_timer = 0.0;
            // Paused belongs in the readout and not only on the button: the
            // keyboard does this without anyone looking at the panel, and a
            // frame-step of a running room is not a frame-step at all.
            *readout = format!(
                "{}{:.0} fps · {} bodies · {} contacts · {adapter_label}",
                if settings.paused { "paused · " } else { "" },
                fps.round(),
                world.bodies.len(),
                world.impacts.len()
            );
        }

        let lights = App::lights_at(*accent_t);
        let blobs = App::blobs_for(world, &lights);
        let rings = App::rings_for(world);
        let panels = [PANEL];

        // --- egui ------------------------------------------------------------
        // The HUD wants mutable access to the settings, the world and the
        // camera. The closure that holds that access dies with this statement,
        // so the borrows are re-taken by hand rather than letting the closure
        // move the destructured fields — which would leave nothing for the frame
        // that follows.
        let raw_input = egui_state.take_egui_input(window.as_ref());
        let count_before = settings.count;
        let mut req = HudRequests::default();
        let full_out = {
            let (settings, world, camera, readout, req) = (
                &mut *settings,
                &mut *world,
                &mut *camera,
                &*readout,
                &mut req,
            );
            egui_ctx.run_ui(raw_input, move |ui| {
                hud(ui, settings, world, camera, readout, req, &BODIES_RANGE);
            })
        };
        // Reset replaces the bodies but leaves the tuned gravity and bounce
        // alone, so it has to happen out here where the whole world is reachable.
        if req.reset {
            App::reset(settings, world);
        }
        if settings.count != count_before {
            App::set_count(settings, world, settings.count);
        }
        if req.fullscreen {
            // `Borderless(None)` is the display's own native mode, and on macOS
            // winit gives the window a full-screen space of its own rather than
            // covering the desktop — which is what a viewer like this wants. The
            // size change arrives as a `Resized` event, so `on_resize` reconfigures
            // the surface and `Gfx` rebuilds its attachments on the next frame.
            if window.fullscreen().is_some() {
                window.set_fullscreen(None);
            } else {
                window.set_fullscreen(Some(Fullscreen::Borderless(None)));
            }
        }
        // Frame-steps are asked for here and paid by the *next* frame's physics
        // block, which is what keeps a scrubbed step exactly FRAME_STEP long.
        if req.steps > 0 {
            *step_requests = step_requests.saturating_add(req.steps).min(MAX_QUEUED_STEPS);
        }

        let egui::FullOutput {
            platform_output,
            // Taken apart rather than borrowed: `TexturesDelta` debug-asserts on
            // drop that both halves have been emptied, so each has to be walked
            // by value (see the two `mem::take` calls below).
            mut textures_delta,
            shapes: shapes_out,
            pixels_per_point,
            // Multi-viewport support would mean creating extra OS windows; this
            // HUD lives entirely in the one viewport, so there is nothing to
            // action here.
            viewport_output: _,
        } = full_out;

        // egui's texture uploads come first. They are device work, needed whether
        // or not this frame's surface turns out to be drawable, and skipping them
        // on an undrawable frame is how the font atlas never arrives. `set` maps
        // each texture id to the deltas queued for it, so the inner list has to be
        // walked too.
        for (id, deltas) in std::mem::take(&mut textures_delta.set) {
            for delta in deltas {
                gpu.egui_renderer
                    .update_texture(&gpu.gfx.device, &gpu.gfx.queue, id, &delta);
            }
        }

        let size = window.inner_size();
        let surface_texture = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                // A good frame clears the note, so a *new* failure gets named
                // even if the same kind was already reported.
                *present_note = None;
                texture
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                // The surface changed behind our back (a size or format shift):
                // reconfigure with the size winit already reported and skip this
                // frame rather than draw into a stale target.
                if size.width > 0 && size.height > 0 {
                    gpu.config.width = size.width;
                    gpu.config.height = size.height;
                    gpu.surface.configure(&gpu.gfx.device, &gpu.config);
                }
                App::drain_frees(gpu, &mut textures_delta);
                return;
            }
            other => {
                // Timeout / Occluded / Lost / Validation: nothing to draw into
                // this frame. Named once per distinct cause rather than once per
                // frame, because a window that stays behind another one is a
                // normal state of affairs, not an error worth a flooded terminal.
                let note = format!("{other:?}");
                if present_note.as_deref() != Some(note.as_str()) {
                    eprintln!("presentation: no surface texture this frame: {note}");
                    *present_note = Some(note);
                }
                App::drain_frees(gpu, &mut textures_delta);
                return;
            }
        };
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (width, height) = (
            surface_texture.texture.width(),
            surface_texture.texture.height(),
        );

        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point,
        };
        let clipped = egui_ctx.tessellate(shapes_out, pixels_per_point);

        let mut encoder = gpu
            .gfx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        // Pass 1: the scene, into the MSAA attachment `Gfx` maintains, resolved
        // onto this frame's surface texture.
        let scene = Scene {
            view: camera.view(),
            proj: camera.projection(width, height),
            eye: camera.eye(),
            lights,
            background: BACKGROUND,
            room: RoomDraw {
                mesh: gpu.room_mesh,
                wall: WALL,
                floor: FLOOR,
                ceil: CEIL,
                grid_spacing: GRID_SPACING,
                edge_color: EDGE,
                half: [sim::ROOM_HALF_X, sim::ROOM_HALF_Y, sim::ROOM_HALF_Z],
            },
            shapes,
            shape_mesh: &gpu.shape_mesh,
            bodies: &world.bodies,
            blobs: &blobs,
            panels: &panels,
            rings: &rings,
            quad_mesh: gpu.quad_mesh,
            ring_mesh: gpu.ring_mesh,
            wireframe: settings.wireframe,
        };
        gpu.gfx.render(
            &mut encoder,
            &RenderTarget {
                view: &view,
                format: gpu.config.format,
                width,
                height,
            },
            &scene,
        );

        // Pass 2: egui, loading what pass 1 left instead of clearing it.
        let egui_buffers = gpu.egui_renderer.update_buffers(
            &gpu.gfx.device,
            &gpu.gfx.queue,
            &mut encoder,
            &clipped,
            &screen,
        );
        {
            let ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                // No depth attachment: egui's pipeline has no depth stage.
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            gpu.egui_renderer
                .render(&mut ui_pass.forget_lifetime(), &clipped, &screen);
        }

        gpu.gfx
            .queue
            .submit(std::iter::once(encoder.finish()).chain(egui_buffers));
        window.pre_present_notify();
        gpu.gfx.queue.present(surface_texture);

        App::drain_frees(gpu, &mut textures_delta);
        egui_state.handle_platform_output(window.as_ref(), platform_output);
    }
}

/// The HUD's body-count slider range, the browser build's `0..=40` stretched to
/// the port's own ceiling. Zero is a real setting rather than an edge case: an
/// empty room is how you check the room itself, and both `settle` and
/// `clamp_to_room` walk an empty `bodies` without incident.
const BODIES_RANGE: std::ops::RangeInclusive<usize> = 0..=48;

/// What the HUD wants done but cannot do from inside its own borrow: anything
/// that reaches past the world it is handed, such as re-seeding and rebuilding
/// it at once. Collected in one struct so that a new request costs a field
/// rather than another parameter on [`hud`].
#[derive(Clone, Copy, Debug, Default)]
struct HudRequests {
    reset: bool,
    steps: u32,
    /// Flip full screen. Asked for rather than done because the window belongs to
    /// `App`, not to the panel, and winit wants the call on the event-loop thread.
    fullscreen: bool,
}

/// Parse a seed the way the probe's `--seed` does — decimal, or hex behind `0x` —
/// so a number read off the HUD can be pasted straight into that flag.
/// Underscores and inner spaces are ignored, which is what makes a 64-bit seed
/// in a narrow box still readable.
fn parse_seed(text: &str) -> Option<u64> {
    let text = text.trim();
    let (radix, digits) = match text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        Some(rest) => (16, rest),
        None => (10, text),
    };
    u64::from_str_radix(&digits.replace('_', "").replace(' ', ""), radix).ok()
}

/// A seed for the "Random seed" button, and unlike `Math.random` it lands in the
/// seed box, so the room that follows is reproducible.
///
/// The clock by itself is not enough: macOS reports `SystemTime` to the
/// microsecond, so two presses inside one microsecond — the two halves of a
/// double-click — would rebuild the identical room and read as a broken button.
/// A per-process counter breaks that tie, and the clock is what keeps two
/// separate runs of the app from agreeing.
fn clock_seed() -> u64 {
    static TICKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let since = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let tick = TICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    mix(since.as_nanos() as u64 ^ tick.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// The splitmix64 finaliser, which is also what the simulation's own generator
/// runs on. Mixing is what stops adjacent ticks — seeds a nanosecond apart —
/// from becoming nearly-equal rooms, which is exactly the case the counter
/// exists to cover.
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Step one frame. Requested through [`HudRequests`] rather than performed here,
/// and it parks the free run on the way: a step that raced the wall clock would
/// not be the fixed-size step the readout implies.
fn request_step(settings: &mut Settings, req: &mut HudRequests) {
    settings.paused = true;
    req.steps = req.steps.saturating_add(1).min(MAX_QUEUED_STEPS);
}

/// The control panel and the status readout, as panels on egui's root UI.
///
/// Both mutate the world and the settings directly, which is what lets the HUD
/// stay a hundred lines long instead of routing every control through a queue.
/// The exceptions are the two that reach past the world this is handed — a reset
/// that also has to re-seed, and a frame-step the frame clock pays for — which
/// come back as [`HudRequests`].
fn hud(
    ui: &mut egui::Ui,
    settings: &mut Settings,
    world: &mut World,
    camera: &mut OrbitCamera,
    readout: &str,
    req: &mut HudRequests,
    bodies: &std::ops::RangeInclusive<usize>,
) {
    // Shortcuts, read through egui so that a widget with keyboard focus keeps
    // its keys — the rule the browser build enforced by checking whether the
    // event target was an input element. The seed box is that widget now, so a
    // typed `.` or a typed `0x` stays in the box.
    if ui.memory(|m| m.focused().is_none()) {
        let (space, k, r, w, period, h, f, c) = ui.input_mut(|i| {
            (
                i.consume_key(egui::Modifiers::NONE, egui::Key::Space),
                i.consume_key(egui::Modifiers::NONE, egui::Key::K),
                i.consume_key(egui::Modifiers::NONE, egui::Key::R),
                i.consume_key(egui::Modifiers::NONE, egui::Key::W),
                i.consume_key(egui::Modifiers::NONE, egui::Key::Period),
                i.consume_key(egui::Modifiers::NONE, egui::Key::H),
                i.consume_key(egui::Modifiers::NONE, egui::Key::F),
                i.consume_key(egui::Modifiers::NONE, egui::Key::C),
            )
        });
        if space {
            settings.paused = !settings.paused;
        }
        if k {
            world.kick(KICK);
        }
        // Reset is requested rather than performed: rebuilding the world means
        // touching the gravity and bounce the sliders are bound to, which the
        // caller does once this borrow has ended.
        if r {
            req.reset = true;
        }
        if w {
            settings.wireframe = !settings.wireframe;
        }
        if period {
            request_step(settings, req);
        }
        if h {
            settings.hud_hidden = !settings.hud_hidden;
        }
        if f {
            req.fullscreen = true;
        }
        // Recentring needs no request: the panel is handed the camera already, and
        // unlike a reset this touches nothing outside it.
        if c {
            camera.recenter();
        }
    }

    // Both panels are the entire overlay, so hiding them is what makes a clean
    // picture of the room possible. Note where this sits: the shortcuts above are
    // read first and still work on a frame with no panel on screen, which is the
    // only reason `h` is a way back and not a trap.
    if settings.hud_hidden {
        return;
    }

    egui::Panel::bottom("readout").show(ui, |ui| {
        ui.label(readout);
    });

    egui::Panel::left("controls")
        .resizable(false)
        .show(ui, |ui| {
            // The panel is taller than the window it opens in. A scroll area
            // keeps every control reachable at the 360x240 minimum; the wheel
            // over the panel belongs to egui, and `window_event` already reads
            // that as `consumed`, so scrolling the list never also dollies the
            // camera.
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Bouncing room");
                ui.label("drag to orbit · scroll to zoom · right-drag grabs the room");
                ui.separator();

                let pause_label = if settings.paused { "Resume" } else { "Pause" };
                // Wrapped rather than plain `horizontal`: four buttons will not
                // fit the panel's width at the minimum window size.
                ui.horizontal_wrapped(|ui| {
                    if ui.button(pause_label).clicked() {
                        settings.paused = !settings.paused;
                    }
                    if ui.button("Kick").clicked() {
                        world.kick(KICK);
                    }
                    if ui.button("Reset").clicked() {
                        req.reset = true;
                    }
                    if ui
                        .button("Step")
                        .on_hover_text("advance one 1/60 s frame, pausing first")
                        .clicked()
                    {
                        request_step(settings, req);
                    }
                    if ui
                        .button("Hide")
                        .on_hover_text("h — the panel is gone until you press h again")
                        .clicked()
                    {
                        settings.hud_hidden = true;
                    }
                });
                // A second button row for the two things that are not simulation
                // controls at all: the window, and where the camera is looking.
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .button("Fullscreen")
                        .on_hover_text("f — the display's own native mode")
                        .clicked()
                    {
                        req.fullscreen = true;
                    }
                    if ui
                        .button("Recenter")
                        .on_hover_text("c — back to the opening framing; auto-orbit keeps its setting")
                        .clicked()
                    {
                        camera.recenter();
                    }
                });
                ui.label("space · k · r · w · . step");
                ui.label("h hud · f fullscreen · c recenters");
                ui.separator();

                if ui.button("Spawn one").clicked() {
                    let opts = if settings.spawn_random {
                        SpawnOpts::default()
                    } else {
                        SpawnOpts::with_scale(settings.spawn_size)
                    };
                    world.record(settings.spawn_shape, opts);
                    settings.count = world.bodies.len();
                }
                // Nine one-word buttons rather than a dropdown: the primitives are
                // the subject of the app, and seeing which one is armed is most of
                // the point. `SpawnOpts::with_scale` has existed in the simulation
                // since it was ported and nothing outside its own tests called it
                // until this row.
                ui.horizontal_wrapped(|ui| {
                    for (i, name) in geom::PRIMITIVE_NAMES.iter().enumerate() {
                        ui.selectable_value(&mut settings.spawn_shape, i, *name);
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut settings.spawn_random, "random size");
                    ui.add(egui::Slider::new(&mut settings.spawn_size, 0.3..=1.0)).on_hover_text(
                        "used only with random size off; the sim picks 0.42-0.78 itself",
                    );
                });
                ui.add(egui::Slider::new(&mut settings.count, bodies.clone()))
                    .on_hover_text("0 leaves an empty room");
                ui.label("bodies");

                ui.add(
                    egui::Slider::new(&mut world.gravity, -40.0..=5.0).text("gravity"),
                );
                ui.add(
                    egui::Slider::new(&mut world.restitution, 0.0..=1.0).text("bounce"),
                );
                ui.add(
                    egui::Slider::new(&mut settings.time_scale, 0.05..=2.0)
                        .text("time")
                        .custom_formatter(|t, _| format!("{t:.2}x")),
                );

                // The four dials the contact solver reads on every wall and pair
                // hit. The browser HUD never exposed these either, so this is
                // new ground for the port rather than a parity catch-up, and
                // `spin coupling` is the one worth playing with.
                ui.add(
                    egui::Slider::new(&mut world.drag, 0.0..=1.0)
                        .text("air drag")
                        .custom_formatter(|t, _| format!("{t:.3}")),
                );
                ui.add(
                    egui::Slider::new(&mut world.wall_friction, 0.0..=1.0)
                        .text("wall grip"),
                );
                ui.add(
                    egui::Slider::new(&mut world.pair_friction, 0.0..=1.0)
                        .text("body grip"),
                );
                ui.add(
                    egui::Slider::new(&mut world.spin_coupling, 0.0..=1.0)
                        .text("spin coupling"),
                )
                .on_hover_text("0 skids, 1 grips at once. 0.45 is what makes bodies roll");

                ui.separator();
                ui.checkbox(&mut world.collide, "body collisions");
                ui.checkbox(&mut settings.wireframe, "wireframe");
                ui.checkbox(&mut camera.auto_orbit, "auto-orbit");

                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    ui.label("seed");
                    let field = ui.add(
                        egui::TextEdit::singleline(&mut settings.seed_text)
                            .char_limit(26)
                            .desired_width(150.0),
                    );
                    let draft = parse_seed(&settings.seed_text);
                    let enter = field.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if ui
                        .button("Apply")
                        .on_hover_text("re-seed the room and respawn these bodies")
                        .clicked()
                        || enter
                    {
                        if let Some(seed) = draft {
                            settings.seed = seed;
                            // Apply means "give me that room", and only
                            // `App::reset` puts a seed back on the generator, so
                            // the two always travel together.
                            req.reset = true;
                        }
                    }
                    if draft.is_none() {
                        ui.colored_label(ui.visuals().error_fg_color, "a number, or 0x…");
                    }
                    if ui.button("Random").clicked() {
                        let seed = clock_seed();
                        settings.seed = seed;
                        settings.seed_text = format!("{seed:#x}");
                        req.reset = true;
                    }
                });
                // The point of showing it at all: this is the number that makes a
                // screenshot reproducible, and `probe --seed` takes it.
                ui.label("what `r` replays · try it with probe --seed");
            });
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_settings() -> Settings {
        Settings {
            paused: false,
            wireframe: false,
            time_scale: 1.0,
            count: 5,
            seed: DEFAULT_SEED,
            seed_text: format!("{DEFAULT_SEED:#x}"),
            hud_hidden: false,
            spawn_shape: 0,
            spawn_size: 0.55,
            spawn_random: true,
        }
    }

    #[test]
    fn the_seed_grammar_matches_the_probe_flags() {
        assert_eq!(parse_seed("0x5EED"), Some(0x5EED));
        assert_eq!(parse_seed("0X5eed"), Some(0x5EED));
        assert_eq!(parse_seed("24045"), Some(24_045));
        // Whatever the seed box is opened with has to parse back.
        assert_eq!(parse_seed(&format!("{DEFAULT_SEED:#x}")), Some(DEFAULT_SEED));
        let seed = clock_seed();
        assert_eq!(parse_seed(&seed.to_string()), Some(seed));
        assert_eq!(parse_seed(&format!("{seed:#x}")), Some(seed));
    }

    #[test]
    fn separators_are_ignored_and_nonsense_is_refused() {
        assert_eq!(parse_seed(" 12 34_56 "), Some(123_456));
        assert_eq!(parse_seed("0x_FF"), Some(255));
        for bad in ["", "0x", "nope", "1.5", "--seed"] {
            assert_eq!(parse_seed(bad), None, "{bad:?} must not reach the world");
        }
        // Refusing an out-of-range draft matters more than accepting it: a
        // wrapping seed would silently mean a room nobody can get back to.
        assert_eq!(parse_seed("99999999999999999999999"), None);
    }

    #[test]
    fn back_to_back_random_presses_cannot_agree() {
        // The clock is coarser than the button on macOS, so a run of presses has
        // to disagree without ever waiting on the timer to move.
        let seeds: Vec<u64> = (0..64).map(|_| clock_seed()).collect();
        let unique: std::collections::HashSet<u64> = seeds.iter().copied().collect();
        assert_eq!(unique.len(), seeds.len(), "a repeated seed means a repeated room");
    }

    #[test]
    fn a_step_parks_the_free_run_and_queues_one_frame() {
        let mut settings = test_settings();
        let mut req = HudRequests::default();
        request_step(&mut settings, &mut req);
        request_step(&mut settings, &mut req);
        assert!(settings.paused, "a step must stop the running room");
        assert_eq!(req.steps, 2);
    }

    #[test]
    fn the_step_queue_is_bounded() {
        // Auto-repeat outruns the display; banking that up would make the next
        // resumed frame jump, which is the opposite of what stepping is for.
        let mut settings = test_settings();
        let mut req = HudRequests::default();
        for _ in 0..100 {
            request_step(&mut settings, &mut req);
        }
        assert_eq!(req.steps, MAX_QUEUED_STEPS);
    }

    #[test]
    fn reset_replays_the_seed_it_was_handed() {
        let mut settings = test_settings();
        settings.seed = 0xABCD_1234;
        let mut world = World::default();
        let room = |world: &World| -> Vec<(usize, glam::Vec3, f32)> {
            world
                .bodies
                .iter()
                .map(|b| (b.shape, b.pos, b.scale))
                .collect()
        };
        App::reset(&mut settings, &mut world);
        let first = room(&world);
        App::reset(&mut settings, &mut world);
        assert_eq!(first, room(&world), "`r` must rebuild, not continue");
        assert_eq!(first.len(), settings.count);
    }

    #[test]
    fn a_different_seed_gives_a_different_room() {
        let mut a = test_settings();
        let mut b = test_settings();
        b.seed = a.seed ^ 0x1;
        let mut wa = World::default();
        let mut wb = World::default();
        App::reset(&mut a, &mut wa);
        App::reset(&mut b, &mut wb);
        let differs = wa
            .bodies
            .iter()
            .zip(wb.bodies.iter())
            .any(|(x, y)| x.pos != y.pos || x.scale != y.scale);
        assert!(differs, "the seed box would be decorative otherwise");
    }

    #[test]
    fn the_panel_lays_out_at_the_smallest_window_the_port_allows() {
        // This drives the real `hud` without a window. egui debug-asserts on an
        // infinite layout rect, which is the specific way a scroll area inside a
        // panel goes wrong, and it asserts nothing about a panel that simply
        // overflowed — so the check is run at the 360x240 minimum winit accepts,
        // where overflow is the interesting case.
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let mut settings = test_settings();
        let mut world = World::default();
        let mut camera = OrbitCamera::default();
        let mut req = HudRequests::default();
        for frame in 0..3 {
            let mut input = egui::RawInput::default();
            input.screen_rect = Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(360.0, 240.0),
            ));
            input.time = Some(frame as f64 / 60.0);
            // The last frame opens with an unparsable box, so the error label —
            // a widget that exists only in the failure path — has to lay out too.
            settings.seed_text = if frame == 2 {
                "not a seed".to_owned()
            } else {
                format!("{DEFAULT_SEED:#x}")
            };
            let mut out = ctx.run_ui(input, |ui| {
                hud(
                    ui,
                    &mut settings,
                    &mut world,
                    &mut camera,
                    "paused · 60 fps · 5 bodies",
                    &mut req,
                    &BODIES_RANGE,
                );
            });
            assert!(
                !out.shapes.is_empty(),
                "frame {frame} painted nothing, so the panels were not laid out"
            );
            // egui queued the font atlas for a renderer that does not exist here.
            // `TexturesDelta` debug-asserts on drop with work still in it, which
            // is the same contract [`App::drain_frees`] keeps in a real frame;
            // `clear` is the sanctioned way out of a headless one.
            out.textures_delta.clear();
        }
    }

    #[test]
    fn a_hidden_hud_is_a_clean_picture_and_still_answers_h() {
        // Three frames: hidden, un-hidden by a key event, visible again. The
        // middle one is the point — hiding the panel has to be reversible from the
        // keyboard alone, which it is because the shortcuts are read before the
        // early return in `hud`. Move that return above the shortcut block and this
        // is the test that says why.
        // A nested fn rather than a closure, because each frame needs exclusive
        // use of the settings it is asserting on between calls.
        fn frame(
            ctx: &egui::Context,
            settings: &mut Settings,
            world: &mut World,
            camera: &mut OrbitCamera,
            req: &mut HudRequests,
            events: Vec<egui::Event>,
        ) -> egui::FullOutput {
            let mut input = egui::RawInput {
                events,
                ..Default::default()
            };
            input.screen_rect = Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(900.0, 600.0),
            ));
            let mut out = ctx.run_ui(input, |ui| {
                hud(
                    ui,
                    settings,
                    world,
                    camera,
                    "60 fps · 5 bodies",
                    req,
                    &BODIES_RANGE,
                )
            });
            // Headless: no renderer to apply the font atlas to, and
            // `TexturesDelta` debug-asserts if it is dropped with work pending.
            out.textures_delta.clear();
            out
        }

        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let mut settings = test_settings();
        let mut world = World::default();
        let mut camera = OrbitCamera::default();
        let mut req = HudRequests::default();

        settings.hud_hidden = true;
        let hidden = frame(&ctx, &mut settings, &mut world, &mut camera, &mut req, vec![]);
        assert!(
            hidden.shapes.is_empty(),
            "a hidden HUD painted {} shapes, so the picture is not clean",
            hidden.shapes.len()
        );

        let key = |pressed: bool| egui::Event::Key {
            key: egui::Key::H,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };
        frame(
            &ctx,
            &mut settings,
            &mut world,
            &mut camera,
            &mut req,
            vec![key(true), key(false)],
        );
        assert!(!settings.hud_hidden, "`h` must bring the panel back");

        let shown = frame(&ctx, &mut settings, &mut world, &mut camera, &mut req, vec![]);
        assert!(!shown.shapes.is_empty(), "the panel should paint again");
    }

    #[test]
    fn a_pan_cannot_slide_the_orbit_centre_out_of_the_room() {
        let mut camera = OrbitCamera::default();
        for _ in 0..40 {
            camera.pan(400.0, -400.0, 0.2);
            App::clamp_target(&mut camera);
        }
        let t = camera.target;
        assert!(
            t.x.abs() <= sim::ROOM_HALF_X
                && t.z.abs() <= sim::ROOM_HALF_Z
                && (0.0..=sim::ROOM_HALF_Y).contains(&t.y),
            "the orbit centre escaped the room: {t:?}"
        );

        camera.recenter();
        assert_eq!(camera.target, OrbitCamera::default().target);
    }

    #[test]
    fn zero_bodies_empties_the_room_and_keeps_the_tuning() {
        let mut settings = test_settings();
        let mut world = World::default();
        App::reset(&mut settings, &mut world);
        assert_eq!(world.bodies.len(), settings.count);

        let tuned = (world.gravity, world.restitution, world.spin_coupling);
        App::set_count(&mut settings, &mut world, 0);
        assert_eq!(settings.count, 0);
        assert!(world.bodies.is_empty(), "the slider's low end must reach 0");
        assert_eq!(
            (world.gravity, world.restitution, world.spin_coupling),
            tuned,
            "emptying the room is not a reset"
        );

        // And the room is still renderable/settleable from there.
        world.settle();
        world.step(1.0 / 60.0);
        assert!(world.bodies.is_empty());
    }
}
