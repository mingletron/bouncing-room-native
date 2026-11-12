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

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::audio::{self, Audio};
use crate::camera::OrbitCamera;
use crate::effects::Effects;
use crate::geom::{self, ShapeDef};
use crate::gfx::{Disc, Gfx, GfxOptions, RenderTarget, Ring, RoomDraw, Scene};
use crate::gpu_types::Light;
use crate::hull::Hull;
use crate::import::{self, Import};
use crate::record::{self, Recorder};
use crate::sim::{self, Snapshot, SpawnOpts, World};

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

/// Past this many generated bodies the room is a crowd: new bodies are spawned
/// smaller, in proportion, so that a few thousand of them still fit in a room
/// sized for a few dozen.
const CROWD_START: usize = 60;
/// The built-in primitives that are convex and faceted — cube, cone, cylinder,
/// icosahedron, tetrahedron, octahedron — and so can collide as hulls when
/// asked. The sphere and capsule are round, and the torus is not convex.
const HULL_PRIMITIVES: [usize; 6] = [1, 4, 5, 6, 7, 8];
/// Rewind keeps at most this many frames...
const HISTORY_MAX_FRAMES: usize = 900;
/// ...and at most this much memory, whichever runs out first. A room of four
/// thousand bodies is over half a megabyte a frame.
const HISTORY_BUDGET_BYTES: usize = 128 << 20;

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

/// Colours handed to imported geometry, cycled by import order. The primitives
/// have a palette of their own in `geom`, and an import has no material to read
/// (the OBJ's MTL is dropped), so it gets one of these instead of the last
/// primitive's colour.
const IMPORT_PALETTE: [[f32; 3]; 4] = [
    [0.78, 0.8, 0.85],
    [0.9, 0.62, 0.3],
    [0.36, 0.78, 0.6],
    [0.66, 0.42, 0.85],
];
/// The collision radius factor for an import. Every built-in primitive has a
/// tuned factor below 1 because its mesh does not fill its bounding sphere; an
/// import is normalised *to* that sphere (rule 2 of [`import`]), so 1.0 is not an
/// approximation of its collider, it is its collider.
const IMPORT_RADIUS_FACTOR: f32 = 1.0;

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
    /// draws its own scale from 0.42 to 0.78. An import always uses this number
    /// and never the file's own units, which are not trustworthy across tools.
    spawn_size: f32,
    spawn_random: bool,
    /// The path in the import box. Not touched until Import is pressed, so a
    /// half-typed path never reaches the filesystem.
    import_draft: String,
    /// What the last import did, in words: how much geometry arrived, any repair
    /// the reader had to make, or why it refused. It stays on screen until the
    /// next attempt, because a truncation notice that vanishes in one frame is a
    /// notice nobody reads.
    import_note: String,
    /// The conversion still running and how many files wait behind it, rebuilt from the
    /// queue every frame. `import_note` is one slot that anything else overwrites, so
    /// busy-ness needs somewhere that cannot go stale: otherwise a conversion taking ten
    /// seconds stops being mentioned after the first frame, and a count appended to the
    /// note reads as part of some other file's result.
    import_converting: Option<(String, usize)>,
    /// The file being dragged over the window right now, if any: the affordance
    /// that says a drop will import rather than do nothing.
    hovered_file: Option<String>,
    /// Real shadows from the key light. Off, the WebGL build's ground blobs
    /// stand in for them.
    shadows: bool,
    trails: bool,
    sparks: bool,
    /// Collision sound on or off. Off mutes the mixer rather than closing the stream:
    /// the difference is audible only in how fast sound returns when it is switched back
    /// on, and leaving the stream open is what makes the switch free of a hiccup.
    sound: bool,
    /// Master volume, 0..1, applied in the mixer on the audio thread — the headroom
    /// division happens after it, so turning it up cannot clip.
    volume: f32,
    /// Why this machine has no output stream, in the words the panel shows: set once,
    /// when the one attempt to open a device fails, rather than asked again each frame.
    audio_note: Option<String>,
    /// The faceted primitives collide as their actual shape instead of as
    /// spheres. Off by default: the sphere model is the one the browser build
    /// had, and what the seeds were tuned against.
    primitive_hulls: bool,
    /// What the recorder last did: "recording", where a clip was saved, or why
    /// it could not be.
    record_note: String,
}

/// The body the pointer is holding, and how far down the pointer's ray it is
/// held. The distance is fixed when the body is picked up (and changed by the
/// wheel), which is what turns a 2D pointer into a 3D position.
#[derive(Copy, Clone, Debug)]
struct Grab {
    index: usize,
    distance: f32,
}

/// The recent past, for rewinding: one snapshot of the world per simulated
/// frame, with the accent light's clock so the lighting rewinds too.
#[derive(Default)]
struct History {
    frames: VecDeque<(Snapshot, f32)>,
    bytes: usize,
    /// Where the room has been scrubbed back to, when it has. `None` is the
    /// usual case: the present, at the end of `frames`.
    cursor: Option<usize>,
}

impl History {
    fn clear(&mut self) {
        self.frames.clear();
        self.bytes = 0;
        self.cursor = None;
    }

    /// The frame the room is showing.
    fn position(&self) -> usize {
        self.cursor.unwrap_or(self.frames.len().saturating_sub(1))
    }

    /// Add the world as it is now. If the room had been scrubbed back, the
    /// frames after that point are a future that is no longer going to happen,
    /// and they go first.
    fn record(&mut self, world: &World, accent_t: f32) {
        if let Some(at) = self.cursor.take() {
            while self.frames.len() > at + 1 {
                if let Some((gone, _)) = self.frames.pop_back() {
                    self.bytes -= gone.bytes();
                }
            }
        }
        let snapshot = world.snapshot();
        self.bytes += snapshot.bytes();
        self.frames.push_back((snapshot, accent_t));
        while self.frames.len() > 1
            && (self.frames.len() > HISTORY_MAX_FRAMES || self.bytes > HISTORY_BUDGET_BYTES)
        {
            if let Some((gone, _)) = self.frames.pop_front() {
                self.bytes -= gone.bytes();
            }
        }
    }

    /// Put the world back to frame `index`. False if there is no such frame.
    fn seek(&mut self, index: usize, world: &mut World, accent_t: &mut f32) -> bool {
        match self.frames.get(index) {
            Some((snapshot, t)) => {
                world.restore(snapshot);
                *accent_t = *t;
                self.cursor = Some(index);
                true
            }
            None => false,
        }
    }
}

/// Where a clip is in its life.
enum Clip {
    Idle,
    Recording(Recorder),
    /// Frames have stopped; the worker is still encoding what it was sent.
    Saving(JoinHandle<Result<(), String>>, PathBuf),
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

/// One imported object, as the app keeps it.
///
/// `shape_index` is where its mesh sits in the three lists that must stay
/// index-aligned because `Body::shape` indexes all of them: `App::shapes`,
/// `Gpu.shape_mesh` and `World::colors`. The name is the file stem — which
/// `Mesh::name` cannot carry, being a `&'static str` — and it is what the HUD
/// lists and a removal would act on.
#[derive(Clone, Debug)]
struct ImportedShape {
    source: String,
    shape_index: usize,
    triangles: usize,
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
    /// Imported objects, in the order they arrived, each named by its file stem.
    imports: Vec<ImportedShape>,
    /// Files waiting for a device: queued from the command line before there is a
    /// window, and by a drag at any time.
    pending: Vec<PathBuf>,
    /// Imports whose conversion is running on a worker thread, one receiver each.
    ///
    /// A format with no reader in this process is converted by an external tool,
    /// which costs seconds — and the thread that would wait for it is the one that
    /// draws the room. So the slow formats go away from the frame and answer here on
    /// a later one, exactly like the file dialog above. In-process formats stay
    /// synchronous: they are milliseconds, and deferring them would only make the
    /// panel lie about when something arrived. Each entry carries the file's name with
    /// its receiver, because a failure has to say which file failed and the panel has to
    /// say what is running now.
    converting: Vec<(String, mpsc::Receiver<Result<Import, String>>)>,
    /// The output stream, once there is one to have. `None` until the first frame that
    /// has a window and wants sound — so a headless run, a test and `--mute` never touch
    /// the audio system at all — and `Err(reason)` on a machine with no usable device,
    /// which the panel states in words instead of panicking: containers, remote sessions
    /// and boxes with no output hardware all land here, and the room does not care.
    audio: Option<Result<Audio, String>>,
    effects: Effects,
    history: History,
    grab: Option<Grab>,
    clip: Clip,
    /// The open file dialog, if one is up: it answers on this channel from its
    /// own thread, so the room keeps animating behind it.
    dialog: Option<mpsc::Receiver<Vec<PathBuf>>>,
    /// `--record S`: record a clip of this many seconds from the first frame,
    /// then quit. `None` once the clip has been started.
    auto_record: Option<f32>,
    /// Set while that clip is running, to the length it should stop at.
    auto_stop: Option<f32>,
    /// Asked to exit; honoured by `about_to_wait`, which has the event loop.
    quit: bool,

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

/// The window binary's own usage. Short on purpose — the probe has the long help,
/// and this only needs to say how to bring a file in without a window open first.
const USAGE: &str = "\
bouncing-room [options]

  --import PATH   read an object into the room once the window exists; repeatable,
                  and equivalent to dragging the file onto the window.
                  Read here: .obj, .stl, .gltf, .glb. Converted by Blender (or
                  assimp, for .fbx) first, which costs seconds: .fbx, .blend,
                  .usd, .usda, .usdc, .usdz.
  --mute          start with collision sound off; `s` turns it back on, and the
                  panel has a volume slider.
  --record S      record S seconds (at most 20) as an animated GIF, print where
                  it was saved, and quit.
  --help, -h      this text

Keys: space pause · k kick · r reset · w wireframe · . step · h hud · f fullscreen
· c recenters · x clear all · , step back · o open a file · v record a clip ·
s collision sound.
Drag to orbit, scroll to zoom, right-drag to pan. Click a body to hold it and
let go to throw it.";

/// The paths named by `--import`, in the order given. An unrecognised flag is
/// reported and ignored rather than fatal: a stray argument from a launcher should
/// not cost the user their window.
fn cli_args() -> (Vec<PathBuf>, Option<f32>, bool) {
    let mut args = std::env::args().skip(1).peekable();
    let mut paths = Vec::new();
    let mut record = None;
    let mut muted = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--import" => match args.next() {
                Some(path) => paths.push(PathBuf::from(path)),
                None => eprintln!("--import needs a path"),
            },
            "--record" => match args.next().and_then(|s| s.parse::<f32>().ok()) {
                Some(seconds) if seconds > 0.0 => record = Some(seconds.min(record::MAX_SECONDS)),
                _ => eprintln!("--record needs a number of seconds"),
            },
            // Sound is the one control worth starting off from a command line: a
            // recording run, a demo capture, or anyone who wants the room silent on
            // launch. `s` turns it back on without a restart.
            "--mute" => muted = true,
            "--help" | "-h" => {
                println!("{USAGE}");
                // Not an error and nothing to unwind: the caller asked for help and
                // has just been given it, before there is a window to tear down.
                std::process::exit(0);
            }
            other => eprintln!("ignoring unrecognised argument {other}"),
        }
    }
    (paths, record, muted)
}

/// Start the app. Called from `main`.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let (imports, record, muted) = cli_args();
    let event_loop = EventLoop::new()?;
    // `Wait` plus a `request_redraw` every `about_to_wait` gives a continuously
    // animating frame that is still paced by the window system, instead of the
    // busy spin `ControlFlow::Poll` gives.
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::new();
    app.auto_record = record;
    // Set before the first frame, so a muted launch never opens an output stream at
    // all rather than opening one and then silencing it.
    app.settings.sound = !muted;
    // Queued rather than imported here: there is no device yet, and registering a
    // shape needs one. The first frame that has a `Gpu` reads them.
    for path in imports {
        app.queue_import(&path);
    }
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
                import_draft: String::new(),
                import_note: String::new(),
                import_converting: None,
                hovered_file: None,
                shadows: true,
                trails: false,
                sparks: true,
                sound: true,
                volume: 0.7,
                audio_note: None,
                primitive_hulls: false,
                record_note: String::new(),
            },
            world,
            shapes: geom::all_primitives(),
            camera: OrbitCamera::default(),
            accent_t: 0.0,
            step_requests: 0,
            imports: Vec::new(),
            pending: Vec::new(),
            converting: Vec::new(),
            audio: None,
            effects: Effects::default(),
            history: History::default(),
            grab: None,
            clip: Clip::Idle,
            dialog: None,
            auto_record: None,
            auto_stop: None,
            quit: false,
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

    /// True for a body the room generated itself rather than one that came off
    /// disk. The shape index is the whole test: generated bodies can only refer to
    /// the nine built-in primitives, so anything at or past that range is an
    /// import. `World` does not track the difference, and the difference matters
    /// whenever a control means "the generated set" rather than "everything".
    fn is_generated(body: &sim::Body) -> bool {
        body.shape < geom::PRIMITIVE_NAMES.len()
    }

    /// How many bodies the count slider is responsible for, which is not how many
    /// are in the room once anything has been imported.
    fn generated_count(world: &World) -> usize {
        world.bodies.iter().filter(|b| Self::is_generated(b)).count()
    }

    /// Grow or shrink the *generated* population to `count`, settling afterwards so
    /// a pile does not start inside itself — the browser build's count slider did
    /// the same thing. What is new is that imported bodies sit outside this count
    /// and are not touched by it.
    fn set_count(settings: &mut Settings, world: &mut World, count: usize) {
        settings.count = count;
        let mut generated = Self::generated_count(world);
        while generated < count {
            let pick = generated % geom::PRIMITIVE_NAMES.len();
            if count <= CROWD_START {
                // Spawn and settle one at a time, as the browser build did.
                world.record(pick, SpawnOpts::default());
            } else {
                // A crowd is spawned in bulk and settled once at the end:
                // settling after each of several thousand spawns is the
                // difference between a slider and a wait.
                world.spawn(pick, Self::crowd_opts(count, generated));
            }
            generated += 1;
        }
        while Self::generated_count(world) > count {
            // Remove the newest generated body rather than the tail of the list.
            // Popping the tail would delete an imported object, which is not what
            // dragging a slider over the generated set means.
            match world.bodies.iter().rposition(Self::is_generated) {
                Some(i) => {
                    world.bodies.remove(i);
                }
                None => break,
            }
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
    /// Rebuild the generated bodies, leaving the world's tuned gravity, bounce and
    /// friction as the HUD set them — which is why this is not "make a new World",
    /// and why a naive reset quietly throws away the user's settings.
    ///
    /// The seed goes back on the generator first. Without that, `r` would keep
    /// walking down the same stream and give a different pile every press, and the
    /// number in the seed box would describe nothing.
    ///
    /// Imports survive a reset. Re-rolling the generated set should not discard
    /// geometry that cost a trip to disk to make; [`App::clear_all`] is the control
    /// that does, and it is a different button.
    fn reset(settings: &mut Settings, world: &mut World) {
        world.set_seed(settings.seed);
        world.impacts.clear();
        world.bodies.retain(|b| !Self::is_generated(b));
        for i in 0..settings.count {
            let pick = i % geom::PRIMITIVE_NAMES.len();
            if settings.count <= CROWD_START {
                world.record(pick, SpawnOpts::default());
            } else {
                world.spawn(pick, Self::crowd_opts(settings.count, i));
            }
        }
        world.settle();
    }

    /// How big body `i` of a crowd of `count` should be. The sizes keep the
    /// spread the simulation would have drawn (0.42 to 0.78) and shrink together
    /// as the cube root of the head count, which holds the total volume of the
    /// crowd roughly steady however many there are.
    fn crowd_opts(count: usize, i: usize) -> SpawnOpts {
        let shrink = (CROWD_START as f32 / count.max(1) as f32).cbrt();
        let vary = (i as f32 * 0.618_034).fract();
        SpawnOpts::with_scale((0.42 + 0.36 * vary) * shrink)
    }

    /// Give the faceted primitives hull colliders, or take them away again.
    fn set_primitive_hulls(world: &mut World, shapes: &[ShapeDef], on: bool) {
        let builtins = geom::PRIMITIVE_NAMES.len();
        if world.colliders.len() < builtins {
            world.colliders.resize(builtins, None);
        }
        for i in HULL_PRIMITIVES {
            world.colliders[i] = if on {
                let points: Vec<glam::Vec3> = shapes[i]
                    .mesh
                    .vertices
                    .iter()
                    .map(|v| glam::Vec3::from(v.position))
                    .collect();
                Hull::from_points(&points).map(Arc::new)
            } else {
                None
            };
        }
    }

    /// Empty the room of every body, generated and imported alike, and release the
    /// imported shapes from the three per-shape lists that `Body::shape` indexes.
    ///
    /// The GPU buffers stay resident: `Gfx` has no eviction, and a handful of
    /// imported meshes is not worth inventing an eviction policy for. After this
    /// nothing refers to them, so they cost memory and no draw calls.
    fn clear_all(
        settings: &mut Settings,
        world: &mut World,
        shapes: &mut Vec<ShapeDef>,
        shape_mesh: &mut Vec<usize>,
        imports: &mut Vec<ImportedShape>,
    ) {
        world.bodies.clear();
        world.impacts.clear();
        let keep = geom::PRIMITIVE_NAMES.len();
        // Truncating rather than clearing: the nine built-ins are addressed by
        // index, and the generated bodies still point at them.
        // Truncating rather than filtering is only sound because imports are
        // appended, so every one of them sits past the nine built-ins. Asserted
        // rather than assumed, because `add_import` is the only place that hands
        // out indices and a future change there could break this silently.
        debug_assert!(
            imports.iter().all(|i| i.shape_index >= keep),
            "an imported shape sits inside the built-in range: {imports:?}"
        );
        shapes.truncate(keep);
        shape_mesh.truncate(keep);
        world.colors.truncate(keep);
        world.colliders.truncate(keep);
        imports.clear();
        settings.count = 0;
        settings.import_note = "cleared every body".to_owned();
    }

    /// Drain the conversions that have answered, leaving the ones that have not.
    ///
    /// Split out of the frame loop so the rules below are testable rather than merely
    /// commented: a frame that polls must never wait on a slow conversion, and two
    /// details carry weight. `retain_mut` walks in the order the receivers were pushed,
    /// so a batch that finishes inside one frame is applied in the order the files were
    /// dropped rather than the order the tools happened to finish in; and a disconnected
    /// receiver becomes an error rather than silence, because a worker that panicked
    /// would otherwise leave the panel saying *converting …* for the rest of the
    /// session. The file's name lives in the vector with its receiver so the answer
    /// that cannot name one — a dead worker — can be told which file it was, at the
    /// point where that message is made. The other answers already name their file:
    /// every `ImportError` variant prints a stem or a path, and a caller that wrapped
    /// them in the path too read "…/scene.blend: scene: cannot convert .blend: …" in a
    /// panel 360 px wide.
    fn take_finished(
        converting: &mut Vec<(String, mpsc::Receiver<Result<Import, String>>)>,
    ) -> Vec<Result<Import, String>> {
        let mut finished = Vec::new();
        converting.retain_mut(|(name, receiver)| match receiver.try_recv() {
            Ok(result) => {
                finished.push(result);
                false
            }
            Err(mpsc::TryRecvError::Empty) => true,
            Err(mpsc::TryRecvError::Disconnected) => {
                // The only answer of the three that has no way to name its file: every
                // `ImportError` variant already prints a stem or a path, so the caller
                // that shows this must not add one — a panel 360 px wide would read
                // "/Users/me/m/scene.blend: scene: cannot convert .blend: …". This one
                // names itself here, where the name is actually missing.
                finished.push(Err(format!(
                    "{name}: the conversion thread ended without an answer"
                )));
                false
            }
        });
        finished
    }

    /// Read a file from disk and report the outcome in the panel, whether that
    /// outcome is geometry or a reason it was refused.
    fn import_one(
        settings: &mut Settings,
        world: &mut World,
        shapes: &mut Vec<ShapeDef>,
        gpu: &mut Gpu,
        imports: &mut Vec<ImportedShape>,
        path: &Path,
    ) {
        match import::import_path(path) {
            Ok(import) => Self::add_import(settings, world, shapes, gpu, imports, import),
            Err(e) => {
                // Shown rather than logged: the panel is where the path was typed,
                // so it is where the answer belongs. The marker is what lets the
                // panel colour it as a failure without a second field to track.
                settings.import_note = format!("✗ {e}");
            }
        }
    }

    /// Register an imported mesh as a shape and put one body holding it into the
    /// room. Kept apart from [`App::import_one`] because the tests feed this a
    /// mesh built in memory, with no file and no filesystem behind it.
    fn add_import(
        settings: &mut Settings,
        world: &mut World,
        shapes: &mut Vec<ShapeDef>,
        gpu: &mut Gpu,
        imports: &mut Vec<ImportedShape>,
        import: Import,
    ) {
        // Read everything the note and the body need before `import.mesh` moves
        // into the shape table.
        let triangles = import.mesh.triangle_count();
        let vertices = import.mesh.vertex_count();
        let source = import.source.clone();
        let notes = import.notes.clone();
        let dropped = import.dropped_triangles;
        let hull_corners = import.hull.as_ref().map(|h| h.verts.len());

        let shape_index = shapes.len();
        let color = IMPORT_PALETTE[imports.len() % IMPORT_PALETTE.len()];
        // A mesh that brought its own colours (a glTF with materials) is drawn
        // in them: the body colour multiplies the vertex colour, so it is white.
        let body_color = if import.mesh.colors.is_some() { [1.0; 3] } else { color };
        let handle = gpu.gfx.upload_mesh(&import.mesh);
        shapes.push(ShapeDef {
            mesh: import.mesh,
            color: body_color,
            gloss: import.gloss.unwrap_or(import::DEFAULT_GLOSS),
        });
        gpu.shape_mesh.push(handle);
        // The impact-ring palette is indexed by shape, so an imported body that
        // wants a sensible ring colour has to have an entry here.
        world.colors.push(color);
        // And so is the collider list. It is allowed to be shorter than the
        // shape table (a missing entry is a sphere), so pad it up to this shape.
        if world.colliders.len() < shape_index {
            world.colliders.resize(shape_index, None);
        }
        world.colliders.truncate(shape_index);
        world.colliders.push(import.hull.map(Arc::new));

        // The fragile part of this feature: three separately-owned lists that
        // `Body::shape` indexes, which must not be allowed to drift.
        debug_assert_eq!(shapes.len(), gpu.shape_mesh.len());
        debug_assert_eq!(shapes.len(), world.colors.len());

        let nth = imports.len();
        imports.push(ImportedShape {
            source: source.clone(),
            shape_index,
            triangles,
        });

        let scale = settings.spawn_size;
        let radius = scale * IMPORT_RADIUS_FACTOR;
        let (hx, hy, hz) = (sim::ROOM_HALF_X, sim::ROOM_HALF_Y, sim::ROOM_HALF_Z);
        // Upright, at rest, above the pile. `World::spawn` is deliberately not
        // used: it randomises orientation and velocity, and an imported object's
        // modelled up-axis is information the file supplied that the room has no
        // business discarding. The golden-angle offset stops repeated imports from
        // landing on top of each other without needing a source of randomness.
        let angle = nth as f32 * 2.399_963_3;
        let reach = (0.8 + 0.4 * nth as f32).min((hx - radius).max(0.0));
        let pos = glam::Vec3::new(
            (reach * angle.cos()).clamp(-(hx - radius), hx - radius),
            (hy - radius - 0.2).max(radius),
            (reach * angle.sin()).clamp(-(hz - radius), hz - radius),
        );
        world.bodies.push(sim::Body {
            shape: shape_index,
            scale,
            radius,
            // The same rule `World::spawn` uses — mass follows volume — restated
            // here rather than called, because an import's volume is its bounding
            // sphere and its placement is not random.
            mass: scale * scale * scale * 4.0 + 0.2,
            pos,
            vel: glam::Vec3::ZERO,
            rot: glam::Mat4::IDENTITY,
            ang_vel: glam::Vec3::ZERO,
        });
        world.settle();

        // The counts and every repair the reader made, in the panel. A mesh that
        // was cut down or had its normals rebuilt has to say so where the user can
        // still see it, not only in a terminal they detached.
        let mut note = format!("✓ {source}: {triangles} triangles, {vertices} vertices");
        if dropped > 0 {
            note.push_str(&format!(" — {dropped} dropped by the index limit"));
        }
        if let Some(corners) = hull_corners {
            note.push_str(&format!("\ncollides as a {corners}-corner hull"));
        }
        for note_extra in notes {
            note.push_str(&format!("\n{note_extra}"));
        }
        settings.import_note = note;
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
                // A press on a body picks it up; a press on empty room orbits.
                self.grab = self.pick_at_cursor();
                self.dragging = self.grab.is_none();
                // Either way the auto-orbit stands still while the button is
                // down: the room should not turn under something being held.
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
                // Letting go of a held body needs no code at all: it keeps the
                // velocity the pointer last gave it, and that is the throw.
                self.grab = None;
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
            WindowEvent::CursorLeft { .. } => {
                // No pointer, so nothing to pick with — and a body held when the
                // pointer leaves is let go where it is.
                self.last_cursor = None;
                self.grab = None;
            }
            WindowEvent::Touch(touch) if !consumed => self.on_touch(touch),
            // A file dragged over the window names itself here, which is the only
            // way to tell the user a drop will do something. winit gives real
            // paths, so no dialog and no dependency is needed for this. These
            // three events arrive on macOS and X11; winit 0.30 does not deliver
            // drops under Wayland, where the path box and `--import` remain.
            WindowEvent::HoveredFile(path) => {
                self.settings.hovered_file = path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned());
            }
            WindowEvent::HoveredFileCancelled => {
                self.settings.hovered_file = None;
            }
            WindowEvent::DroppedFile(path) => {
                self.settings.hovered_file = None;
                self.queue_import(&path);
            }
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
                self.grab = None;
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
        if self.quit {
            _event_loop.exit();
            return;
        }
        if !self.occluded {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Everything that talks to the window system goes *now*, while the
        // event loop — and with it the connection to the display — still
        // exists. `run_app` consumes the loop, so by the time it returns and
        // `App` is dropped in the ordinary way the connection is already gone.
        // Under Wayland that was a segfault on every exit: egui's clipboard
        // runs a thread of its own on that connection, and it was still using
        // it. The order matters too: the audio thread first, then egui's state, then
        // the surface, then the window the surface was made from.
        //
        // Sound goes first because its callback runs on a thread of its own and reads
        // the mixer, and cpal 0.15 offers no shutdown — only pause. Pausing is what
        // shows the callback has stopped before anything it could reach is gone;
        // dropping the handle straight after is what releases the device.
        if let Some(Ok(stream)) = self.audio.take() {
            if let Err(why) = stream.pause() {
                eprintln!("audio: {why}");
            }
        }
        self.egui_state = None;
        self.gpu = None;
        self.window = None;
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
        if units == 0.0 {
            return;
        }
        match self.grab.as_mut() {
            // While a body is held the wheel moves *it* along the pointer's ray,
            // nearer or farther, rather than moving the camera.
            Some(grab) => grab.distance = (grab.distance * (units * 0.08).exp()).clamp(1.5, 60.0),
            None => self.camera.dolly((-units * 0.12).exp()),
        }
    }

    /// The body under the pointer, if there is one, ready to be held.
    fn pick_at_cursor(&self) -> Option<Grab> {
        let (x, y) = self.last_cursor?;
        let size = self.window.as_ref()?.inner_size();
        let (origin, dir) = self.camera.ray(x, y, size.width, size.height);
        let (index, _) = self.world.pick(origin, dir)?;
        Some(Grab {
            index,
            distance: (self.world.bodies[index].pos - origin).dot(dir),
        })
    }

    /// Accept a file for import, or say immediately why it cannot be. A format
    /// nothing can read is refused here rather than queued, so the answer is on
    /// screen while the pointer is still moving, and the path lands in the box as
    /// well — editing it into a relative path is then a matter of typing.
    fn queue_import(&mut self, path: &Path) {
        if !import::is_supported(path) {
            self.settings.import_note = import::unsupported_message(path);
            return;
        }
        self.pending.push(path.to_path_buf());
        self.settings.import_draft = path.display().to_string();
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
            imports,
            pending,
            converting,
            audio: audio_state,
            effects,
            history,
            grab,
            clip,
            dialog,
            last_cursor,
            auto_record,
            auto_stop,
            quit,
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

        // The very first frame goes into the history before anything moves, so
        // that rewinding all the way back reaches the room as it was spawned.
        if history.frames.is_empty() {
            history.record(world, *accent_t);
        }

        // Open the output device on the first frame that has a window and wants sound,
        // never in `new`: a machine with no device gets one `Err` remembered here and
        // shown once in the panel rather than a stream that cannot exist and a panic that
        // would take the window with it. Retrying every frame would mean asking the audio
        // system the same failing question sixty times a second.
        if settings.sound && audio_state.is_none() {
            let opened = Audio::open(settings.volume);
            if let Err(why) = opened.as_ref() {
                // Also to stderr: the panel line is small, and a headless or remote
                // session is exactly where nobody is looking at the panel.
                eprintln!("audio: {why} — running silent");
                // Stated once, where the panel can see it: `hud` is a separate function
                // with its own borrow list and no view of this field, so the reason
                // travels in Settings rather than growing another parameter.
                settings.audio_note = Some(why.clone());
            }
            *audio_state = Some(opened);
        }
        // A remembered failure is cleared while sound is off, which makes the control
        // that switches sound off and back on double as the manual retry: a user who has
        // just plugged in an output device (or started a PulseAudio/ PipeWire session
        // under the app) can get sound without quitting, while a permanently broken audio
        // system is still only asked once per sound-on period rather than sixty times a
        // second.
        if !settings.sound && matches!(audio_state, Some(Err(_))) {
            *audio_state = None;
            settings.audio_note = None;
        }
        if let Some(Ok(stream)) = audio_state.as_mut() {
            // Cheaper than tracking whether either control moved, and it cannot be
            // missed: two locks a frame, each held for a store.
            stream.set_muted(!settings.sound);
            stream.set_volume(settings.volume);
        }

        // A held body is pulled toward the pointer before the step, so the
        // step carries it there. A body that no longer exists is let go.
        if let Some(held) = *grab {
            match (*last_cursor, held.index < world.bodies.len()) {
                (Some((x, y)), true) => {
                    let size = window.inner_size();
                    let (origin, dir) = camera.ray(x, y, size.width, size.height);
                    world.hold(held.index, origin + dir * held.distance);
                }
                (_, false) => *grab = None,
                _ => {}
            }
        }

        // The one place the world advances. Each step is followed by the
        // effects, which read what it did, and by a snapshot for rewinding.
        let mut advance = |world: &mut World, accent_t: &mut f32, dt: f32, light_dt: f32| {
            world.step(dt);
            *accent_t += light_dt;
            effects.after_step(world, dt, settings.trails, settings.sparks);
            // Sound reads the same fresh-contact filter sparks do: an impact is pushed
            // with age 0 and has been aged by exactly this step's `dt`, so "new this
            // step" is testable without a device. And nothing here can reach the physics:
            // the mixer is downstream of the step, on the far side of a mutex.
            if settings.sound {
                if let Some(Ok(stream)) = audio_state.as_mut() {
                    let hits: Vec<(audio::Hit, u64)> = world
                        .impacts
                        .iter()
                        .enumerate()
                        .filter(|(_, hit)| hit.age <= dt * 1.001)
                        .map(|(i, hit)| (audio::hit_for(hit, sim::ROOM_HALF_X), i as u64))
                        .collect();
                    stream.note(hits);
                }
            }
            history.record(world, *accent_t);
        };
        if settings.paused {
            // Scrubbing: each queued request advances exactly one frame of
            // simulated time, lights and all. Anything still queued from a
            // released pause key is dropped rather than banked up.
            for _ in 0..*step_requests {
                advance(world, accent_t, FRAME_STEP, FRAME_STEP);
            }
        } else {
            advance(world, accent_t, raw_dt * settings.time_scale, raw_dt);
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

        let panels = [PANEL];

        // --- egui ------------------------------------------------------------
        // The HUD wants mutable access to the settings, the world and the
        // camera. The closure that holds that access dies with this statement,
        // so the borrows are re-taken by hand rather than letting the closure
        // move the destructured fields — which would leave nothing for the frame
        // that follows.
        let raw_input = egui_state.take_egui_input(window.as_ref());
        let count_before = settings.count;
        let hulls_before = settings.primitive_hulls;
        let mut req = HudRequests::default();
        let extra = HudExtra {
            history_len: history.frames.len(),
            history_at: history.position(),
            recording: match clip {
                Clip::Recording(recorder) => Some(recorder.seconds()),
                _ => None,
            },
            saving: matches!(clip, Clip::Saving(..)),
            dialog_open: dialog.is_some(),
        };
        let full_out = {
            let (settings, world, camera, readout, req, imports, extra) = (
                &mut *settings,
                &mut *world,
                &mut *camera,
                &*readout,
                &mut req,
                &*imports,
                &extra,
            );
            egui_ctx.run_ui(raw_input, move |ui| {
                hud(
                    ui,
                    settings,
                    world,
                    camera,
                    readout,
                    req,
                    &BODIES_RANGE,
                    imports,
                    extra,
                );
            })
        };
        // Anything that replaces or reorders the bodies invalidates what points
        // at them by index: the held body, and every trail.
        let mut bodies_replaced = false;
        // Reset replaces the bodies but leaves the tuned gravity and bounce
        // alone, so it has to happen out here where the whole world is reachable.
        if req.reset {
            App::reset(settings, world);
            bodies_replaced = true;
        }
        if settings.count != count_before {
            App::set_count(settings, world, settings.count);
            bodies_replaced = true;
        }
        if settings.primitive_hulls != hulls_before {
            App::set_primitive_hulls(world, shapes, settings.primitive_hulls);
        }
        // Rewinding. Both forms pause first: a room that is running would
        // simply play forward again from wherever it was put.
        let seek = match (req.scrub, req.step_back) {
            (Some(index), _) => Some(index),
            (None, true) => Some(history.position().saturating_sub(1)),
            (None, false) => None,
        };
        if let Some(index) = seek {
            settings.paused = true;
            if history.seek(index, world, accent_t) {
                bodies_replaced = true;
            }
        }
        if req.fullscreen {
            // `Borderless(None)` is the display's own native mode. On macOS winit
            // gives the window a full-screen space of its own rather than
            // covering the desktop; on Linux it is a request the compositor or
            // window manager carries out — which is what a viewer like this
            // wants either way. The
            // size change arrives as a `Resized` event, so `on_resize` reconfigures
            // the surface and `Gfx` rebuilds its attachments on the next frame.
            if window.fullscreen().is_some() {
                window.set_fullscreen(None);
            } else {
                window.set_fullscreen(Some(Fullscreen::Borderless(None)));
            }
        }
        if req.clear {
            App::clear_all(settings, world, shapes, &mut gpu.shape_mesh, imports);
            // Nothing asked for a body any more, so neither the queue nor the answers
            // still coming back. Dropping a receiver is the whole cancellation: the
            // worker's `send` then fails, and it already ignores that. It does not stop
            // the tool that is already running — `run_capped` polls a child process with
            // no cancellation hook — so a Blender started a second ago finishes its
            // conversion and throws the result away, holding its own CPU and temp
            // directory until it does. Stopping it would mean a flag the poll loop reads;
            // for a control this rare, discarding the answer is the cheaper honest half.
            converting.clear();
            // Queued but not yet started, which is the window a drop leaves open: the
            // drop handler runs between frames, and `x` is handled in the same draw as
            // the queue below, so a file dropped just before the press would otherwise
            // be imported into the room that has just been emptied.
            pending.clear();
            settings.import_converting = None;
            // The snapshots refer to shapes that have just been released.
            history.clear();
            if settings.primitive_hulls {
                App::set_primitive_hulls(world, shapes, true);
            }
            bodies_replaced = true;
        }
        if bodies_replaced {
            *grab = None;
            effects.clear();
            if let Some(Ok(stream)) = audio_state.as_mut() {
                // A hit from before a reset should not keep ringing after it, for the
                // same reason a trail may not streak from where a body was to where it
                // suddenly is.
                stream.clear();
            }
        }

        // The file dialog runs on a thread of its own and answers here. It is
        // created on this thread, which macOS requires, and waited for on that
        // one, so the room never stops drawing behind it.
        if req.open && dialog.is_none() {
            let (answer, receiver) = mpsc::channel();
            let picking = rfd::AsyncFileDialog::new()
                .set_title("Import an object")
                .add_filter("3D objects", &import::importable_extensions())
                .set_parent(window.as_ref())
                .pick_files();
            std::thread::spawn(move || {
                let picked = pollster::block_on(picking).unwrap_or_default();
                let _ = answer.send(picked.iter().map(|f| f.path().to_path_buf()).collect());
            });
            *dialog = Some(receiver);
        }
        if let Some(receiver) = dialog.as_ref() {
            match receiver.try_recv() {
                Ok(paths) => {
                    for path in paths {
                        if import::is_supported(&path) {
                            pending.push(path);
                        } else {
                            settings.import_note = import::unsupported_message(&path);
                        }
                    }
                    *dialog = None;
                }
                Err(mpsc::TryRecvError::Disconnected) => *dialog = None,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }

        // `--record`: press the button once at the start and once when the
        // clip is long enough, then leave when the file has been written.
        if auto_record.take().is_some_and(|seconds| {
            *auto_stop = Some(seconds);
            true
        }) {
            req.record = true;
        } else if let Some(limit) = *auto_stop {
            match clip {
                Clip::Recording(recorder) if recorder.seconds() >= limit => req.record = true,
                Clip::Idle => {
                    println!("{}", settings.record_note);
                    *quit = true;
                    *auto_stop = None;
                }
                _ => {}
            }
        }

        // Recording: the button starts a clip and stops it, and a clip also
        // stops itself when it has run as long as one is allowed to.
        let stop_clip = matches!(clip, Clip::Recording(recorder) if req.record || recorder.full());
        if stop_clip {
            if let Clip::Recording(recorder) = std::mem::replace(clip, Clip::Idle) {
                settings.record_note =
                    format!("saving {} frames…", recorder.frames());
                let (worker, path) = recorder.finish();
                *clip = Clip::Saving(worker, path);
            }
        } else if req.record && matches!(clip, Clip::Idle) {
            let size = window.inner_size();
            let (width, height) = record::clip_size(size.width, size.height);
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            match Recorder::start(record::default_clip_path(stamp), width, height) {
                Ok(recorder) => {
                    settings.record_note = format!("recording {width}x{height}");
                    *clip = Clip::Recording(recorder);
                }
                Err(e) => settings.record_note = format!("✗ {e}"),
            }
        }
        if matches!(clip, Clip::Saving(worker, _) if worker.is_finished()) {
            if let Clip::Saving(worker, path) = std::mem::replace(clip, Clip::Idle) {
                settings.record_note = match worker.join() {
                    Ok(Ok(())) => format!("✓ saved {}", path.display()),
                    Ok(Err(e)) => format!("✗ {e}"),
                    Err(_) => "✗ the encoder stopped unexpectedly".to_owned(),
                };
            }
        }
        // Imports are applied here rather than in the panel because registering
        // one needs the device, the shape table, the colour list and the world at
        // the same time, which is exactly what the borrow-splitting above forbids.
        // Anything still queued when there was no device stays queued: a
        // Answers first, and only those: nothing here waits. Before the queue, so the
        // frame a conversion lands in is the frame the next queued file starts — the
        // other order costs one idle frame per queued file, and changes nothing else.
        for result in App::take_finished(converting) {
            match result {
                Ok(import) => {
                    App::add_import(settings, world, shapes, gpu, imports, import);
                }
                Err(e) => settings.import_note = format!("✗ {e}"),
            }
        }

        // `--import` on the command line lands on the first frame that has one.
        let mut queued = std::mem::take(pending);
        queued.extend(req.imports.iter().cloned());
        let mut waiting = Vec::new();
        for path in queued {
            if import::needs_converter(&path) {
                // One conversion at a time, the rest left in `pending` in the order they
                // arrived. Blender's start-up dominates the cost of a conversion, so ten
                // running at once make each one slower than if they had queued, hold ten
                // scenes' worth of memory for the results, and can each run to the 120 s
                // ceiling — which is what dragging a fistful of `.fbx` files off Finder
                // would otherwise do.
                if !converting.is_empty() {
                    waiting.push(path);
                    continue;
                }
                // Off the frame thread, answered on a later one. The message is
                // written when the work starts, not when it lands: a conversion
                // that takes ten seconds has to say what it is busy with, or it is
                // indistinguishable from the window having frozen.
                let (answer, receiver) = mpsc::channel();
                let name = path.display().to_string();
                settings.import_note = format!(
                    "converting {name} — this format has no reader in the app, so an \
                     external tool has to look at it and that takes a few seconds"
                );
                std::thread::spawn(move || {
                    let _ = answer.send(import::import_path(&path).map_err(|e| e.to_string()));
                });
                // The name travels with the receiver: a failure has to say which file
                // failed once more than one is in the queue, and the panel has to say
                // what is busy right now without depending on a note nothing owns.
                converting.push((name, receiver));
                continue;
            }
            App::import_one(settings, world, shapes, gpu, imports, &path);
        }
        // Deferred work goes back where the next frame will find it, in order.
        pending.extend(waiting);
        // Rebuilt every frame rather than appended to `import_note`: that is one slot,
        // and whatever else wrote it — a successful in-process import, a refusal — would
        // end up carrying a conversion's queue count as if it belonged to it. This line
        // is the panel admitting the room was asked for more than one thing at a time,
        // for as long as that stays true.
        settings.import_converting = converting
            .first()
            .map(|(name, _)| (name.clone(), pending.len()));
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
        // Built here rather than before the HUD ran, so that a reset, a rewind
        // or a cleared room is what this frame shows.
        let lights = App::lights_at(*accent_t);
        // Real shadows and the blobs that stand in for them are alternatives.
        let blobs = if settings.shadows {
            Vec::new()
        } else {
            App::blobs_for(world, &lights)
        };
        let rings = App::rings_for(world);
        let lines = effects.lines(world);
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
            shadows: settings.shadows,
            lines,
            outlined: grab.map(|g| g.index),
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

        // A clip frame, when one is due: the same scene again at the clip's own
        // size, with no panel. It has to follow the submit above — both renders
        // stage into the same buffers, and a buffer is read when its commands
        // are submitted.
        if let Clip::Recording(recorder) = clip {
            if recorder.due(raw_dt) {
                let (clip_width, clip_height) = recorder.size();
                let clip_scene = Scene {
                    proj: camera.projection(clip_width, clip_height),
                    ..scene
                };
                match gpu.gfx.render_to_pixels(&clip_scene, clip_width, clip_height) {
                    Ok(pixels) => recorder.push(pixels),
                    Err(e) => settings.record_note = format!("✗ {e}"),
                }
            }
        }

        App::drain_frees(gpu, &mut textures_delta);
        egui_state.handle_platform_output(window.as_ref(), platform_output);
    }
}

/// The HUD's body-count slider range, the browser build's `0..=40` stretched to
/// the port's own ceiling. Zero is a real setting rather than an edge case: an
/// empty room is how you check the room itself, and both `settle` and
/// `clamp_to_room` walk an empty `bodies` without incident.
const BODIES_RANGE: std::ops::RangeInclusive<usize> = 0..=4000;

/// What the HUD wants done but cannot do from inside its own borrow: anything
/// that reaches past the world it is handed, such as re-seeding and rebuilding
/// it at once. Collected in one struct so that a new request costs a field
/// rather than another parameter on [`hud`].
#[derive(Clone, Debug, Default)]
struct HudRequests {
    reset: bool,
    steps: u32,
    /// Flip full screen. Asked for rather than done because the window belongs to
    /// `App`, not to the panel, and winit wants the call on the event-loop thread.
    fullscreen: bool,
    /// Empty the room of every body, imports included. Requested for the same
    /// reason as `reset`: it has to release imported shapes from the per-shape
    /// lists as well as the world.
    clear: bool,
    /// Files to read. The panel cannot do this itself: an import needs the device
    /// to upload a mesh, the shape table, the colour list and the world at once.
    imports: Vec<PathBuf>,
    /// Put the room back to this frame of the history.
    scrub: Option<usize>,
    /// Go back one frame from wherever the room is.
    step_back: bool,
    /// Bring up the file dialog. Requested because the dialog wants the window.
    open: bool,
    /// Start a clip, or stop the one that is running.
    record: bool,
}

/// What the panel shows but does not own: state that lives on `App` and only
/// needs reading here. One struct, so a new readout is a field and not another
/// parameter on [`hud`].
#[derive(Clone, Debug, Default)]
struct HudExtra {
    /// Frames of history available to rewind through, and which one is showing.
    history_len: usize,
    history_at: usize,
    /// Seconds of clip recorded so far, while one is being recorded.
    recording: Option<f32>,
    /// A finished clip is still being written.
    saving: bool,
    dialog_open: bool,
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
    imports: &[ImportedShape],
    extra: &HudExtra,
) {
    // Shortcuts, read through egui so that a widget with keyboard focus keeps
    // its keys — the rule the browser build enforced by checking whether the
    // event target was an input element. The seed box is that widget now, so a
    // typed `.` or a typed `0x` stays in the box.
    if ui.memory(|m| m.focused().is_none()) {
        let (space, k, r, w, period, h, f, c, x, comma, o, v, s) = ui.input_mut(|i| {
            (
                i.consume_key(egui::Modifiers::NONE, egui::Key::Space),
                i.consume_key(egui::Modifiers::NONE, egui::Key::K),
                i.consume_key(egui::Modifiers::NONE, egui::Key::R),
                i.consume_key(egui::Modifiers::NONE, egui::Key::W),
                i.consume_key(egui::Modifiers::NONE, egui::Key::Period),
                i.consume_key(egui::Modifiers::NONE, egui::Key::H),
                i.consume_key(egui::Modifiers::NONE, egui::Key::F),
                i.consume_key(egui::Modifiers::NONE, egui::Key::C),
                i.consume_key(egui::Modifiers::NONE, egui::Key::X),
                i.consume_key(egui::Modifiers::NONE, egui::Key::Comma),
                i.consume_key(egui::Modifiers::NONE, egui::Key::O),
                i.consume_key(egui::Modifiers::NONE, egui::Key::V),
                i.consume_key(egui::Modifiers::NONE, egui::Key::S),
            )
        });
        // `,` is the other half of `.`: one frame back instead of one forward.
        if comma {
            req.step_back = true;
        }
        if o {
            req.open = true;
        }
        if v {
            req.record = true;
        }
        if s {
            settings.sound = !settings.sound;
        }
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
        // Clearing is requested rather than done: it has to release the imported
        // shapes from the per-shape lists, which the panel is not holding.
        if x {
            req.clear = true;
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
                ui.label("space · k · r · w · . step · , back");
                ui.label("h hud · f fullscreen · c recenters");
                ui.label("o open · v record");
                ui.label("click a body to hold it, let go to throw; the wheel pushes and pulls");
                ui.separator();

                // Rewind. The slider's right-hand end is the present; dragging
                // it left pauses the room and puts it back to that frame, and
                // resuming from there discards what had followed.
                if extra.history_len > 1 {
                    let last = extra.history_len - 1;
                    let mut at = extra.history_at.min(last);
                    let behind = last - at;
                    let moved = ui
                        .add(
                            egui::Slider::new(&mut at, 0..=last)
                                .show_value(false)
                                .text(if behind == 0 {
                                    "rewind".to_owned()
                                } else {
                                    format!("{behind} frames back")
                                }),
                        )
                        .on_hover_text(", steps back one frame · resuming plays on from here")
                        .changed();
                    if moved {
                        req.scrub = Some(at);
                    }
                }

                ui.horizontal_wrapped(|ui| {
                    let label = if extra.recording.is_some() { "Stop" } else { "Record" };
                    if ui
                        .add_enabled(!extra.saving, egui::Button::new(label))
                        .on_hover_text("v — an animated GIF of the room, without this panel")
                        .clicked()
                    {
                        req.record = true;
                    }
                    if let Some(seconds) = extra.recording {
                        ui.colored_label(
                            ui.visuals().error_fg_color,
                            format!("● {seconds:.1} s of {:.0}", record::MAX_SECONDS),
                        );
                    }
                });
                if !settings.record_note.is_empty() {
                    if settings.record_note.starts_with('✗') {
                        ui.colored_label(ui.visuals().error_fg_color, settings.record_note.as_str());
                    } else {
                        ui.label(settings.record_note.as_str());
                    }
                }
                ui.separator();

                ui.horizontal_wrapped(|ui| {
                    if ui.button("Spawn one").clicked() {
                        let opts = if settings.spawn_random {
                            SpawnOpts::default()
                        } else {
                            SpawnOpts::with_scale(settings.spawn_size)
                        };
                        world.record(settings.spawn_shape, opts);
                        // The slider governs the generated population, which once
                        // anything has been imported is no longer the length of the
                        // body list.
                        settings.count = App::generated_count(world);
                    }
                    if ui
                        .button("Clear all")
                        .on_hover_text("x — every body out; unlike `r` this drops imported objects too")
                        .clicked()
                    {
                        req.clear = true;
                    }
                });
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
                ui.add(egui::Slider::new(&mut settings.count, bodies.clone()).logarithmic(true))
                    .on_hover_text("0 leaves an empty room; past 60 they shrink to fit");
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
                ui.checkbox(&mut world.hulls, "hull colliders")
                    .on_hover_text("imported objects collide as their convex hull, not a sphere");
                ui.checkbox(&mut settings.primitive_hulls, "…for the faceted primitives too")
                    .on_hover_text("cubes land flat; off is the browser build's all-spheres model");
                ui.checkbox(&mut settings.wireframe, "wireframe");
                ui.checkbox(&mut settings.shadows, "shadows")
                    .on_hover_text("cast from the ceiling light; off uses the ground blobs");
                ui.checkbox(&mut settings.sparks, "sparks");
                ui.checkbox(&mut settings.trails, "trails");
                ui.checkbox(&mut settings.sound, "sound").on_hover_text(
                    "hits are synthesised from the contact: harder is louder, smaller is \
                     higher, and the pitch ladder keeps a crowd consonant",
                );
                if let Some(why) = settings.audio_note.as_ref() {
                    // A machine with no output device keeps simulating exactly as it
                    // would with one; the panel says why the silence is not a mute.
                    ui.label(format!("no audio: {why}"));
                } else if settings.sound {
                    ui.add(
                        egui::Slider::new(&mut settings.volume, 0.0..=1.0)
                            .text("volume")
                            .show_value(false),
                    );
                }
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

                ui.separator();
                if imports.is_empty() {
                    ui.label("nothing imported");
                } else {
                    // Listed by name, so it is visible which bodies came off disk
                    // and which file the note under them is about.
                    for imported in imports {
                        ui.label(format!(
                            "{} · {} triangles",
                            imported.source, imported.triangles
                        ));
                    }
                }
                ui.label(format!(
                    "import {} — drag a file onto the window, or type its path",
                    import::supported_hint()
                ));
                ui.horizontal_wrapped(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut settings.import_draft).desired_width(190.0),
                    );
                    if ui.button("Import").clicked() {
                        let draft = settings.import_draft.trim().to_owned();
                        if !draft.is_empty() {
                            req.imports.push(PathBuf::from(draft));
                        }
                    }
                    if ui
                        .add_enabled(!extra.dialog_open, egui::Button::new("Open…"))
                        .on_hover_text("o — choose files in the system's own dialog")
                        .clicked()
                    {
                        req.open = true;
                    }
                });
                if let Some(hovered) = settings.hovered_file.as_deref() {
                    ui.label(format!("release to import {hovered}"));
                }
                if !settings.import_note.is_empty() {
                    // A failure gets the error colour; a success is plain text
                    // carrying the counts and every repair the reader made.
                    if settings.import_note.starts_with('✗') {
                        ui.colored_label(
                            ui.visuals().error_fg_color,
                            settings.import_note.as_str(),
                        );
                    } else {
                        ui.label(settings.import_note.as_str());
                    }
                }
                // Its own line, for as long as the queue says so — including the frames
                // after something else wrote the note above.
                if let Some((running, queued)) = settings.import_converting.as_ref() {
                    let waiting = if *queued > 0 {
                        format!(" · {queued} more queued")
                    } else {
                        String::new()
                    };
                    ui.label(format!("converting {running}{waiting}"));
                }
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
            import_draft: String::new(),
            import_note: String::new(),
            import_converting: None,
            hovered_file: None,
            shadows: true,
            trails: false,
            sparks: true,
            sound: false,
            volume: 0.7,
            audio_note: None,
            primitive_hulls: false,
            record_note: String::new(),
        }
    }

    /// A stand-in for a registered import. The panel lists these, and the layout
    /// test wants that row rendered too rather than only its empty state.
    fn sample_imports() -> Vec<ImportedShape> {
        vec![ImportedShape {
            source: "teapot".to_owned(),
            shape_index: geom::PRIMITIVE_NAMES.len(),
            triangles: 1_234,
        }]
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
    fn a_conversion_answers_when_it_is_ready_without_anyone_waiting() {
        // The frame loop's side of the converter: a poll must never block, answers land
        // in the order the files were asked for rather than the order the tools finished,
        // and a worker that died has to surface as a message instead of leaving the panel
        // promising a conversion for the rest of the session.
        let (done_answer, done) = mpsc::channel();
        let (slow_answer, slow) = mpsc::channel::<Result<Import, String>>();
        let (dead_answer, dead) = mpsc::channel::<Result<Import, String>>();
        drop(dead_answer);
        let mut converting = vec![
            ("a.fbx".to_owned(), done),
            ("b.fbx".to_owned(), slow),
            ("c.fbx".to_owned(), dead),
        ];

        done_answer
            .send(Ok(Import {
                mesh: geom::quad_mesh(),
                source: "first".to_owned(),
                notes: Vec::new(),
                dropped_triangles: 0,
                hull: None,
                gloss: None,
            }))
            .expect("the receiver is still there");
        // `slow_answer` deliberately holds nothing yet: that receiver stands for a
        // conversion still running when the frame polls.

        // Answers come back in the order the files were queued, not the order the tools
        // finished: the import palette, the shape indices and the note the user reads all
        // follow that order. The dead worker reports on this same poll — it has nothing
        // left to wait for — while the in-flight one keeps its receiver.
        let first = App::take_finished(&mut converting);
        assert_eq!(first.len(), 2, "the answered one and the dead one: {first:?}");
        assert_eq!(converting.len(), 1, "the one that has not answered is left alone");
        assert!(matches!(first[0], Ok(_)), "the file queued first is applied first");
        assert_eq!(
            first[1].as_ref().err().map(String::as_str),
            Some("c.fbx: the conversion thread ended without an answer"),
            "a worker that died has to be reported, not left promising a conversion"
        );

        // A second poll of the same frame finds nothing new and, above all, does not
        // wait: the remaining conversion is still in flight.
        assert!(
            App::take_finished(&mut converting).is_empty(),
            "an in-flight conversion is not an error, and not a reason to block"
        );

        // Then it answers, on whatever frame happens to be polling.
        slow_answer
            .send(Err("second gave up".to_owned()))
            .expect("the receiver is still there");
        let later = App::take_finished(&mut converting);
        assert_eq!(
            later.first()
                .and_then(|r| r.as_ref().err())
                .map(String::as_str),
            Some("second gave up"),
            "a conversion that failed is an answer, not a silence"
        );
        assert!(converting.is_empty(), "and its receiver is gone");
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
        let imports = sample_imports();
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
            // The same argument for the import row: a multi-line truncation note,
            // a hover label that only exists mid-drag, and a path wider than the
            // panel all have to survive the narrow window.
            if frame == 1 {
                settings.import_note = format!(
                    "✓ big: 21844 triangles, {} vertices — 8156 dropped by the index \
                     limit\nwireframe edges capped at 60000 segments",
                    import::MAX_VERTS
                );
                settings.hovered_file = Some("teapot.fbx".to_owned());
                settings.import_draft =
                    "/Volumes/assets/a/path/longer/than/the/panel/teapot.obj".to_owned();
            }
            let mut out = ctx.run_ui(input, |ui| {
                hud(
                    ui,
                    &mut settings,
                    &mut world,
                    &mut camera,
                    "paused · 60 fps · 5 bodies",
                    &mut req,
                    &BODIES_RANGE,
                    &imports,
                    // With history and a clip running, so the rewind slider
                    // and the recording row are laid out too.
                    &HudExtra {
                        history_len: 120,
                        history_at: 80,
                        recording: Some(1.5),
                        saving: false,
                        dialog_open: false,
                    },
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
            imports: &[ImportedShape],
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
                    imports,
                    &HudExtra::default(),
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
        let hidden = frame(
            &ctx,
            &mut settings,
            &mut world,
            &mut camera,
            &mut req,
            &[],
            vec![],
        );
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
            &[],
            vec![key(true), key(false)],
        );
        assert!(!settings.hud_hidden, "`h` must bring the panel back");

        let shown = frame(
            &ctx,
            &mut settings,
            &mut world,
            &mut camera,
            &mut req,
            &[],
            vec![],
        );
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

    /// A body as `add_import` would put one in: a shape index past the built-ins,
    /// a collider equal to its own bounding sphere, at rest.
    fn imported_body(shape_index: usize, scale: f32) -> sim::Body {
        sim::Body {
            shape: shape_index,
            radius: scale * IMPORT_RADIUS_FACTOR,
            scale,
            mass: scale * scale * scale * 4.0 + 0.2,
            pos: glam::Vec3::new(0.0, 1.0, 0.0),
            vel: glam::Vec3::ZERO,
            rot: glam::Mat4::IDENTITY,
            ang_vel: glam::Vec3::ZERO,
        }
    }

    #[test]
    fn the_count_slider_and_reset_manage_generated_bodies_not_imports() {
        // The rule the panel implies: the slider and `r` are about the generated
        // population. Without it, dragging the count down would delete an object
        // that cost a trip off disk, in an app whose whole physics is a list.
        let mut settings = test_settings();
        let mut world = World::default();
        App::reset(&mut settings, &mut world);
        assert_eq!(world.bodies.len(), settings.count);

        let imported_index = geom::PRIMITIVE_NAMES.len();
        world.bodies.push(imported_body(imported_index, 0.6));
        assert_eq!(
            App::generated_count(&world),
            settings.count,
            "the import is not part of the count"
        );

        App::set_count(&mut settings, &mut world, 2);
        assert_eq!(App::generated_count(&world), 2, "the slider shrank");
        assert_eq!(world.bodies.len(), 3, "and the import survived");
        assert!(
            world.bodies.iter().any(|b| b.shape == imported_index),
            "clearing to two must remove generated bodies, not the newest one"
        );

        App::reset(&mut settings, &mut world);
        assert_eq!(
            App::generated_count(&world),
            settings.count,
            "reset rebuilt the generated set from the seed"
        );
        assert!(
            world.bodies.iter().any(|b| b.shape == imported_index),
            "reset leaves imports where they are; Clear all is the control that does not"
        );
    }

    #[test]
    fn clear_all_empties_the_room_and_releases_imported_shapes() {
        let mut settings = test_settings();
        settings.count = 4;
        let mut world = World::default();
        App::reset(&mut settings, &mut world);
        let imported_index = geom::PRIMITIVE_NAMES.len();
        world.bodies.push(imported_body(imported_index, 0.5));
        world.colors.push([0.5, 0.5, 0.5]);

        let mut shapes = geom::all_primitives();
        shapes.push(ShapeDef {
            mesh: geom::quad_mesh(),
            color: [0.5, 0.5, 0.5],
            gloss: import::DEFAULT_GLOSS,
        });
        let mut handles: Vec<usize> = (0..shapes.len()).collect();
        let mut imports = sample_imports();

        App::clear_all(&mut settings, &mut world, &mut shapes, &mut handles, &mut imports);

        assert!(world.bodies.is_empty(), "every body is gone");
        assert!(world.impacts.is_empty(), "and every ring with them");
        let keep = geom::PRIMITIVE_NAMES.len();
        assert_eq!(shapes.len(), keep, "imported shapes released");
        assert_eq!(handles.len(), keep, "and their mesh handles with them");
        assert_eq!(world.colors.len(), keep, "and the ring palette");
        assert!(imports.is_empty(), "the panel list is empty");
        assert_eq!(settings.count, 0, "the slider matches the empty room");

        // The three lists are addressed by `Body::shape`, so they have to stay the
        // same length as each other for the next spawn to be sound.
        assert_eq!(shapes.len(), handles.len());
        assert_eq!(shapes.len(), world.colors.len());

        // And the room is usable again afterwards.
        App::set_count(&mut settings, &mut world, 3);
        assert_eq!(world.bodies.len(), 3);
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

    #[test]
    fn rewinding_restores_an_earlier_frame_and_resuming_drops_the_future() {
        let mut world = World::default();
        world.set_seed(DEFAULT_SEED);
        for i in 0..6 {
            world.record(i, SpawnOpts::default());
        }
        let mut history = History::default();
        let mut accent = 0.0f32;
        history.record(&world, accent);
        let mut positions = vec![world.bodies[0].pos];
        for _ in 0..30 {
            world.step(FRAME_STEP);
            accent += FRAME_STEP;
            history.record(&world, accent);
            positions.push(world.bodies[0].pos);
        }
        assert_eq!(history.frames.len(), 31);
        assert_eq!(history.position(), 30);

        assert!(history.seek(10, &mut world, &mut accent));
        assert_eq!(world.bodies[0].pos, positions[10], "frame 10 is frame 10 again");
        assert!((accent - 10.0 * FRAME_STEP).abs() < 1e-5, "and the lights went back with it");
        assert_eq!(history.position(), 10);
        assert_eq!(history.frames.len(), 31, "scrubbing alone discards nothing");

        // Playing on from frame 10 replays exactly what happened before — the
        // generator's state is part of the snapshot — and the old frames 11..30
        // are replaced rather than kept as a parallel future.
        world.step(FRAME_STEP);
        accent += FRAME_STEP;
        history.record(&world, accent);
        assert_eq!(history.frames.len(), 12);
        assert_eq!(world.bodies[0].pos, positions[11]);
        assert!(!history.seek(99, &mut world, &mut accent));
    }

    #[test]
    fn history_stays_inside_its_frame_budget() {
        let mut world = World::default();
        world.spawn(0, SpawnOpts::default());
        let mut history = History::default();
        for _ in 0..HISTORY_MAX_FRAMES + 50 {
            history.record(&world, 0.0);
        }
        assert_eq!(history.frames.len(), HISTORY_MAX_FRAMES);
        let counted: usize = history.frames.iter().map(|(s, _)| s.bytes()).sum();
        assert_eq!(history.bytes, counted, "the running byte count has not drifted");
    }

    #[test]
    fn a_crowd_is_spawned_smaller_and_still_fits_the_room() {
        let mut settings = test_settings();
        let mut world = World::default();
        world.set_seed(DEFAULT_SEED);
        App::set_count(&mut settings, &mut world, 1500);
        assert_eq!(world.bodies.len(), 1500);
        let largest = world.bodies.iter().map(|b| b.scale).fold(0.0f32, f32::max);
        assert!(largest < 0.3, "a crowd of 1500 is made of small bodies, got {largest}");
        for b in &world.bodies {
            assert!(b.pos.y >= b.radius - 1e-3 && b.pos.y <= sim::ROOM_HALF_Y - b.radius + 1e-3);
        }
        // And it can be simulated: a second of it leaves everything in the room.
        for _ in 0..60 {
            world.step(FRAME_STEP);
        }
        for b in &world.bodies {
            assert!(b.pos.is_finite());
            assert!(b.pos.x.abs() <= sim::ROOM_HALF_X + 0.05, "{:?}", b.pos);
            assert!(b.pos.y >= -0.05 && b.pos.y <= sim::ROOM_HALF_Y + 0.05, "{:?}", b.pos);
        }
        // Shrinking back to a room the browser build could show still works.
        App::set_count(&mut settings, &mut world, 9);
        assert_eq!(world.bodies.len(), 9);
    }

    #[test]
    fn the_faceted_primitives_get_hulls_only_when_asked() {
        let shapes = geom::all_primitives();
        let mut world = World::default();
        App::set_primitive_hulls(&mut world, &shapes, true);
        for i in HULL_PRIMITIVES {
            assert!(world.colliders[i].is_some(), "{} has no hull", geom::PRIMITIVE_NAMES[i]);
        }
        // sphere, capsule, torus
        for i in [0, 2, 3] {
            assert!(world.colliders[i].is_none(), "{} should stay a sphere", geom::PRIMITIVE_NAMES[i]);
        }
        let cube = world.colliders[1].as_ref().unwrap();
        assert_eq!((cube.verts.len(), cube.planes.len()), (8, 6));
        App::set_primitive_hulls(&mut world, &shapes, false);
        assert!(world.colliders.iter().all(|c| c.is_none()));
    }
}
