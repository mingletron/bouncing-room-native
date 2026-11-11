//! Offscreen render probe: the native equivalent of the browser build's
//! `bouncing-room/scripts/check.mjs` visual check.
//!
//! It builds a real scene — seeded physics run forward a couple of seconds, the
//! three lights, shadow blobs, the ceiling panel and the impact rings — renders
//! it headlessly into a texture, writes a PNG and then *reads its own picture
//! back*: every body's centre is projected into the frame and compared with the
//! same pixel from an otherwise identical empty room. A body that is invisible
//! means the room's interior-culling trick is wound the wrong way round, which is
//! exactly the one thing in this port that cannot be decided by reading the
//! contract: WebGPU's framebuffer convention inverts GL's, so `FrontFace::Cw`
//! plus `cull_mode: Front` is a guess until a picture says otherwise.
//!
//! With `--both`, it renders the scene twice, once per winding, and requires
//! precisely one of them to pass. Passing on both would mean the room is not
//! being culled at all, and passing on neither would mean the bodies are not
//! being drawn — neither is a correct port.
//!
//! Run with `--help` for the full option list; `--out probe.png --seconds 2.2`
//! is the interesting default.

use std::path::PathBuf;

use bouncing_room_native::gfx::{Disc, Gfx, GfxOptions, RenderTarget, Ring, RoomDraw, Scene};
use bouncing_room_native::gpu_types::Light;
use bouncing_room_native::sim::{SpawnOpts, World};
use bouncing_room_native::{OrbitCamera, geom, import, sim};

use glam::{Vec3, Vec4};

const USAGE: &str = "\
probe [options]

  --out PATH         PNG to write (default: probe.png)
  --width N          frame width in pixels (default: 900)
  --height N         frame height in pixels (default: 600)
  --count N          bodies in the room (default: 14)
  --seed N           physics seed, so the picture is reproducible (default: 0x5EED)
  --seconds S        simulated seconds before the shutter (default: 2.2)
  --front-face V     only render this room winding: cw | ccw | both (default: both)
  --keep-rings       draw impact rings (on by default; --no-rings to drop)
  --import PATH      put an imported object (.obj or .stl) in the scene and
                     require it to show up too; repeatable
  --wireframe    draw the edge overlay as well as the solids (the visibility
                     check reads solid pixels either way)
  --hud              also render the egui overlay pass and check it composited
  --help             this text

The exit status is 0 only when exactly one winding shows every body.";

/// How far apart a body's pixel and the empty-room pixel have to be to count as
/// a change. Room surfaces are dark and smooth (walls ~0.2, floor ~0.13), bodies
/// are saturated, and the lit result varies over a wide range, so this is
/// deliberately loose.
const PIXEL_EPS: u8 = 26;

/// The share of a body's screen-space disc that has to have changed for the body
/// to count as drawn. Not 100%, because the disc is the body's *bounding* circle:
/// a cone seen edge-on fills maybe a third of it and a flat torus less. The wrong
/// winding leaves nothing at all, so 0 — the gap between a wrong answer and a
/// right one is enormous, and this threshold only has to clear the shapes.
const DISC_COVER: f32 = 0.12;

/// Below this many in-frame samples, the body is a smudge and says nothing either
/// way, so it is not counted.
const MIN_SAMPLES: usize = 24;

/// The share of judgeable bodies that have to be visible for a winding to pass.
const VISIBILITY_PASS: f32 = 0.85;

/// Width of the left strip used to check that the HUD composited over the scene.
/// The control panel is a little narrower than this, and it fills its area with
/// an opaque background, so a composited HUD repaints essentially all of it.
const HUD_PANEL_WIDTH: u32 = 240;

/// The share of that strip that has to change for the overlay to count as drawn.
const HUD_COVER: f32 = 0.5;

/// What the room is cleared to where nothing is drawn (from the WebGL build's
/// `scene.background`).
const BACKGROUND: [f32; 3] = [0.052, 0.058, 0.072];

/// The room's surface colours and grid pitch, as tuned in the browser build.
const WALL: [f32; 3] = [0.2, 0.22, 0.25];
const FLOOR: [f32; 3] = [0.13, 0.14, 0.16];
const CEIL: [f32; 3] = [0.16, 0.17, 0.19];
const EDGE: [f32; 4] = [0.55, 0.62, 0.72, 0.4];
const GRID_SPACING: f32 = 1.2;

/// Shadow-blob colour and the ceiling fixture, again from the browser build.
const BLOB_COLOR: [f32; 3] = [0.015, 0.017, 0.022];
const PANEL_COLOR: [f32; 3] = [1.0, 0.94, 0.82];
const PANEL_ALPHA: f32 = 0.42;
const PANEL_INNER: f32 = 0.45;
const BLOB_INNER: f32 = 0.15;

struct Args {
    out: PathBuf,
    width: u32,
    height: u32,
    count: usize,
    seed: u64,
    seconds: f32,
    /// Which room windings to render.
    front_faces: Vec<wgpu::FrontFace>,
    rings: bool,
    hud: bool,
    /// The edge overlay on every mesh, which is how a screenshot can show an
    /// import that had whole faces sampled out of it.
    wireframe: bool,
    /// Files to import into the scene, each of which then has to pass the same
    /// visibility check as a generated primitive.
    imports: Vec<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        out: PathBuf::from("probe.png"),
        width: 900,
        height: 600,
        count: 14,
        seed: 0x5EED,
        seconds: 2.2,
        front_faces: vec![wgpu::FrontFace::Cw, wgpu::FrontFace::Ccw],
        rings: true,
        hud: false,
        wireframe: false,
        imports: Vec::new(),
    };

    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        // Value-taking flags read their argument out of `argv` so that
        // `--width=900` and `--width 900` both work.
        let mut value = |name: &str| -> Result<String, String> {
            argv.next()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match flag.as_str() {
            "--out" => args.out = PathBuf::from(value("--out")?),
            "--width" => args.width = value("--width")?.parse().map_err(|_| "--width: N")?,
            "--height" => args.height = value("--height")?.parse().map_err(|_| "--height: N")?,
            "--count" => args.count = value("--count")?.parse().map_err(|_| "--count: N")?,
            "--seed" => {
                let text = value("--seed")?;
                // Accept both `1234` and `0x5EED`, since the default is written
                // in hex.
                args.seed = match text.strip_prefix("0x").or(text.strip_prefix("0X")) {
                    Some(hex) => u64::from_str_radix(hex, 16),
                    None => text.parse(),
                }
                .map_err(|_| "--seed: N or 0x…")?;
            }
            "--seconds" => args.seconds = value("--seconds")?.parse().map_err(|_| "--seconds: S")?,
            "--front-face" => match value("--front-face")?.as_str() {
                "cw" => args.front_faces = vec![wgpu::FrontFace::Cw],
                "ccw" => args.front_faces = vec![wgpu::FrontFace::Ccw],
                "both" => args.front_faces = vec![wgpu::FrontFace::Cw, wgpu::FrontFace::Ccw],
                other => return Err(format!("--front-face: cw | ccw | both, not {other}")),
            },
            "--import" => args.imports.push(PathBuf::from(value("--import")?)),
            "--wireframe" => args.wireframe = true,
            "--hud" => args.hud = true,
            "--no-rings" => args.rings = false,
            "--keep-rings" => args.rings = true,
            "--help" | "-h" => return Err(String::new()),
            other => return Err(format!("unknown option {other}")),
        }
    }
    if args.width == 0 || args.height == 0 {
        return Err("width and height must be at least 1".into());
    }
    Ok(args)
}

/// A frame read back from the GPU: RGBA8, rows unpadded, plus the extent.
struct Frame {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

impl Frame {
    /// The RGBA8 pixel at (x, y), clamped to the frame.
    fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        let i = ((y * self.width + x) as usize) * 4;
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            if !message.is_empty() {
                eprintln!("probe: {message}");
            }
            eprintln!("{USAGE}");
            std::process::exit(if message.is_empty() { 0 } else { 2 });
        }
    };

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        // No display handle: this probe never opens a window, and wgpu picks
        // the platform's backend from the defaults here: Metal on macOS, Vulkan
        // on Linux.
        display: None,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: None,
        ..Default::default()
    }))
    .expect("probe: no GPU adapter (does this machine have Metal, Vulkan or D3D12?)");
    println!(
        "probe: adapter {} ({:?})",
        adapter.get_info().name,
        adapter.get_info().backend
    );
    if args.wireframe {
        // Worth saying out loud, because it changes what a pass means: the check
        // asks whether a body's pixels differ from the reference render, and an
        // edge overlay adds pixels of its own. The room's own outline is drawn on
        // both sides so it cancels, but a mesh's edges are not, so with this flag a
        // body can register from its wireframe alone rather than from a surface.
        // The winding verdict is only as strong as a solid run.
        println!(
            "probe: wireframe on — a body may count as visible from its edges, not \
             only its surfaces; run without this flag for the strict verdict"
        );
    }

    // Imported geometry goes through the same reader the app uses, which is what
    // makes a passing probe say "this mesh is drawn and visible" rather than
    // merely "it parsed".
    let imports: Vec<import::Import> = args
        .imports
        .iter()
        .map(|path| {
            import::import_path(path).unwrap_or_else(|e| panic!("probe: --import {e}"))
        })
        .collect();
    for import in imports.iter() {
        println!(
            "probe: read {} — {} triangles, {} vertices{}",
            import.source,
            import.mesh.triangle_count(),
            import.mesh.vertex_count(),
            if import.notes.is_empty() {
                String::new()
            } else {
                format!(" — {}", import.notes.join("; "))
            }
        );
    }
    let mut shapes = geom::all_primitives();
    // The probe keeps its own colour for an import rather than reading the app's,
    // for the same reason it keeps its own lights: a shared constant that is wrong
    // cancels out in a check that uses it.
    for import in imports.iter() {
        shapes.push(geom::ShapeDef {
            mesh: import.mesh.clone(),
            color: [0.78, 0.8, 0.85],
            gloss: 0.35,
        });
    }
    let mut results = Vec::new();
    let mut hud_coverage = Vec::new();

    for front_face in args.front_faces {
        let mut gfx = Gfx::new(
            &instance,
            None,
            GfxOptions {
                room_front_face: front_face,
            },
        )
        .unwrap_or_else(|e| panic!("probe: {e}"));

        // Meshes belong to the device, so they are uploaded once per
        // `Gfx` — a probe run is short enough that the duplication does not
        // matter, and it keeps each winding's render independent.
        let shape_mesh: Vec<usize> = shapes
            .iter()
            .map(|s| gfx.upload_mesh(&s.mesh))
            .collect();
        let room_mesh = gfx.upload_mesh(&geom::room_mesh(
            sim::ROOM_HALF_X,
            sim::ROOM_HALF_Y,
            sim::ROOM_HALF_Z,
        ));
        let quad_mesh = gfx.upload_mesh(&geom::quad_mesh());
        let ring_mesh = gfx.upload_mesh(&geom::ring_mesh(44));

        // Two worlds from the same seed: identical physics, one with bodies and
        // one without. The second is the reference the first is compared
        // against, so a body that any wall swallowed shows up as "this pixel
        // never changed".
        let mut world = seeded_world(args.seed, args.count);
        // Imported bodies join the world under test but never the empty reference,
        // so an imported mesh that fails to draw fails for the same reason and in
        // the same terms as a body that a wall swallowed: "this pixel never
        // changed". Their shape indices continue past the built-ins exactly as the
        // app hands them out, which is what the per-shape lists are indexed by.
        for i in 0..imports.len() {
            let scale = 0.9 + 0.1 * i as f32;
            world.bodies.push(sim::Body {
                shape: geom::PRIMITIVE_NAMES.len() + i,
                scale,
                // Rule 2 of the importer: the mesh arrives normalised to a unit
                // bounding radius, so this sphere is the collider, not an
                // approximation of one.
                radius: scale,
                mass: scale * scale * scale * 4.0 + 0.2,
                pos: glam::Vec3::new(0.0, 1.2 + 1.8 * i as f32, 0.0),
                vel: glam::Vec3::ZERO,
                rot: glam::Mat4::IDENTITY,
                ang_vel: glam::Vec3::ZERO,
            });
        }
        world.settle();
        run_physics(&mut world, args.seconds);
        let mut empty = seeded_world(args.seed, 0);
        run_physics(&mut empty, args.seconds);

        // A fixed camera, and the same simulated time for both worlds and both
        // windings, so every pixel difference is attributable to one variable.
        let camera = OrbitCamera::default();
        let lights = lights_at(args.seconds);

        let with_bodies = render(
            &mut gfx,
            &shapes,
            &shape_mesh,
            room_mesh,
            quad_mesh,
            ring_mesh,
            &world,
            &camera,
            &lights,
            args.rings,
            args.wireframe,
            args.width,
            args.height,
            None,
        );
        let reference = render(
            &mut gfx,
            &shapes,
            &shape_mesh,
            room_mesh,
            quad_mesh,
            ring_mesh,
            &empty,
            &camera,
            &lights,
            false,
            // The wireframe goes into the reference render as well as the one under
            // test, so the room's own outline cancels out of the comparison and the
            // only edge lines that can register are the bodies' — see the note at
            // the top of `main` about what a wireframe run does to the verdict.
            args.wireframe,
            args.width,
            args.height,
            None,
        );

        // A third render of the same frame, this time with egui over the top.
        // The windowed build composes its HUD exactly this way, so the difference
        // between this image and the one above is the evidence that the two-pass
        // structure works — which is otherwise invisible from here, since a window
        // behind other windows reports itself occluded and never presents.
        let hud_frame = args.hud.then(|| {
            let mut renderer = egui_wgpu::Renderer::new(
                &gfx.device,
                gfx.surface_format,
                egui_wgpu::RendererOptions::default(),
            );
            render(
                &mut gfx,
                &shapes,
                &shape_mesh,
                room_mesh,
                quad_mesh,
                ring_mesh,
                &world,
                &camera,
                &lights,
                args.rings,
                args.wireframe,
                args.width,
                args.height,
                Some(&mut renderer),
            )
        });

        // Report what the renderer was actually built with, not what was asked
        // for, and write one PNG per winding side by side: `<stem>-cw.png`,
        // `<stem>-ccw.png`.
        let actual = gfx.room_front_face();
        let seen = visibility_of(&world, &with_bodies, &reference, &camera);
        let stem = args
            .out
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("probe")
            .to_owned();
        let path = args.out.with_file_name(format!("{stem}-{}.png", tag(actual)));
        write_png(&path, &with_bodies);

        println!("  front_face {actual:?}: {} -> {}", seen.describe(), path.display());
        for miss in &seen.misses {
            println!("    left no trace: {miss}");
        }
        results.push((actual, seen));

        if let Some(frame) = &hud_frame {
            let path = args
                .out
                .with_file_name(format!("{stem}-{}-hud.png", tag(actual)));
            write_png(&path, frame);
            let coverage = left_strip_difference(frame, &with_bodies, HUD_PANEL_WIDTH);
            println!(
                "  hud overlay: {:.0}% of the left {HUD_PANEL_WIDTH} px strip changed -> {}",
                coverage * 100.0,
                path.display()
            );
            hud_coverage.push((actual, coverage));
        }
    }

    // The verdict, which is the whole point of the probe.
    if results.iter().all(|(_, seen)| seen.judged == 0) {
        eprintln!("probe: not one body landed inside the frame — widen --width/--height or raise --count");
        std::process::exit(2);
    }
    if results.iter().all(|(_, seen)| seen.visible == 0) {
        eprintln!(
            "probe: nothing was visible on either winding — the scene is not being \
             drawn at all"
        );
        std::process::exit(1);
    }

    let passing: Vec<_> = results
        .iter()
        .filter(|(_, seen)| seen.ratio() >= VISIBILITY_PASS)
        .map(|(face, _)| *face)
        .collect();

    let winding_ok = matches!(passing.as_slice(), [_]);

    match passing.as_slice() {
        [face] => {
            let default = GfxOptions::default().room_front_face;
            if *face == default {
                println!(
                    "probe: ok — only {face:?} shows the interior, which is what \
                     GfxOptions::default() uses"
                );
            } else {
                eprintln!(
                    "probe: {face:?} is the winding that shows the interior, but \
                     GfxOptions::default() uses {default:?} — fix gfx.rs"
                );
                std::process::exit(1);
            }
        }
        [] => {
            eprintln!("probe: no winding shows the bodies inside the room");
            std::process::exit(1);
        }
        both => {
            eprintln!(
                "probe: every body is visible on both windings ({:?}) — the room is \
                 not culling its near walls, so the cutaway is not actually working",
                both.iter().map(|f| format!("{f:?}")).collect::<Vec<_>>()
            );
            std::process::exit(1);
        }
    }

    // The overlay check only runs when it was asked for, and only means anything
    // once the winding check has passed.
    if args.hud && winding_ok {
        if let Some((face, coverage)) = hud_coverage.iter().find(|(_, c)| *c < HUD_COVER) {
            eprintln!(
                "probe: the HUD did not composite on {face:?} — only {:.0}% of the \
                 panel strip changed, less than {:.0}%",
                coverage * 100.0,
                HUD_COVER * 100.0
            );
            std::process::exit(1);
        }
        println!("probe: ok — the egui overlay composited over the scene");
    }
}

/// `Cw` / `Ccw`, for file names.
fn tag(face: wgpu::FrontFace) -> &'static str {
    match face {
        wgpu::FrontFace::Cw => "cw",
        wgpu::FrontFace::Ccw => "ccw",
    }
}

fn seeded_world(seed: u64, count: usize) -> World {
    let mut world = World::default();
    world.set_seed(seed);
    for i in 0..count {
        world.record(i % geom::PRIMITIVE_NAMES.len(), SpawnOpts::default());
    }
    world.settle();
    world
}

/// Advance the simulation in fixed steps, so the frame the probe renders does not
/// depend on how fast this machine runs.
fn run_physics(world: &mut World, seconds: f32) {
    const STEP: f32 = 1.0 / 60.0;
    let steps = (seconds / STEP).round() as i32;
    for _ in 0..steps {
        world.step(STEP);
    }
}

/// The browser build's three lights, with the accent at simulated time `t`:
/// a warm key in the ceiling fixture, a cool fill from a low corner, and a
/// coloured accent orbiting the room.
fn lights_at(t: f32) -> [Light; 3] {
    let a = t * 0.55;
    let (hx, hy, hz) = (sim::ROOM_HALF_X, sim::ROOM_HALF_Y, sim::ROOM_HALF_Z);
    [
        Light::new(Vec3::new(0.0, hy - 0.65, 0.0), Vec3::new(1.45, 1.32, 1.12)),
        Light::new(
            Vec3::new(-hx * 0.6, 1.1, hz * 0.6),
            Vec3::new(0.26, 0.38, 0.58),
        ),
        Light::new(
            Vec3::new(
                a.cos() * hx * 0.55,
                2.4 + (a * 1.7).sin() * 1.3,
                a.sin() * hz * 0.55,
            ),
            Vec3::new(0.72, 0.24, 0.55),
        ),
    ]
}

/// Ground shadows splayed outward from the key light, as in the browser build's
/// `blobsFor`.
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
                center: Vec3::new(
                    (b.pos.x + ox).clamp(-sim::ROOM_HALF_X + 0.05, sim::ROOM_HALF_X - 0.05),
                    0.0,
                    (b.pos.z + oz).clamp(-sim::ROOM_HALF_Z + 0.05, sim::ROOM_HALF_Z - 0.05),
                ),
                // Floor discs face up; the renderer drops a disc whose plane
                // faces away from the camera rather than letting it hang through
                // a culled wall.
                normal: Vec3::Y,
                radius: b.scale * (1.05 + h * 0.12),
                alpha: 0.48 / (1.0 + h * 0.42),
                inner: BLOB_INNER,
                color: BLOB_COLOR,
            }
        })
        .collect()
}

fn panels() -> Vec<Disc> {
    vec![Disc {
        center: Vec3::new(0.0, sim::ROOM_HALF_Y - 0.002, 0.0),
        normal: -Vec3::Y,
        radius: 1.7,
        alpha: PANEL_ALPHA,
        inner: PANEL_INNER,
        color: PANEL_COLOR,
    }]
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

#[allow(clippy::too_many_arguments)]
fn render(
    gfx: &mut Gfx,
    shapes: &[geom::ShapeDef],
    shape_mesh: &[usize],
    room_mesh: usize,
    quad_mesh: usize,
    ring_mesh: usize,
    world: &World,
    camera: &OrbitCamera,
    lights: &[Light; 3],
    rings: bool,
    // The edge overlay rather than only the solids. Off by default: the visibility
    // check reads solid pixels either way, but a wireframe is how an import that
    // had whole faces sampled out of it shows that for what it is.
    wireframe: bool,
    width: u32,
    height: u32,
    hud: Option<&mut egui_wgpu::Renderer>,
) -> Frame {
    let color = gfx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("probe target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gfx.surface_format,
        // The MSAA sample view is created inside `Gfx::render`; this texture only
        // takes the resolve and then feeds the copy.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = color.create_view(&wgpu::TextureViewDescriptor::default());

    let blobs = blobs_for(world, lights);
    let fixtures = panels();
    let splash = if rings { rings_for(world) } else { Vec::new() };

    let scene = Scene {
        view: camera.view(),
        proj: camera.projection(width, height),
        eye: camera.eye(),
        lights: *lights,
        background: BACKGROUND,
        room: RoomDraw {
            mesh: room_mesh,
            wall: WALL,
            floor: FLOOR,
            ceil: CEIL,
            grid_spacing: GRID_SPACING,
            edge_color: EDGE,
            half: [sim::ROOM_HALF_X, sim::ROOM_HALF_Y, sim::ROOM_HALF_Z],
        },
        shapes,
        shape_mesh,
        bodies: &world.bodies,
        blobs: &blobs,
        panels: &fixtures,
        rings: &splash,
        quad_mesh,
        ring_mesh,
        wireframe,
    };

    let target = RenderTarget {
        view: &view,
        format: gfx.surface_format,
        width,
        height,
    };

    let mut encoder = gfx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("probe"),
        });
    gfx.render(&mut encoder, &target, &scene);

    // Pass 2, when this run asked for a HUD: egui over the top of the scene, on
    // the same target the scene resolved onto.
    let extra = match hud {
        Some(renderer) => draw_hud(renderer, gfx, &mut encoder, &view, width, height),
        None => Vec::new(),
    };

    // Read the frame back. Rows have to start on a 256-byte boundary, so the
    // staging row is wider than the image and the copy loop below un-pads it.
    let bytes_per_pixel = 4u32;
    let unpadded = width * bytes_per_pixel;
    let padded = unpadded.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = gfx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe readback"),
        size: u64::from(padded) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &color,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    // wgpu 30 hands back a submission index; the map below waits on that one
    // submission rather than on "whatever is queued".
    let submission = gfx
        .queue
        .submit(std::iter::once(encoder.finish()).chain(extra));

    let buffer_slice = staging.slice(..);
    let (sender, reader) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    gfx.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            // A timeout rather than an indefinite wait: a hung readback should
            // fail this binary, not sit there until someone kills it.
            timeout: Some(std::time::Duration::from_secs(30)),
        })
        .expect("probe: device poll failed");
    reader
        .recv()
        .expect("probe: map callback never ran")
        .expect("probe: buffer failed to map");

    // wgpu 30 made this fallible (`Result<BufferView, MapRangeError>`).
    let data = buffer_slice
        .get_mapped_range()
        .expect("probe: mapped range unavailable");
    let mut pixels = vec![0u8; (unpadded * height) as usize];
    for row in 0..height {
        let from = ((row * padded) as usize)..(((row * padded) as usize) + unpadded as usize);
        let to = ((row * unpadded) as usize)..(((row * unpadded) as usize) + unpadded as usize);
        pixels[to].copy_from_slice(&data[from]);
    }
    drop(data);
    staging.unmap();

    // Byte order follows the format name: `Rgba8Unorm` hands back RGBA, and
    // anything named `Bgra…` hands back BGRA. The headless path uses Rgba8Unorm;
    // the conversion is here so the probe stays correct if a surface format is
    // ever reused for it.
    let bgra = matches!(
        gfx.surface_format,
        wgpu::TextureFormat::Bgra8Unorm
            | wgpu::TextureFormat::Bgra8UnormSrgb
            | wgpu::TextureFormat::Bc7RgbaUnorm
            | wgpu::TextureFormat::Bc7RgbaUnormSrgb
    );
    if bgra {
        for pixel in pixels.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }

    Frame {
        pixels,
        width,
        height,
    }
}

/// What the picture comparison could tell us.
struct Visibility {
    /// Bodies whose neighbourhood changed against the empty-room render.
    visible: usize,
    /// Bodies on screen at all, i.e. the ones that could be judged.
    judged: usize,
    /// Bodies whose centre lands outside the frame or behind the camera. They are
    /// evidence of nothing, and there are always a few: the default camera looks
    /// at a 14.4-unit-wide room from 18 units out, so the near corners fall off
    /// the bottom edge of the image.
    off_frame: usize,
    /// Every body that was on screen but left no trace: what it was, where it
    /// is, and where it should have appeared. A failure this specific is what
    /// turns "79% visible" into an answer instead of a mystery.
    misses: Vec<String>,
}

impl Visibility {
    fn ratio(&self) -> f32 {
        if self.judged == 0 {
            0.0
        } else {
            self.visible as f32 / self.judged as f32
        }
    }

    fn describe(&self) -> String {
        let skipped = if self.off_frame > 0 {
            format!(", {} off-frame not judged", self.off_frame)
        } else {
            String::new()
        };
        format!(
            "{}/{} bodies visible ({:.0}%{skipped})",
            self.visible,
            self.judged,
            self.ratio() * 100.0
        )
    }
}

/// The largest channel difference between two frames at one pixel.
fn channel_delta(frame: &Frame, reference: &Frame, x: u32, y: u32) -> u8 {
    let here = frame.pixel(x, y);
    let there = reference.pixel(x, y);
    here.iter()
        .zip(there.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0)
}

/// How many bodies actually show up on screen: each one's screen-space disc is
/// compared against an otherwise identical render of the same frame with no
/// bodies in it, and a body counts as drawn if enough of that disc changed.
///
/// What that proves is the thing worth proving — that the room is not swallowing
/// the geometry inside it. It does not prove that every body is individually
/// unoccluded: a body behind another body still scores as visible, because the
/// reference has nothing there either. Disambiguating that would need a depth
/// pass per body, which is more machinery than the question asks for.
fn visibility_of(
    world: &World,
    frame: &Frame,
    reference: &Frame,
    camera: &OrbitCamera,
) -> Visibility {
    let view = camera.view();
    let proj = camera.projection(frame.width, frame.height);
    let view_proj = proj * view;
    // Pixels per world unit at depth d is `f * height / (2 d)`, with `f` the
    // projection's own Y scale — that is what sizes the sample disc below, so it
    // tracks the body instead of a guessed patch.
    let y_scale = proj.y_axis.y * frame.height as f32 * 0.5;
    let on_screen = |pos: Vec3| {
        let clip = view_proj * Vec4::new(pos.x, pos.y, pos.z, 1.0);
        if clip.w <= 1e-4 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        let x = (ndc.x * 0.5 + 0.5) * frame.width as f32;
        // Framebuffer rows run top to bottom, NDC Y runs bottom to top.
        let y = (1.0 - (ndc.y * 0.5 + 0.5)) * frame.height as f32;
        if !(0.0..frame.width as f32).contains(&x) || !(0.0..frame.height as f32).contains(&y) {
            return None;
        }
        Some((x as u32, y as u32))
    };

    let mut out = Visibility {
        visible: 0,
        judged: 0,
        off_frame: 0,
        misses: Vec::new(),
    };
    for (i, body) in world.bodies.iter().enumerate() {
        let Some((cx, cy)) = on_screen(body.pos) else {
            out.off_frame += 1;
            continue;
        };
        let (cx, cy) = (cx as f32, cy as f32);

        // The body's radius in pixels, from its own depth. View space looks down
        // -Z, so depth is the negated z coordinate.
        let view_pos = view * body.pos.extend(1.0);
        let depth = (-view_pos.z).max(1e-3);
        let radius = (body.radius * y_scale / depth).max(1.0);

        // Sample the silhouette rather than the centre. A torus has no centre to
        // find — that is what fooled the first version of this check — and a cone
        // on its side has barely any.
        let mut changed = 0usize;
        let mut samples = 0usize;
        for py in (cy - radius).floor() as i64..=(cy + radius).ceil() as i64 {
            for px in (cx - radius).floor() as i64..=(cx + radius).ceil() as i64 {
                let (dx, dy) = (px as f32 - cx, py as f32 - cy);
                if dx * dx + dy * dy > radius * radius {
                    continue;
                }
                // Off-frame samples are excluded from the denominator too, so a
                // body half off the edge is judged on the half we can see.
                if px < 0 || py < 0 || px >= frame.width as i64 || py >= frame.height as i64 {
                    continue;
                }
                samples += 1;
                if channel_delta(frame, reference, px as u32, py as u32) >= PIXEL_EPS {
                    changed += 1;
                }
            }
        }
        if samples < MIN_SAMPLES {
            out.off_frame += 1;
            continue;
        }

        out.judged += 1;
        if changed as f32 / samples as f32 >= DISC_COVER {
            out.visible += 1;
            continue;
        }
        let shape = geom::PRIMITIVE_NAMES
            .get(body.shape)
            .copied()
            .unwrap_or("unknown");
        out.misses.push(format!(
            "#{i} {shape} r={:.2} at ({:.2}, {:.2}, {:.2}): only {changed}/{samples} pixels \
             inside its {radius:.0} px disc changed, near ({cx:.0}, {cy:.0})",
            body.radius, body.pos.x, body.pos.y, body.pos.z
        ));
    }
    out
}

/// A stand-in for the app's HUD, used to exercise the two-pass composition the
/// windowed build relies on: the scene resolved onto a target, then egui painted
/// over it with `LoadOp::Load`.
///
/// It is not `app`'s own HUD — that one sits behind private types — but it stands
/// in the same places and drives the same machinery: egui's font atlas arriving
/// through `TexturesDelta`, a left panel of controls, a status line along the
/// bottom. If the format or the load ops were wrong, this is where it would show.
///
/// Returns the command buffers egui's `update_buffers` produced, which the caller
/// has to submit alongside the frame.
fn draw_hud(
    renderer: &mut egui_wgpu::Renderer,
    gfx: &Gfx,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    width: u32,
    height: u32,
) -> Vec<wgpu::CommandBuffer> {
    let ctx = egui::Context::default();
    ctx.set_visuals(egui::Visuals::dark());

    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(width as f32, height as f32),
        )),
        // egui 0.37 keeps the scale factor on the viewport info rather than on
        // `RawInput`, and a headless context with no viewport info runs at one
        // point per pixel — which is what makes the coverage check below
        // predictable, since the panel then occupies a known number of pixels.
        ..Default::default()
    };

    // Throwaway state for the sliders: the probe is checking that the widgets
    // draw, not that they do anything.
    let mut gravity = sim::GRAVITY;
    let mut bounce = sim::RESTITUTION;
    let mut count = 14usize;
    let mut collide = true;
    let mut wireframe = false;

    let mut out = ctx.run_ui(input, |ui| {
        egui::Panel::bottom("readout").show(ui, |ui| {
            ui.label("60 fps · 14 bodies · 3 contacts · probe");
        });
        egui::Panel::left("controls")
            .resizable(false)
            .show(ui, |ui| {
                ui.heading("Bouncing room");
                ui.label("drag to orbit · scroll to zoom");
                ui.separator();
                ui.horizontal(|ui| {
                    ui.button("Pause");
                    ui.button("Kick");
                    ui.button("Reset");
                });
                ui.label("space · k · r · w");
                ui.separator();
                ui.button("Spawn one");
                ui.add(egui::Slider::new(&mut count, 1..=48));
                ui.label("bodies");
                ui.add(egui::Slider::new(&mut gravity, -40.0..=5.0).text("gravity"));
                ui.add(egui::Slider::new(&mut bounce, 0.0..=1.0).text("bounce"));
                ui.separator();
                ui.checkbox(&mut collide, "body collisions");
                ui.checkbox(&mut wireframe, "wireframe");
            });
    });

    // The same order the app uses: textures first (device work), then the vertex
    // and index data for this frame's draw list.
    for (id, deltas) in std::mem::take(&mut out.textures_delta.set) {
        for delta in deltas {
            renderer.update_texture(&gfx.device, &gfx.queue, id, &delta);
        }
    }
    let screen = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [width, height],
        pixels_per_point: out.pixels_per_point,
    };
    let jobs = ctx.tessellate(out.shapes, out.pixels_per_point);
    let extra = renderer.update_buffers(&gfx.device, &gfx.queue, encoder, &jobs, &screen);

    let hud_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("probe ui"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            // `Load`, not `Clear`: this pass paints on top of the scene that is
            // already in the target. Clearing here is the classic way to end up
            // with a HUD and no room.
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    renderer.render(&mut hud_pass.forget_lifetime(), &jobs, &screen);

    // Drained here rather than after the submit, so `TexturesDelta`'s drop assert
    // stays satisfied on a path that has no submit of its own to wait behind.
    for id in std::mem::take(&mut out.textures_delta.free) {
        renderer.free_texture(&id);
    }

    extra
}

/// Pixels of the left `panel_width` columns that differ between two frames, as a
/// share of that strip. A HUD composited correctly moves most of it; a HUD that
/// never arrived, or a pass that cleared the scene, moves none of it.
fn left_strip_difference(frame: &Frame, reference: &Frame, panel_width: u32) -> f32 {
    let width = panel_width.min(frame.width);
    let mut changed = 0usize;
    let mut total = 0usize;
    for y in 0..frame.height {
        for x in 0..width {
            total += 1;
            if channel_delta(frame, reference, x, y) >= PIXEL_EPS {
                changed += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        changed as f32 / total as f32
    }
}

fn write_png(path: &std::path::Path, frame: &Frame) {
    let file = std::fs::File::create(path).expect("probe: cannot create the output PNG");
    let mut encoder = png::Encoder::new(file, frame.width, frame.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    // The target is a plain unorm texture, so the bytes are already
    // display-referred. png 0.18 exposes no sRGB-chunk setter, and decoders
    // treat 8-bit truecolour as sRGB anyway, so the values land as rendered.
    let mut writer = encoder.write_header().expect("probe: PNG header failed");
    writer
        .write_image_data(&frame.pixels)
        .expect("probe: PNG data failed");
}
