# Bouncing room — native

A room full of bouncing primitives, rendered natively in Rust with
[wgpu](https://wgpu.rs) (Metal on macOS, Vulkan on Linux) and controlled from an
[egui](https://github.com/emilk/egui) HUD. It is the native port of a
dependency-free WebGL app of the same name; the interface the port is written
against — vertex formats, bind-group layouts, lighting maths, the simulation
invariants — is [`docs/PORT_CONTRACT.md`](docs/PORT_CONTRACT.md).

## What it looks like

![Nine generated primitives settled on the floor of a lit room, seen from outside through the cutaway walls](docs/screenshots/room.png)

Nine primitives — sphere, cube, capsule, torus, cone, cylinder, icosahedron,
tetrahedron, octahedron — generated in code, tumbling and piling up under gravity.
The room's box is wound so that *front* faces are culled: orbiting outside gives a
cutaway, while a camera inside the room is fully enclosed. Three point lights do
the shading — a warm key in the ceiling fixture, a cool fill from a low corner, and
a coloured accent that orbits the room so the picture never quite settles — plus
hemispheric ambient, a rim term, ground shadow blobs splayed away from the key
light, and a floor grid drawn in the fragment shader without screen-space
derivatives.

## Run it

```sh
cargo run                        # the window
cargo test                       # 51 unit + 6 fixture + 24 simulation invariants
cargo run --bin probe -- --out probe.png    # headless render, then checks its own pixels
cargo run -- --help              # CLI flags, including --import
```

It runs on macOS and Linux from the same source, with no platform-specific code
or build flags. On Linux it needs a Vulkan driver, and opens its window on Wayland
or X11, whichever the session offers.

The probe is the interesting one. It renders a **seeded** scene offscreen, writes a
PNG, and then reads the picture back: every body's centre is projected into the
frame and compared against the same pixel from an otherwise identical *empty*
room. It requires exactly one room winding to show every body — both windings
passing would mean the room is not being culled at all, and neither passing would
mean nothing is being drawn. That winding is the one thing in this port that no
spec could decide, because WebGPU's framebuffer convention inverts GL's.

## Controls

| Input | Effect |
| --- | --- |
| drag | orbit |
| right-drag | grab the room: slides the orbit centre, which is clamped inside the walls |
| scroll, pinch | dolly |
| `space` | pause / resume |
| `.` | frame-step: pauses, then advances exactly 1/60 s — two 1/120 substeps, so a scrubbed step is indistinguishable from a frame of a running sim |
| `k` | kick: a random impulse on every body |
| `r` | replay the generated room from the seed in the panel |
| `x` | clear every body, imported ones included |
| `w` | wireframe overlay |
| `c` | recenter the camera; leaves the auto-orbit setting alone |
| `h` | hide the whole HUD — for a clean picture; `h` again brings it back |
| `f` | full screen (the display's native mode) |

The panel carries sliders for gravity, bounce, simulation speed, body count (0
empties the room), air drag, wall grip,
body grip and spin coupling; toggles for body-body collisions, wireframe and
auto-orbit; buttons for pause, step, kick, reset, clear, spawn and re-seed; a
picker for which of the nine primitives the next spawn adds and how big it is; and
a seed box.

The four contact dials are worth a moment: `spin coupling` is what turns a sliding
body into a rolling one, and at 0 the room is effectively frictionless ice.

## Importing objects

![Three imported objects among the primitives: a pale spike, a pale faceted diamond, and a crumpled sheet that arrived sampled because it was too big for the index format](docs/screenshots/imported.png)

Drag a file onto the window, type its path into the panel, or pass
`--import PATH` at startup. Supported: **`.obj`** (Wavefront, read through
[`tobj`](https://crates.io/crates/tobj)) and **`.stl`** (binary and ASCII, parsed
in-tree). Whatever arrives obeys four rules:

1. **One body per file.** A multi-object OBJ or a whole `.blend`-style assembly
   merges into a single mesh, because bodies collide as bounding spheres and
   splitting an assembly would let its parts bounce away from each other.
2. **Normalised to a unit bounding radius**, measured to the farthest vertex, so
   the file's units are never trusted — a cube authored in millimetres and one
   authored in kilometres import alike.
3. **Inside the renderer's 16-bit index budget.** Oversized meshes are welded (an
   STL writes one vertex per corner, which welds down by about 3x) and then have
   whole faces **sampled out at an even stride** rather than cut from the tail:
   files are written in spatially coherent order, so a tail-cut draws half an
   object while its collider still spans the whole thing.
4. **Given wireframe edges**, synthesised from the triangles, so the `w` overlay
   works for imports too.

Every repair and reduction is reported in the panel rather than applied silently.

`.fbx`, `.blend`, `.usd` and `.usdz` are refused with a message rather than
attempted: FBX is a binary AST with mutually incompatible dialects, `.blend` is a
self-describing blob whose layout tracks Blender's own version, and USD is a
*composition* system where the geometry you see may be instanced, inherited
across layers, or refined from a proxy. No maintained Rust reader exists for any
of them. The seam for adding a converter is `import::import_path` — Blender and
assimp both read all six and write OBJ — and everything downstream of it is
already format-agnostic.

The image above is the same scene with `tests/fixtures/spike.obj`,
`diamond.stl` and `crumple.stl` added; the crumpled sheet is 24 200 facets, of
which 12 100 were sampled out. In wireframe the sampling is visible for what it
is — a whole object, perforated, rather than half of one:

![The same three imports drawn as wireframes; the sampled sheet reads as a lattice](docs/screenshots/imported-wireframe.png)

## Reproducing these pictures

All three come from the headless probe, which is how these images can be
deterministic — a fixed seed, fixed simulated time, no window, no wall clock.
They are renders of the 3D scene, **not window captures**: none of them shows the
egui panel, which `h` hides in the live app anyway.

```sh
cargo run --bin probe -- --front-face ccw --width 1280 --height 800 \
    --count 14 --seconds 2.2 --out docs/screenshots/room
cargo run --bin probe -- --front-face ccw --width 1280 --height 800 \
    --count 12 --seconds 2.4 --out docs/screenshots/imported \
    --import tests/fixtures/spike.obj --import tests/fixtures/diamond.stl \
    --import tests/fixtures/crumple.stl
# and again with --wireframe for the third
```

The probe appends the winding to the name it is given (`room-ccw.png`), hence the
renaming. It prints the counts and every repair for each import as it reads them,
so the sampling above is visible in the output rather than only in the picture.

## Where things live

| File | What it is |
| --- | --- |
| `src/geom.rs` | the nine primitives, the room box, and the helper quad and ring — all generated at startup |
| `src/sim.rs` | the physics: walls, pairs, contact-point friction, substepping |
| `src/gfx.rs`, `src/gpu_types.rs` | four pipelines that share one bind-group layout, MSAA, and the CPU/GPU structs they read |
| `shaders/*.wgsl` | lit, flat and line shaders |
| `src/import.rs` | reading `.obj` and `.stl` into the shape list |
| `src/app.rs`, `src/camera.rs` | the window, the HUD, the frame loop, the orbit camera |
| `src/bin/probe.rs` | the offscreen render and the pixel checks described above |
| `docs/PORT_CONTRACT.md` | the interface the port is written against |
| `tests/`, `tests/fixtures/` | the invariants, and real files to import |

## Limits worth knowing

- **Colliders are bounding spheres.** That is the simulation's model, not an
  shortcut in the importer: a teapot bounces as a sphere. The built-in primitives
  get tuned radius factors (0.74–1.0) because their meshes do not fill their
  spheres; an import gets 1.0 because it is normalised *to* its sphere, so an
  elongated model over-covers its collider.
- **Imported files bring no materials.** This renderer has one colour and one gloss
  per body, so an OBJ's `mtllib` is dropped and imports cycle a small palette.
- **No up-axis or unit inference.** OBJ and STL carry neither, so the file's axes
  are kept as modelled: a Z-up CAD part lands on its side.
- **No drag-and-drop under Wayland.** winit 0.30 delivers file drops on macOS and
  X11 only, so in a Wayland session a dropped file is ignored; use the path box or
  `--import`. Running with `WAYLAND_DISPLAY=` unset forces X11 (through XWayland),
  where dropping works.
- **The HUD is the only UI.** No menus, no file dialog: import by drop, path box,
  or flag.
