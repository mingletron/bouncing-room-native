# Port contract — native (Rust/wgpu) port of the WebGL app

This is the interface every part of the port is written against. The reference
implementation is `/Users/work/Developer/bouncing-room` (browser WebGL); read its
`src/*.js` for behaviour and its `scripts/check.mjs` for the invariants the port
must satisfy.

Conventions that are the same as the JS version unless stated otherwise:
right-handed, Y-up, world units, room spans x/z in [-halfX, halfX] and y in
[0, halfY]. Room constants: `halfX = halfZ = 7.2`, `halfY = 5.4`.

Differences forced by WebGPU:
- **Depth range is 0..1**, not -1..1. Use `glam::Mat4::perspective_rh` (which maps
  to 0..1) and *not* `perspective_rh_gl`.
- **Cull direction is inverted relative to GL.** WebGPU's framebuffer winding
  convention means the room interior trick ("cull the faces pointing at you")
  needs `front_face` set to the opposite handedness from what the JS vertex order
  implies. Do not guess: the probe must verify the room renders from outside, and
  the renderer exposes `FRONT_FACE` as a single constant to flip.
- No `precision` declarations; `inversesqrt` is `inverseSqrt`.

## File ownership

| File | Owner |
| --- | --- |
| `src/lib.rs`, `src/main.rs`, `src/app.rs`, `src/camera.rs`, `src/gfx.rs`, `src/gpu_types.rs`, `src/import.rs`, `src/bin/probe.rs`, `Cargo.toml` | integrator |
| `src/sim.rs`, `src/geom.rs`, `tests/sim_invariants.rs` | simulation port (built in `scratch-sim/` first) |
| `shaders/lit.wgsl`, `shaders/line.wgsl`, `shaders/flat.wgsl` | shader port |

Nobody edits a file owned by someone else. `Cargo.toml` is final — do not add
dependencies without saying so in your report.

## Vertex format (all meshes, all pipelines)

Interleaved, stride 24 bytes:

| location | type | offset |
| --- | --- | --- |
| 0 | `vec3<f32>` position | 0 |
| 1 | `vec3<f32>` normal | 12 |

A mesh is `Vec<Vertex>` plus either triangle indices (`Option<Vec<u16>>`; `None`
means the vertex list *is* the triangle list) and a `Vec<u16>` of GL_LINES-style
edge indices for wireframes and the room outline.

## Bind groups — identical layout for all three pipelines

That is deliberate: one pipeline-layout-compatible pair of bind groups, reused by
lit, line and flat.

**Group 0 — per frame, `FrameUniforms`, 240 bytes**

```wgsl
struct Light { pos: vec4<f32>, color: vec4<f32> }   // 32 bytes each
struct FrameUniforms {
  @offset(0)   view:  mat4x4<f32>,
  @offset(64)  proj:  mat4x4<f32>,
  @offset(128) lights: array<Light, 3>,
  @offset(224) cam_pos: vec4<f32>,
}
```

`pos.w` and `color.w` are unused padding; keep them 0 and 1. Light count is fixed
at 3 (key / fill / orbiting accent). `@size(240)` on the struct.

**Group 1 — per object, `ObjectUniforms`, 112 bytes**

```wgsl
struct ObjectUniforms {
  @offset(0)   model: mat4x4<f32>,
  @offset(64)  color: vec4<f32>,
  @offset(96)  params: vec4<f32>,
}
```

Semantics per pipeline (all three read the same struct):

- `lit`: `color.rgb` = base colour, `color.a` = 1. `params = (gloss, specStrength,
  isRoom, gridSpacing)`.
- `line`: `color.rgba` = line colour including alpha. `params` ignored.
- `flat`: `color.rgb` + `params.x` = alpha, `params.y` = inner radius (the radius
  fraction out to which the disc is fully opaque, before it fades to nothing at 1).

There is **no separate normal-matrix uniform.** Model scales are always uniform, so
`normalize(mat3(model) * normal)` is correct. In WGSL, extract the upper-left 3x3
by constructing `mat3x3(model[0].xyz, model[1].xyz, model[2].xyz)`.

## Vertex out (lit pipeline) and fragment in

```wgsl
struct VsOut {
  @location(0) world:  vec3<f32>,
  @location(1) normal: vec3<f32>,
  @builtin(position) clip: vec4<f32>,   // `line` may use it for a size falloff
}
```

Fragment shaders write `@location(0) vec4<f32>`.

## Lighting maths to reproduce

Match the JS output closely enough that a side-by-side screenshot looks like the
same app; the exact tonemap is worth keeping.

- Flip the normal to face the viewer: `if dot(N, V) < 0 { N = -N; }` (back faces
  are never lit, which matters because most objects draw with culling off).
- Hemispheric ambient: `mix(vec3(0.045,0.05,0.062), vec3(0.15,0.16,0.19), 0.5+0.5*N.y)`.
- Per light: `att = 1/(1 + dist2*0.021)`, diffuse `base*color*ndl*att`, Blinn-Phong
  spec `color*pow(ndh, gloss)*ndl*att`, then `spec *= specStrength`.
- Rim: `pow(1-max(dot(N,V),0), 3) * 0.055 * vec3(0.65,0.78,1.0)`.
- Room-only (`isRoom > 0.5`): base colour mixes to `floorColor` when `N.y > 0.5`
  and to `ceilColor` when `N.y < -0.5`; floor grid lines on x/z at `gridSpacing`
  and wall seams, antialiased with a distance-scaled width (camera distance is
  available as `length(cam_pos.xyz - world)`). **Do not use `dpdx`/`dpdy`/`fwidth`
  derivatives for the grid** — the JS version deliberately avoids derivatives, and
  the distance-scaled width is what keeps it portable.
- Tonemap, keep identical: `c = max(c,0); c = 1 - exp(-c*1.25); return pow(c, 1/0.92-ish)`
  — the JS is `pow(1-exp(-c*1.25), vec3(0.92))`.

Colours and light intensities are already tuned in the JS (`src/main.js`): key
`[1.45,1.32,1.12]`, fill `[0.26,0.38,0.58]`, accent `[0.72,0.24,0.55]`, wall
`[0.2,0.22,0.25]`, floor `[0.13,0.14,0.16]`, ceiling `[0.16,0.17,0.19]`.

## Pipeline state the renderer must set (for reference)

| pipeline | cull | depth write | depth compare | blend | polygon offset |
| --- | --- | --- | --- | --- | --- |
| room (lit) | Front (see caveat above) | on | Less | off | (1.5, 1.5) |
| objects (lit) | None | on | Less | off | (1.5, 1.5) |
| discs (flat) | None | **off** | Less | alpha | none |
| lines | None | **off** | LessEqual | alpha | none |

Order: clear → room → ground discs + ceiling panel → objects → lines (room
outline, wireframe, impact rings). MSAA: 4 samples.

## Simulation and geometry types (the integrator's stubs are authoritative)

```rust
#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex { pub position: [f32; 3], pub normal: [f32; 3] }

pub struct Mesh { pub name: &'static str, pub vertices: Vec<Vertex>,
                  pub indices: Option<Vec<u16>>, pub edges: Vec<u16> }

pub struct ShapeDef { pub mesh: Mesh, pub color: [f32; 3], pub gloss: f32 }
pub fn all_primitives() -> Vec<ShapeDef>;          // 9 shapes
pub fn room_mesh(hx: f32, hy: f32, hz: f32) -> Mesh;
pub fn quad_mesh() -> Mesh;                        // unit XZ quad, [-1,1]
pub fn ring_mesh(segments: u32) -> Mesh;           // unit circle as line pairs

#[repr(C)] pub struct Body { pub shape: usize, pub scale: f32, pub radius: f32,
    pub mass: f32, pub pos: Vec3, pub vel: Vec3, pub rot: Mat4, pub ang_vel: Vec3 }

pub struct Impact { pub pos: Vec3, pub n: Vec3, pub color: [f32; 3],
    pub base_radius: f32, pub age: f32, pub life: f32 }

pub struct World { /* bodies, impacts, gravity -17, restitution 0.84, drag 0.06,
    collide bool, wall_friction 0.16, pair_friction 0.12, spin_coupling 0.45 */ }
```

`World::step(dt)` substeps internally (max 4 substeps at ~1/120 s). Behaviour that
must be preserved and is covered by tests: friction acts on **contact-point slip**
(`v + omega × r_contact`), not on tangential velocity — that is what makes bodies
roll instead of skidding; the normal impulse pushes the pair *apart*; contacts
below a resting speed stop bouncing so piles settle; spin only ever comes from the
tangential part of an impulse (a radial impulse has no moment arm).

Every primitive has bounding radius exactly 1.0, and its collision radius is
`scale * radius_factor(shape name)` where the factors live in `sim.rs`
(sphere 1.0, cube 0.82, capsule 0.9, torus 0.86, cone 0.82, cylinder 0.85,
icosahedron 0.95, tetrahedron 0.74, octahedron 0.86).

## Imported geometry — `src/import.rs`

The one exception to "everything is generated in code". A file enters through
`import::import_path`, which returns a `Mesh` that already obeys four rules, and
the app layer never has to re-check them:

1. **One body per file.** Multi-object files (an OBJ with several `o`/`g` groups)
   merge into a single mesh, because the sim collides bodies as spheres and
   splitting an assembly would let its parts bounce apart from each other.
2. **Normalised to a unit bounding radius**, so rule 2 of the section above applies
   unchanged with `radius_factor = 1.0`: the collision sphere *is* the mesh's
   bounds, not an approximation. File units are never trusted — only proportions.
3. **Inside the u16 index budget** (`MAX_VERTS`). Over-budget meshes have whole
   faces **sampled out at an even stride**, never cut from the tail: files are
   written in spatially coherent order, so a tail-cut draws half an object while
   its collider still spans the whole thing. Welding can only lower the vertex
   count, so sizing the stride from the corner count cannot overshoot.
4. **`edges` are synthesised** from the triangles, or the wireframe toggle would
   draw nothing for an import.

Every repair (sampling out, degenerate faces removed, normals rebuilt from the
winding, edge cap) is reported in `Import::notes` and shown in the HUD. Silent
geometry changes count as bugs.

`.obj` reads through the `tobj` crate — added with the integrator's approval,
because face-element index forms (`v`, `v/vt`, `v//vn`), negative indices and
polygon fan-triangulation are the parts of OBJ that hand-rolling gets wrong.
`.stl` is parsed here (binary and ASCII; the encoding is decided by header count
*and* length rather than by a `solid` prefix, since plenty of binary STLs put the
object name in that free-text field), and welded on position **and** normal so
faceted models stay faceted instead of rounding their own edges off.

`.fbx`, `.blend`, `.usd` and `.usdz` are refused with a message, not attempted: no
maintained Rust reader exists for them (USD in particular is a composition system,
so "read the mesh" means implementing a resolver). The seam for a converter is
`import_path` — Blender and assimp both read all six and write OBJ — and
everything downstream of it is already format-agnostic.

`probe --import PATH` renders an imported mesh through the real GPU path and
subjects it to the same pixel-visibility check as a generated primitive, so
"imported" has a definition stronger than "parsed without error".

Imported shapes are appended in lockstep to `App::shapes`, `Gpu.shape_mesh` and
`World::colors`; all three are addressed by `Body::shape` and must stay the same
length. `Body::shape >= PRIMITIVE_NAMES.len()` is the entire test for "this body
is an import" — which is what lets the count slider and `r` manage only the
generated population while `Clear all` (`x`) drops imports too.

One buffer invariant the format implies: `Gfx::upload_mesh` pads index lists to
`COPY_BUFFER_ALIGNMENT`. u16 entries cost 2 bytes each, so an **odd** index count
(any odd triangle count) is 2 mod 4 and wgpu rejects the copy. Every generated
primitive happens to have an even triangle count, which is how this stayed latent
until a sampled file produced an odd one.
