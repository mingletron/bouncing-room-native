# Port contract — native (Rust/wgpu) port of the WebGL app

This is the interface every part of the port is written against. The reference
implementation is `/Users/work/Developer/bouncing-room` (browser WebGL); read its
`src/*.js` for behaviour and its `scripts/check.mjs` for the invariants the port
must satisfy. It describes the code as it is now, not the plan from the first
commit: instancing, shadow maps, convex hulls, imports, recording and rewind all
postdate the original document, and where the old version was wrong that is called
out below rather than quietly overwritten.

Conventions shared with the JS version unless stated otherwise: right-handed,
Y-up, world units; the room spans x/z in `[-halfX, halfX]` and y in `[0, halfY]`;
`halfX = halfZ = 7.2`, `halfY = 5.4`.

## Things WebGPU forced, corrected

- **Depth range is 0..1**, not -1..1. glam 0.34 moved the constructors into
  `glam::camera`: the scene projection is
  `glam::camera::rh::proj::directx::perspective` (`src/camera.rs:80`), *not* the
  `opengl` or `vulkan` sibling. The shadow faces use the **left-handed** sibling,
  `lh::proj::directx::perspective` with `lh::view::look_to_mat4`
  (`src/gfx.rs:277`, rationale at `:270-274`) — cube-map faces are left-handed and
  a right-handed view renders each face mirrored. The old contract had no vocabulary
  for that distinction, which is how it became a trap.
- **The winding claim in the old contract was wrong.** It said WebGPU inverts GL's
  convention, so `front_face` must be flipped away from what the JS vertex order
  implies, and named a `FRONT_FACE` constant. The room in fact draws correctly with
  `FrontFace::Ccw` — the same handedness GL used — culling `Front`. The setting is
  `GfxOptions { room_front_face }` (`src/gfx.rs:84-88`, default `Ccw`), read back
  with `Gfx::room_front_face()` (`:806`) so a caller never assumes it. **The rule
  that mattered most still stands: do not reason this out, verify it with the
  probe**, which renders both windings and requires exactly one of them to show
  every body (`src/bin/probe.rs:534-583`). `Ccw` holds on Metal *and* Vulkan under
  wgpu 30 (`src/gfx.rs:22-27`).
- **No derivative functions anywhere in the shaders** — `dpdx`, `dpdy`, `fwidth`
  are absent by design, in the grid as originally and now in shadow sampling too:
  `lit.wgsl` uses `textureSampleCompareLevel` (fixed LOD 0) rather than the
  gradient-taking `textureSampleCompare` (`shaders/lit.wgsl:135-139`). Grid
  antialiasing is `smoothstep` against a CPU-computed, distance-scaled width
  (`:146-151`, `:212`).
- `@offset`/`@size` on WGSL structs are **not usable**: naga 30 rejects them
  (`shaders/lit.wgsl:39-41`). Member order alone determines layout, which is why a
  member a shader does not read must still be declared — deleting it shifts every
  field after it (`flat.wgsl:8-10`, `line.wgsl:6-8`, `trail.wgsl:6-8`).

## File ownership

| Owner | Files |
| --- | --- |
| integrator | `src/lib.rs`, `src/main.rs`, `src/app.rs`, `src/camera.rs`, `src/gfx.rs`, `src/gpu_types.rs`, `src/import.rs`, `src/record.rs`, `src/effects.rs`, `src/bin/probe.rs`, `README.md`, `docs/`, `Cargo.toml`, `.github/` |
| simulation port | `src/sim.rs`, `src/hull.rs`, `src/geom.rs`, `tests/sim_invariants.rs`, `tests/new_physics.rs`, `scratch-sim/` |
| shader port | `shaders/lit.wgsl`, `shaders/line.wgsl`, `shaders/flat.wgsl`, `shaders/shadow.wgsl`, `shaders/trail.wgsl` |
| either, with a note | `tests/import_fixtures.rs`, `tests/gpu_layout.rs`, `tests/fixtures/` |

Nobody edits a file owned by someone else. `Cargo.toml` was declared final in the
first version of this document; that has since been crossed deliberately, with
approval. The additions are `tobj` (Wavefront), `gltf` (glTF/GLB, default features
off — this renderer has no texture coordinates to put an image on), `rfd` (the
system file dialog, needed because winit delivers no file drops under Wayland) and
`gif` (clip recording). A converter branch in `src/import.rs` needs **no** new
dependency: it is `std::process::Command` plus the glTF reader that is already here.

## Shared CPU/GPU structs — `src/gpu_types.rs`

Sizes **and every field offset** are asserted at compile time (`:110-141`): four
`size_of` lines, then fifteen `offset_of!` ones. The offsets are what make the mistake
that sizes cannot see a compile error — exchanging `floor_color` with `ceil_color`, or
`cam_pos` with `shadow`, keeps every size intact, compiles, and leaves the shaders
reading the camera position out of the shadow block for the rest of the session.
Measured: with only the size asserts in place, that swap passed the whole suite.

What neither kind of assert can do is see the WGSL side. A shader struct that shrank
below the Rust struct would still validate, because the bind group declares only
`min_binding_size: Some(288)` (`src/gfx.rs:370-372`), and the render would read garbage
silently — that half is `tests/gpu_layout.rs`, below. And the attribute tables in
`gfx.rs` restate `Instance`/`LineVertex` offsets by hand with nothing comparing them
here: change one and the assert fires, but `gfx.rs` moves only because whoever fixed it
remembered.

| Struct | Rust repr | Fields in order (byte offsets) | Assert | Role |
| --- | --- | --- | --- | --- |
| `Light` | `repr(C, align(16))` | `pos: Vec4` @0, `color: Vec4` @16 | 32 | element of the uniform array |
| `FrameUniforms` | `repr(C, align(16))` | `view` @0, `proj` @64, `lights: [Light; 3]` @128 (stride 32), `cam_pos` @224, `floor_color` @240, `ceil_color` @256, `shadow` @272 | 288 | the per-frame uniform (group 0) |
| `Instance` | `repr(C)`, align 4, **no** `align(16)` | `model: [[f32;4];4]` @0,16,32,48, `color` @64, `params` @80 | 96 | a **vertex-buffer element**, step=instance |
| `LineVertex` | `repr(C)` | `pos: [f32;3]` @0, `color: [f32;4]` @12 | 28 | a **vertex-buffer element**, step=vertex |

`Instance` used to be a uniform (`ObjectUniforms`, `repr(C, align(16))`) and is now
a vertex record; giving it `align(16)` would make it 112 bytes and the assert fires.
`LineVertex`'s `color` at offset 12 is legal for a `Float32x4` attribute — WebGPU
requires only a multiple of the component size — and depends on `[f32; 4]` being
align-4 in Rust.

`FrameUniforms` is mirrored byte-for-byte in **four** shader files
(`lit.wgsl:48-58`, `flat.wgsl:16-26`, `line.wgsl:14-24`, `trail.wgsl:13-23`). Add a
field and all four copies plus the Rust struct must move together.
`tests/gpu_layout.rs` **discovers** the `.wgsl` files instead of listing them, parses
each copy out of the shader text, compares its **names, order and types** to the Rust
declaration, then computes the offsets with WGSL's alignment rules — including the way a
struct rounds its own size up — and compares every one against `offset_of!` from the
Rust struct, and the total against `size_of`. So the shaders are checked against Rust's
layout rather than against a third hand-written transcription of it, which is the
difference between catching a same-typed field swap and not. Deletion, insertion,
reordering and a changed member type in one file alone are all caught; a rename in the
Rust struct is a compile error in the test's own `offset_of!` table.
`shaders/*.wgsl` is the file list, so a fifth copy is checked without anyone editing the
test, and the count of copies is pinned at "at least four" so a vanished one fails loudly
instead of quietly narrowing the check to nothing.

Naga is not called, because naga 30 has no `@offset`, so the parsing is hand-written —
and a hand-written parser that silently stopped seeing members would turn every other
test in this file green. `the_parser_notices_every_mutation` exists for that: it feeds
the parser four mutated copies of a real struct (one member deleted, one retyped, one
inserted, one renamed) and insists each reads differently from the original. The parser
splits members at commas of angle-bracket depth zero, so `array<Light, 3>` stays one
member, and comments come out of the whole file before anything is searched for, because
a comment can hold a comma, a brace or the words `struct FrameUniforms {`.

`Light` is checked in **all four** copies, not just `lit.wgsl`: each file declares its
own, and the stride of `array<Light, 3>` at byte 128 comes from the local one, so a
`vec3` slipped into `line.wgsl` moves `cam_pos` and everything after it for that
pipeline alone.

`Light::new` fixes the padding contract: `pos.w = 0`, `color.w = 1`. `LIGHT_COUNT`
is 3 — key, fill, orbiting accent. `floor_color`/`ceil_color` moved out of the old
per-object block into the frame block because with instancing they are the same for
every room instance (`gpu_types.rs:52-54`). `shadow` is
`(near, far, strength, bias)`; `strength` is 0 when shadows are off, which is how
the shader early-outs without unbinding anything.

## Mesh and vertex format

A mesh is `geom::Mesh { name: &'static str, vertices: Vec<Vertex>, indices:
Option<Vec<u16>>, edges: Vec<u16>, colors: Option<Vec<[u8; 4]>> }`
(`src/geom.rs:42-52`). `None` indices means the vertex list *is* the triangle list;
`edges` are GL_LINES-style pairs for wireframes and the room outline; `colors` is
per-vertex RGBA8, honoured **only when its length equals the vertex count**
(`src/gfx.rs:825-831`) and otherwise silently replaced by white — the load-bearing
detail for imports, since only a glTF file supplies colours.

The old contract's "stride 24, locations 0 and 1, all pipelines" is now one of
**three** bound buffers for a mesh draw:

| slot | stride | step | locations → format |
| --- | --- | --- | --- |
| 0 mesh vertices | 24 | Vertex | 0 → `Float32x3` @0, 1 → `Float32x3` @12 |
| 1 vertex colours | 4 | Vertex | 8 → `Unorm8x4` @0 |
| 2 instances | 96 | Instance | 2,3,4,5,6,7 → `Float32x4` @0,16,32,48,64,80 |

A shader omits the attributes it does not read: `flat.wgsl` skips the normal and
the vertex colour, `line.wgsl` skips normal, params and colour, `shadow.wgsl` reads
positions and four of the model columns. The trail pipeline is separate entirely —
one buffer, `LineVertex` at stride 28, locations 0 and 1, no instance buffer, no
model matrix (`shaders/trail.wgsl:1-3`).

Which generated meshes are indexed: sphere, capsule, torus, room, quad, ring.
Which are unindexed triangle soup: cube, tetrahedron, octahedron, icosahedron,
cone, cylinder (`geom.rs:389-455`). Every import is indexed.

## Bind groups — three layouts, not one compatible pair

The original "identical layout for all three pipelines" idea is gone: instancing
moved per-object data into a vertex buffer, so there is no per-draw bind group any
more.

- **`bgl_frame`** (`src/gfx.rs:360-376`): group 0 binding 0, uniform,
  `min_binding_size = 288`, visibility VERTEX+FRAGMENT. Bound by `frame_bg`,
  rewritten once per frame (`:940-957`), set once at the top of the scene pass
  (`:1036`). Used by room, object, flat, line, trail.
- **`bgl_shadow_read`** (`:378-398`): group 1 binding 0 `texture_depth_cube` (Depth
  sample type), binding 1 a **comparison** sampler, both FRAGMENT-only. A
  per-resource bind group, set once per frame just after the room pipeline is
  selected (`:1042`) and still bound for the object pipeline, because room and
  object share `lit_layout`.
- **`bgl_face`** (`:400-412`): group 0 binding 0, uniform with **dynamic offset**,
  `min_binding_size = 64`, VERTEX-only (no fragment stage). One bind group, six
  256-byte slots (`FACE_SLOT_BYTES`, `:72-74` — 256 because that is the default
  `min_uniform_buffer_offset_alignment`), selected per shadow pass.

Pipeline layouts: `lit_layout = [frame, shadow read]`; `plain_layout = [frame]`;
`shadow_layout = [face]`. `immediate_size: 0` throughout — there are no push
constants in this renderer.

## Pipelines — six, not three

| pipeline | shaders | topology | cull | depth write / compare | blend | depth bias | samples |
| --- | --- | --- | --- | --- | --- | --- | --- |
| room | lit vs/fs | TriangleList | **Front**, `front_face = GfxOptions::room_front_face` (default `Ccw`) | on / Less | none | — | 4 |
| object | lit vs/fs | TriangleList | None | on / Less | none | — | 4 |
| flat (discs) | flat vs/fs | TriangleList | None | **off** / Less | alpha | — | 4 |
| line (outline, wireframe, rings) | line vs/fs | **LineList** | None | **off** / **LessEqual** | alpha | — | 4 |
| trail (trails, sparks) | trail vs/fs | **LineList** | None | off / LessEqual | alpha | — | 4 |
| shadow | shadow vs only, `fragment: None` | TriangleList | None | on / Less | n/a, no colour attachment | **const 2, slope 2.0, clamp 0** | **1** |

`front_face` defaults to `Ccw` for every pipeline except the room, where it is the
option; with culling off it has no effect (`src/gfx.rs:534-536`).

**The old "polygon offset (1.5, 1.5)" column was fictional** — WebGPU has no colour
polygon offset. Z-fighting is solved with geometry instead: the room outline is
scaled inward by `ROOM_OUTLINE_INWARD = 0.9965` about `(0, half_y*0.5, 0)`
(`:45-49`, `:1320-1331`), wireframe shells are scaled out by `WIREFRAME_SHELL =
1.006` (`:51-53`), discs and rings are lifted 0.006 along their normal by
`disc_matrix` (`gpu_types.rs:154-176`, `disc_matrix`), and the line pipelines compare with
`LessEqual` (`:572-578`). The only real rasteriser bias is the shadow pass's depth
bias, which exists to stop self-shadowing acne.

The colour target is always `Gfx::surface_format`, chosen **non-sRGB where
possible** (`pick_surface_format`, `:1404-1416`: `Rgba8Unorm`, then `Bgra8Unorm`,
then any sRGB, then the first offered) because the tonemap below is already
display-referred and an sRGB target would double-encode it. Headless with no
surface gives `Rgba8Unorm`.

## Passes and their order

Per `Gfx::render`, in order (`src/gfx.rs:920-1077`):

1. CPU staging: rewrite `frame_buffer` (288 bytes); rebuild the instance buffer
   (grow by re-creating, doubling to a power of two above what the frame needs);
   truncate the line list to an even count and rewrite the line buffer.
2. **Six shadow passes**, one per cube face, when `scene.shadows` (`render_shadows`,
   `:1081-1113`): `color_attachments: &[]`, depth attachment = the D2 view of cube
   array layer *i*, load Clear(1.0)/Store, pipeline `shadow`, bind group 0 with
   dynamic offset `i * 256`, one `fill` per non-empty shape. Only bodies are
   casters — the room, discs, lines and trails never enter the cube
   (`:1079-1080`).
3. **The scene pass**: MSAA colour attachment (4 samples) with
   `resolve_target = Some(target.view)` where the target is the caller's
   non-multisampled texture (surface or capture), `load: Clear(background)`, plus
   the 4-sample `Depth32Float` attachment. Inside it, in order: bind the frame
   group and the instance buffer once → **room** (exactly instance 0) → **discs**
   (ground blobs *and* the ceiling panel in one instanced draw) → **bodies**, one
   draw per shape that has bodies → **lines**: room outline, then wireframe shells
   per shape, then impact rings → **trails/sparks**, non-indexed.
4. In the windowed app, a **second pass on the same resolved target** for egui:
   `load: Load`, no depth attachment at all (`src/app.rs:1768-1782`), submitted in
   the same batch. A windowed frame is therefore 8 passes with shadows on, 2
   without.

Draw-call count is **O(shapes), not O(bodies)**: with *S* shapes present, the
default 14-body probe frame is 13 scene draws + 6×S = 54 shadow draws = 67 draws
across 7 passes. Pre-instancing, the same scene needed one draw per body per pass,
and the HUD's 4000-body ceiling would have been ~28 000 draws.

`ensure_targets` keys the MSAA+depth pair by `(width, height, format)` and keeps at
most two sets, evicting the oldest — the two-size case is the window plus a clip
(`:878-918`).

## The shadow map

`SHADOW_SIZE = 1024` per face, `SHADOW_NEAR = 0.08`, `SHADOW_FAR = 30.0`,
`SHADOW_STRENGTH = 0.82`, `SHADOW_BIAS = 0.0004` (`src/gfx.rs:63-71`). One
`Depth32Float` texture, `depth_or_array_layers: 6`, one mip, sample count 1, usage
`RENDER_ATTACHMENT | TEXTURE_BINDING`; six D2 views (one per layer, used as depth
attachments) and one Cube view (used for sampling) over the *same* texture — it is
never copied into a cube. Sampler: `compare = LessEqual`, linear min/mag, rest
default.

`shadow_face_matrices(light) -> [Mat4; 6]` (`:268-292`) is public and builds the
matrices in the cube map's own face order (+X, -X, +Y, -Y, +Z, -Z), 90° FOV, 1:1
aspect, left-handed view, 0..1 depth.

Sampling (`shaders/lit.wgsl:113-141`): early return 1.0 when `frame.shadow.z <= 0`;
the sample point is pushed `n * 0.035` along the *shading* normal — already flipped
toward the viewer — on top of the pipeline depth bias and a reference bias in
`frame.shadow.w`; the reference depth reproduces the face projection's own mapping,
using `z = max(|v.x|,|v.y|,|v.z|)` because that is the component the hardware picks
the face by; five comparison taps over a kernel that widens with distance
(`spread = z * 0.0035`), with the tangent basis switching its reference up vector on
±Y faces; result `mix(1.0, lit * 0.2, strength)`, so a fully shadowed surface keeps
~0.256 of the key light. Only light 0 is shadowed (`:192-194`).

**Two cross-file invariants nothing enforces**, worth knowing before touching
either side: `SHADOW_NEAR/FAR` feed both the face projection *and*
`FrameUniforms.shadow.xy`, which the shader inverts by hand; and the shadow centre
(`scene.lights[0].pos`, `:1082`) must be the light the shader measures from
(`frame.lights[0].pos.xyz`, `lit.wgsl:121`). The key light is inside the room
(`(0, 4.75, 0)`), which is why a cube was chosen over a single 2D map.

## Instancing

`Instance` records travel in one vertex buffer as six instance-rate attributes, so a
mesh's bodies are one draw call. Staging (`stage_instances`, `:1254-1318`) counts
bodies per shape, gives each non-empty shape a contiguous run, and scatters bodies
into it — instance order is grouped by shape, and each `(mesh, range)` becomes one
`fill`. Room is always instance 0; discs, rings and wireframe shells reuse the same
record type with pipeline-specific `params`.

`params` semantics: same meanings as the original contract, new carrier.

- `lit` room instance: `(14.0, 0.16, 1.0, gridSpacing)`; body instances:
  `(gloss, 0.9, 0.0, 1.0)`. Read at `lit.wgsl:168,179,203,209-211`.
- `flat`: `(alpha*fade, inner, 0, 0)` — the fraction of the disc radius out to
  which it is fully opaque, then a fade to nothing at 1.
- `line`: params unused; the shader no longer declares that attribute.
- `shadow` and `trail` read no instance params at all.

Colour is instance colour × per-vertex colour (`lit.wgsl:103`), which is what lets
one glTF body keep two materials. Normals:
`mat3x3(model[0].xyz, model[1].xyz, model[2].xyz)` — no normal-matrix uniform,
correct because model scales are always uniform.

Face-away culling happens on the CPU before staging: discs with
`facing <= 0.02`, and rings below `RING_MIN_FACING = 0.18` or apparent size `0.04`,
are never uploaded.

## Lighting to reproduce

Unchanged and still exact: normal flipped toward the viewer; hemispheric ambient
`mix(vec3(0.045,0.05,0.062), vec3(0.15,0.16,0.19), 0.5+0.5*N.y)`;
`att = 1/(1 + dist2*0.021)`; Blinn-Phong spec `color*pow(ndh, gloss)*ndl*att` then
`* specStrength`; rim `pow(1-max(dot(N,V),0),3) * 0.055 * vec3(0.65,0.78,1.0)`;
room floor/ceiling mix and grid; tonemap
`pow(1 - exp(-max(c,0)*1.25), vec3(0.92))` with alpha forced to 1. Colours and
intensities: key `[1.45,1.32,1.12]`, fill `[0.26,0.38,0.58]`, accent
`[0.72,0.24,0.55]`, wall `[0.2,0.22,0.25]`, floor `[0.13,0.14,0.16]`, ceiling
`[0.16,0.17,0.19]`.

Additions worth holding to: the shadow term above, and one gloss scale —
`PRIMITIVE_GLOSS` spans 15..92 (`geom.rs:120`) and a glTF roughness maps to
`8 + (1-roughness)² * 88` (`import.rs:981`, in `parse_gltf`), so any new default belongs in that
range. `import::DEFAULT_GLOSS = 40.0` is the middle of it, and every caller reads it
from there: an import with no material is not each caller's aesthetic choice to make.
The old value was 0.35, which is a roughness fraction wearing an exponent's name, and
`pow(ndh, 0.35)` is ≈1 for every half-vector — a specular wash across the whole
surface rather than a highlight. It outlived its own fix once, because two copies
existed: the app moved to 40.0 and the probe kept 0.35, so the imported objects in the
committed screenshots stayed waxy after the code was already right.

## Simulation — `src/sim.rs`, `src/hull.rs`

```rust
#[repr(C)] #[derive(Copy, Clone, Debug)]
pub struct Body { pub shape: usize, pub scale: f32, pub radius: f32, pub mass: f32,
                  pub pos: Vec3, pub vel: Vec3, pub rot: Mat4, pub ang_vel: Vec3 }

pub struct Impact { pub pos: Vec3, pub n: Vec3, pub color: [f32; 3],
                    pub base_radius: f32, pub age: f32, pub life: f32 }

pub struct World {
    // room and contents
    half_x, half_y, half_z: f32,          // 7.2 / 5.4 / 7.2
    bodies: Vec<Body>, impacts: Vec<Impact>,
    // tuning
    gravity: f32 /* -17 */, restitution /* 0.84 */, drag /* 0.06 */, collide: bool,
    wall_friction /* 0.16 */, pair_friction /* 0.12 */, spin_coupling /* 0.45 */,
    // per-shape side tables, addressed by Body::shape
    colors: Vec<[f32; 3]>,
    colliders: Vec<Option<Arc<Hull>>>,    // NEW — absent or None entry ⇒ sphere
    hulls: bool,                          // NEW — master switch, default true
}
```

Substepping is unchanged: `step(dt)` runs at most `MAX_SUBSTEPS = 4` ticks of
`SUBSTEP_TARGET = 1/120 s`, and the f64 snapping in `substep_count` (`:784-797`) is
still load-bearing for reproducibility. So are `MAX_SPIN = 9.0`,
`WALL_REST_SPEED = 0.55`, `PAIR_REST_SPEED = 0.5`, `WALL_IMPACT_SPEED = 1.1`,
`PAIR_IMPACT_SPEED = 2.4`, and `MAX_IMPACTS = 48` with oldest-first eviction using
`remove`, not `swap_remove`, so ring order stays stable.

**The four classic behaviours are properties of the sphere path.** They remain true
and remain pinned by `tests/sim_invariants.rs` (24 tests, byte-identical since the
physics work landed): friction acts on **contact-point slip**
(`vel + omega × r_contact`), the pair normal impulse pushes the pair *apart*,
contacts below the resting speed stop bouncing so piles settle, and spin comes only
from the tangential part of an impulse — including the JS torque-sign quirk where
both bodies get `+cross(n, J)` even though b's moment arm is `-n * radius`
(`sim.rs:1035-1045`).

**Hull contacts break two of those four as literally worded**, which is why the
sentence above is scoped. A hull contact has a real moment arm, so its *normal*
impulse can spin a body (`:1108-1121`, `:1189`); friction there is a Coulomb cap on
the tangential impulse (`friction * HULL_FRICTION_SCALE = 3.5`) followed by
`ang_vel *= HULL_CONTACT_SPIN_KEEP = 0.985` per contact — a faceted body rocking on
a corner has to shed that energy or it never lies down. `spin_coupling` is unused by
either hull path. That difference is the reason hulls exist: a cube comes to rest
flat on a face (`tests/new_physics.rs:50`), which a sphere never does.

- The choice is per **shape index**, never per body: `hull_in` (`:615-620`) returns
  `None` when `world.hulls` is false or the shape has no registered hull. Two bodies
  of one shape always collide the same way.
- `Hull::from_points` (`hull.rs:90`) is a bounded approximation, not an exact convex
  hull: support points from 96 probe directions (26 axis/diagonal + 70 Fibonacci)
  merged within `VERTEX_MERGE = 1e-4`, so ≤96 vertices; faces found by brute force
  over vertex triples with `PLANE_SLACK = 1e-4`; refused (`None`) when it encloses
  no volume, and the caller keeps the sphere. The result is slightly **inner**, and
  separation uses both hulls' face normals only — no edge-edge separating axis — so
  edge-against-edge can register a touch a hair early. Both compromises are
  deliberate: bounded cost, paid once per shape.
- Hull-vs-sphere contacts walk the sphere centre back onto the planes (up to 6
  relaxation rounds) so a corner or edge contact gets a corner or edge normal
  instead of the nearest face normal (`hull.rs:198-250`).
- `radius_factor` still matters for a hull body: it sets `Body.radius`, which
  `clamp_to_room`, `settle`, `hold` target clamping, `pick` and the broadphase bound
  all use. Consequence: a cube *primitive* with a hull came to rest as a
  0.82-sphere when it was settled, then falls the rest of the way onto its face.
  Imports avoid it because their factor is 1.0.
- No hulls are registered by default — `colliders` starts empty, which is precisely
  why the 24 inherited tests still pass untouched. The app fills slots
  `[1,4,5,6,7,8]` (cube, cone, cylinder, icosahedron, tetrahedron, octahedron) only
  when asked, and never sphere, capsule or torus (`app.rs:71`).

**Broadphase.** Past `BRUTE_FORCE_MAX = 64` bodies the pair loop goes through a
counting-sort grid (`grid_pairs`, `:703-771`): cell size `2×largest radius + 0.05`,
dims clamped to 256, each body scanning 27 cells, emitting `(i, j)` with `j > i`,
each body's list `sort_unstable()`d — so contact order matches the brute-force
order and a seeded replay is preserved. The list is built once per substep, before
that substep's position corrections; the 0.05 margin appears both in the cell size
and in the per-pair reach test precisely to absorb that drift. At or below 64 bodies
the original inline loop runs verbatim (`:561-593`) and, with no hulls registered,
the entire hull machinery is unreachable: the ≤64 path is bit-identical to the
pre-crowd code, same concentric nudge, same two `rng.range(-0.02, 0.02)` draws in
the same order. `settle` has its own copy of the same threshold (`:405`).

**Picking and holding.** `OrbitCamera::ray(x, y, width, height) -> (eye, unit dir)`
(`camera.rs:86`) un-projects through `(P·V)⁻¹`. `World::pick(origin, dir)`
(`sim.rs:305`) returns the nearest body whose *drawn* size —
`max(scale, radius) * 0.95` — the ray passes through, because what is picked is the
picture rather than the collider; hulls are not raycast. `World::hold(index,
target)` (`:326`) is a velocity rewrite, not a constraint solver: clamp the target
inside the room by the body's radius, `vel = (target - pos) * HOLD_STIFFNESS`
(14.0), cap at `HOLD_MAX_SPEED` (34.0), damp `ang_vel *= 0.9`. The app issues
exactly one hold per frame **before** `step`, so the held body then travels the
ordinary integrate/wall/pair path and gravity is added in every substep afterwards.
Releasing needs no code at all: the body keeps the velocity the pointer last gave
it, and that is the throw.

**Rewind.** `World::snapshot() -> Snapshot` and `World::restore(&Snapshot)`
(`:288-300`). A snapshot carries bodies, in-flight impacts and the generator's
position in its stream, and deliberately **not** the tuning, room size, `colors`,
`colliders` or `hulls`: rewinding the room must not undo a slider. Its fields are
private; `bytes()` and `body_count()` are the only accessors. History limits are app
policy — one snapshot per simulated frame, evicted oldest-first past 900 frames or
128 MiB, always keeping at least one (`app.rs:73-76, 211-219`) — and the app pairs
each snapshot with the accent-light clock, which is not in the world either. Trails
and sparks are decoration and never part of a snapshot.

## Imports — `src/import.rs`

`import_path(path) -> Result<Import, ImportError>`, where `Import` carries `mesh`
(normalised, indexed, edges filled, colours when the format had them), `source`
(file stem), `notes`, `dropped_triangles`, `hull: Option<Hull>` and
`gloss: Option<f32>`. The old contract said it "returns a `Mesh`"; it has not done
that since imports gained materials and hulls.

Four rules, all enforced in `finish`, none re-checked by the caller:

1. **One body per file.** Multi-group OBJs, multi-node glTF scenes and whole
   `.blend` files merge into a single mesh: the sim collides bodies as spheres or
   hulls, so splitting an assembly into parts would let them bounce apart.
2. **Normalised to a unit bounding radius**, measured to the farthest **vertex**
   from the box centre — the same measurement `Mesh::max_radius()` makes — *not*
   half the box diagonal. On an irregular model that diagonal is a corner no vertex
   occupies, so scaling by it left the mesh smaller than its collider: an import
   touched the floor in the simulation while floating above it on screen.
3. **Inside the u16 index budget** (`MAX_VERTS = 65_532`). Weld first (position
   **and** normal, so faceted models stay faceted), then, if still over, sample
   whole faces out at an even stride — never cut from the tail, because files are
   written in spatially coherent order and a tail-cut draws half an object while its
   collider still spans the whole thing. `MAX_EDGE_PAIRS = 60_000` caps the
   wireframe; the stride is sized from the corner count, and since welding can only
   lower the vertex count it cannot overshoot.
4. **`edges` synthesised** from the triangles, or the `w` overlay would draw nothing
   for an import.

Everything else is reported rather than done quietly — in `Import::notes` and in the
panel: faces sampled out, degenerate or non-finite faces dropped, normals rebuilt
from the winding, the edge cap, glTF primitives that were points/lines/strips,
textured materials that kept only their base colour, meshes too flat to get a hull.
A silent geometry change counts as a bug.

**Formats.** Read in-process: `.obj` (`tobj`), `.stl` (binary and ASCII, parsed
here; the encoding is decided by the header's count against the file length rather
than by a `solid` prefix, since plenty of binary STLs put the object name in that
free-text field), `.gltf` and `.glb` (the `gltf` crate reads the structure; this
module resolves buffers itself — the GLB binary chunk, `data:` URIs, and plain
sibling `.bin` files, refusing anything absolute or containing `..`).

Converted by an external tool: `.fbx`, `.blend`, `.usd`, `.usda`, `.usdc`,
`.usdz`. No maintained Rust reader exists for any of them — FBX is a binary AST with
mutually incompatible dialects, `.blend` is a self-describing blob whose layout
tracks Blender's DNA version, and USD is a composition system where the geometry you
see may be instanced, inherited across layers, or refined from a proxy, so "read the
mesh" means implementing a resolver. The tool writes a self-contained GLB and the
existing glTF reader takes it from there, so welding, normalising, the budget,
edges and hulls are identical either way. Measured rather than assumed, because
these are the two things that look like bugs the first time you hit them:

| format | recipe | fallback |
| --- | --- | --- |
| `.blend` | `blender -b file.blend --python-expr …` → GLB | none |
| `.fbx` | `blender -b --python-expr` with `wm.fbx_import` → GLB | `assimp export` |
| `.usd`/`.usda`/`.usdc`/`.usdz` | same, with `wm.usd_import` | none |

`blender -b file.fbx` — the argument form that works for `.blend` — **fails**
("File format is not supported"), because FBX is an addon import, not something the
.blend loader reads; the operator has to be called explicitly. `assimp export
modern.blend` **fails** too ("BLENDER magic bytes are missing, couldn't find GZIP
header either"), because assimp's Blender loader predates the format Blender 2.8+
writes.

Tool discovery is `BLENDER`/`ASSIMP` environment variables, then `PATH`, then the
macOS application bundle. Nothing is installed behind the user; a machine with
neither gets an error naming the variables to set. Conversion runs under
`MAX_CONVERT_SECONDS = 120` with a kill-on-timeout (`std::process` has no
wait-with-timeout, so it polls `try_wait`, and kills *and reaps* the child on the
timeout and on a failed wait alike). The output goes into a temp directory created
**exclusively** — `create_dir`, stepping the attempt counter until it succeeds — so a
directory or symlink another local process planted under this name gets skipped rather
than adopted, and it is removed by a `Drop` guard rather than by a call on each exit
path, which is what keeps that promise true after someone adds a `return`. That buys
exclusivity, not secrecy: the converted file is as readable as anything in a shared
temp directory, and nothing in it is worth reading. File paths travel as **environment
variables**, never interpolated into the Python that is executed — a folder name
containing an apostrophe is a syntax error at best and an injection at worst if the
quoting is ever half-finished.

Two bounds apply to a conversion, and neither is the one that gates a file this process
reads: `MAX_FILE_BYTES` never sees a `.blend`, because that file is never read here.
What is bounded is the timeout and **what the converter wrote**. A few megabytes of
`.usdz` — a zip archive — or a `.blend` of procedural objects can come out as many times
that in glTF, so the output's size is checked before a byte of it is read, and a file
over the limit gets an error naming both sizes rather than a glTF reader that cannot
hold it. No later recipe is tried for that case: the second tool would write a file the
same size and cost another start-up to say the same thing. And each recipe starts from
*no output at all*, because they share one path — a Blender killed mid-export at the
timeout would otherwise leave a truncated `.glb` that the following recipe, exiting 0
without writing anything itself, would have credited as its own conversion.

Because conversion costs seconds, the window layer runs it on a worker thread and
collects the answer on a later frame — the same shape it already uses for the file
dialog. Only the convertible formats go that way: an in-process import is
milliseconds, and deferring it would make the panel lie about when something
arrived. `import_path` itself stays synchronous, which is what lets
`probe --import asset.fbx` run headless — the one place a convertible format is asked to
convert on the thread that drew the request, and deliberately so.

Three rules live inside that deferral, and they are the parts most likely to be broken
quietly by a later edit:

- **One conversion at a time.** Anything else a dropped batch queued stays in `pending`,
  in the order it arrived, and starts on a later frame. A dozen Blenders launched by one
  Finder drag would each hold a scene's worth of result bytes, each run to the 120 s
  ceiling, and make each other slower for no gain — Blender's start-up is the cost.
- **The drain never waits, and keeps its order.** `App::take_finished` is a function
  rather than a block inside the frame loop precisely so it can be tested: answers come
  back in queue order rather than tool-finish order (the import palette and the shape
  indices follow it), an in-flight conversion is left alone rather than reported as an
  error, and a worker that died surfaces as a message instead of leaving the panel
  promising *converting…* for the rest of the session.
- **Clear all cancels both halves.** `x` empties `pending` *and* drops the receivers, so
  a file dropped between frames and a conversion still running cannot put a body back
  into the room the viewer just emptied. Dropping a receiver is the whole cancellation:
  the worker's `send` fails and is ignored. The tool already running is **not** stopped —
  `run_capped` has no cancellation hook — so it finishes, holding its CPU and its temp
  directory, and throws the result away.

One residue is stated rather than fixed: **quitting the app during a conversion leaves
the temp directory behind and lets the tool run on**. A `Drop` guard cannot help at
process exit — the worker thread dies with the process, and the name is never reused —
and the orphaned Blender burns a core until its own 120 s budget ends, after the window
is gone. Closing it needs two things neither is in place: a cancel flag `run_capped`
polls between its 50 ms waits, and `App::exiting` joining the worker's `JoinHandle`
after setting it (the handle would join the `converting` tuple, which already carries the
file name). That is a real change to the conversion API — `import_path` grows a way to be
told to stop — so it is left here as the next thing to do rather than done halfway.
Nothing else about the queue is unowned: `x` already drops both the queue and the
answers, which is the case a viewer actually hits.

And a convertible path is checked for existence before a converter is started, so a
mistyped panel entry (`scene.blend` where the file is `scene1.blend`) answers "no such
file" rather than spending a Blender start-up to come back with "cannot convert .blend:
… exited with 1", which reads like the format is beyond the app.

**What the converter tests can and cannot prove.** Three of them live in
`import.rs`. One asks Blender which operators the recipes call and fails if the
answer is no. One writes a cube as `.fbx`, `.blend`, `.usd`, `.usda` and `.usdz`,
reads every one back through its own recipe, and insists on the four rules and a note
naming the tool that ran. One forces the recipe list to assimp alone, because the
fallback row is otherwise unreachable on any machine that has Blender — which is every
machine that develops this, so an unrunnable fallback is a fallback that stops
working. All three skip *out loud* when a tool is missing, and they do: the release
runners have neither Blender nor assimp, so CI never converts anything and the
converter is verified on developer machines and by `probe --import asset.fbx`.

The operator question is worth its odd shape. `hasattr(bpy.ops.wm, name)` and
`.idname()` both answer for any string at all — Blender's operator proxy fabricates a
stub, measured as `WM_OT_definitely_not_an_operator` — so existence is checked by
membership of the category's own operator listing, and the test carries a canary name
that must come back missing, because a probe that cannot fail is worth as much as the
silent skip it replaced: the first version of this guessed `wm.fbx_export`, which
Blender 5.0 does not have, and the one test that used it skipped quietly while
reporting green.

**Per-shape bookkeeping.** Four lists are addressed by `Body::shape` and must not
drift: `App::shapes`, `Gpu::shape_mesh`, `World::colors`, `World::colliders`.
`colliders` may be *shorter* than the others (an absent entry means sphere); the
other three are kept the same length and asserted at `app.rs:803-804`, in `add_import`. The entire
test for "this body is an import" is `body.shape >= PRIMITIVE_NAMES.len()`
(`app.rs:566-568`, as `is_generated`) — which is what lets the count slider and `r` manage only the
generated population while `Clear all` (`x`) truncates all four lists back to the
built-ins and clears history. Imported GPU buffers are truncated away rather than
freed: `Gfx` has no eviction, and a handful of meshes is not worth inventing one
for.

**A buffer invariant the index format implies:** `Gfx::upload_mesh` pads index and
edge lists to `COPY_BUFFER_ALIGNMENT` (4 bytes). u16 entries cost 2 bytes, so an odd
entry count is `2 mod 4` and wgpu rejects the copy outright. Every generated
primitive happens to have an even triangle count, which is how this stayed latent
until an import sampled down to an odd one.

## The probe — `src/bin/probe.rs`

Renders a seeded scene offscreen and reads its own pixels back, so "imported" and
"culled correctly" mean something stronger than "did not error". Flags: `--out`,
`--width/--height`, `--count`, `--seed` (decimal or `0x` hex — the same grammar as
the HUD's seed box, which is what makes a picture reproducible from a number on
screen), `--seconds`, `--front-face cw|ccw|both`, `--keep-rings`/`--no-rings`,
`--import PATH` (repeatable), `--wireframe`, `--gif S`, `--no-hulls`,
`--primitive-hulls`, `--hud`, `--no-shadows`, `--help`. Exit 0 requires exactly one
winding to show every body *and* that winding to be the one
`GfxOptions::default()` builds.

Visibility thresholds: `PIXEL_EPS = 26` per channel, `DISC_COVER = 0.12` of the
body's screen disc, `MIN_SAMPLES = 24`, `VISIBILITY_PASS = 0.85`; `--hud` then wants
the left strip to differ by ≥50% over a 240 px panel. Imported bodies join the world
under test and **never** the empty reference, so an import that fails to draw fails
in the same terms as a body a wall swallowed. `--wireframe` weakens the verdict —
edge pixels are not surfaces — which is why the probe warns when it is on and why it
passes the flag to the reference render as well: the room's outline then cancels out
of the comparison and only the bodies' edges can count.

The probe keeps its own copy of every scene constant — colours, lights, grid, shadow
blobs, the ceiling panel — *on purpose*: a shared constant that is wrong cancels out
in a check that uses it. The one exception is `import::DEFAULT_GLOSS`, and the
exception is deliberate too: a material-less import's gloss is not the probe's
aesthetic to hold independently, and its pictures are published as what the app looks
like. Sharing that default is what makes a wrong value show up in the README instead
of hiding there — which is exactly how the 0.35 survived its own fix.

## Known gaps, stated plainly

Things a future change could break silently, because nothing checks them:

- **Grid vs brute-force equivalence.** Nothing compares the two broadphase paths on
  the same room, or asserts that crossing the 64-body boundary changes nothing. The
  guarantee rests on `grid_pairs`' per-body sort and a comment.
- `settle_with_grid` is reached only implicitly, and is not identical to the ≤64
  `settle` in its degenerate-direction branch.
- **Hull-vs-hull contacts** run only inside an aggregate energy-and-containment
  test; no test pins a specific hull-hull normal, depth, or that a normal impulse
  spins both bodies. `HULL_FRICTION_SCALE` and `HULL_CONTACT_SPIN_KEEP` are private
  and only indirectly pinned.
- Hull impact rings (`base_radius = reach*0.45`, position at the contact point)
  differ from sphere rings and are untested.
- No test drives a **crowd of hull bodies** past the grid threshold.
- `pick`/`hold` are tested in isolation: a once-per-frame hold against 2–4
  substeps, holding while another body pushes, and picking a long thin import are
  all untested.
- Rewind safety when the shape table changed under a snapshot is handled by
  convention (`history.clear()`), not asserted; `Snapshot::bodies` still indexes
  shapes that may be gone.
- **Trails and sparks have no pixel coverage** — the probe passes `lines: &[]` — so
  they are unit-tested only.
- A clip whose encoder is still finishing when the window closes is not joined at
  exit, which can truncate the GIF.
- `pick_surface_format` falls back to `formats[0]`, which would panic on an empty
  capability list. Unreachable in practice, but it is the one place "the pipelines
  must agree with whatever we picked" rests on an unwrap.
- Picking with the mouse, the rewind slider, the file dialog and drag-and-drop are
  covered by tests of their parts, not driven end to end; the release workflow builds
  for Linux and macOS but has never been run.
- **Quitting mid-conversion orphans the tool.** The child runs to its 120 s ceiling with
  no window left, and its temp directory survives, because a `Drop` guard cannot run at
  process exit and the name is never reused. Closing it means a cancel flag `run_capped`
  polls and a `JoinHandle` join from `App::exiting` — see the converter section.
- **Nothing drives a converted import through the window layer end to end.** The queue,
  the drain and the cancellation are tested as functions (`App::take_finished`); the
  conversion itself is tested through `import_path` and through the probe. What no test
  covers is a real drop reaching `pending`, then a panel showing *converting…* while an
  external process runs — that shape is only checked by hand.
