// lit.wgsl — port of LIT_VERT + LIT_FRAG from
// /Users/work/Developer/bouncing-room/src/shaders.js
// (room + primitives: Blinn-Phong with three point lights, hemispheric ambient,
// rim light, room floor/ceiling colours, floor grid and wall seams.)
//
// Structural changes forced by WGSL / the port contract (see
// docs/PORT_CONTRACT.md); the maths is otherwise a line-for-line port:
//
//   * The GLSL's loose uniforms become the contract's two bind groups, both in
//     group(0)/group(1) binding(0): FrameUniforms (240 bytes) and
//     ObjectUniforms (112 bytes). For this pipeline
//     obj.color.rgb = base colour and
//     obj.params = (gloss, specStrength, isRoom, gridSpacing). The contract's
//     @size/@offset annotations do not parse in naga 30 (see the note at the
//     struct declarations below), so the byte layout is spelled out with an
//     explicit pad member and the sizes come out as the contract requires.
//   * GLSL supplied a precomputed u_rot mat3 from the CPU. There is no
//     normal-matrix uniform in the contract, so the rotation part is taken from
//     the model matrix's columns. Model scales are uniform, and the fragment
//     shader normalizes the interpolated normal, so the scale cancels out —
//     same result as u_rot * a_nrm.
//   * Loose `varying`s become a VsOut struct passed from @vertex to @fragment.
//     WebGPU depth is already 0..1 (the CPU projection does it), so there is no
//     z remap here, and none was added.
//   * WGSL has no implicit scalar-to-vector broadcast, so the vector/scalar
//     mixes the GLSL wrote as bare scalars (- 0.5, vec2(x, 0.0)) are spelled
//     out with explicit vec constructors here. Scalar * / on vectors *is*
//     allowed in WGSL, so those were left alone.
//   * `inversesqrt` -> `inverseSqrt`, `gl_FragColor` -> an @location(0) return
//     value, and `precision` qualifiers simply do not exist in WGSL.
//   * The GLSL's separate u_floorColor/u_ceilColor live in ObjectUniforms as
//     floor_color/ceil_color (see the struct note below).
//   * Still no derivative functions (dpdx/dpdy/fwidth) anywhere — grid
//     antialiasing stays distance-scaled, exactly as in the GLSL, which is what
//     kept the GLSL working under WebGL 1.
//
// Entry points are vs_main / fs_main; the contract does not name them, so the
// Rust side must either use these names or pass entry_point explicitly.

struct Light {
  pos: vec4<f32>,   // xyz = position, w unused (kept 0)
  color: vec4<f32>, // rgb = colour * intensity, a unused (kept 1)
}

// The contract writes these structs with @size/@offset attributes. naga 30 (the
// WGSL compiler inside wgpu 30) rejects both: `@size` on a struct declaration
// and `@offset` on a member are "unknown attribute" at parse time. So the byte
// layout the contract asks for is reached the only legal way — declaration
// order plus one explicit padding member — and the sizes come out exactly:
// FrameUniforms = 240 bytes (view 0, proj 64, lights 128, cam_pos 224),
// ObjectUniforms = 128 bytes. The integrator widened that struct from the
// contract's 112 rather than padding it, so that the room's floor and ceiling
// colours are uniforms here instead of compiled-in constants — that keeps them
// tunable from Rust, as they are in the WebGL build. No member is dead.
struct FrameUniforms {
  view: mat4x4<f32>,               // byte 0
  proj: mat4x4<f32>,               // byte 64
  lights: array<Light, 3>,         // byte 128, element stride 32
  cam_pos: vec4<f32>,              // byte 224
}

struct ObjectUniforms {
  model: mat4x4<f32>,              // byte 0
  color: vec4<f32>,                // byte 64
  params: vec4<f32>,               // byte 80
  floor_color: vec4<f32>,          // byte 96, read only when isRoom
  ceil_color: vec4<f32>,           // byte 112, read only when isRoom
}

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

@group(1) @binding(0)
var<uniform> obj: ObjectUniforms;

const LIGHT_COUNT: i32 = 3; // key / fill / orbiting accent

struct VsOut {
  @location(0) world: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @builtin(position) clip: vec4<f32>,
}

@vertex
fn vs_main(
  @location(0) a_pos: vec3<f32>,
  @location(1) a_nrm: vec3<f32>,
) -> VsOut {
  var vs_out: VsOut;
  let world = obj.model * vec4<f32>(a_pos, 1.0);
  vs_out.world = world.xyz;
  // mat3(model) by columns; equivalent to the GLSL's u_rot.
  let rot = mat3x3<f32>(obj.model[0].xyz, obj.model[1].xyz, obj.model[2].xyz);
  vs_out.normal = rot * a_nrm;
  vs_out.clip = frame.proj * frame.view * world;
  return vs_out;
}

// Distance between p and the nearest grid line on either axis, measured in
// world units, then antialiased over `width` world units. Identical to gridMask
// in LIT_FRAG; the -0.5 shifts half a cell so lines land on cell boundaries.
fn grid_mask(p: vec2<f32>, spacing: f32, width: f32) -> f32 {
  let half = vec2<f32>(0.5);
  let g = abs(fract(p / spacing - half) - half) * spacing;
  let d = min(g.x, g.y);
  return 1.0 - smoothstep(0.0, width, d);
}

fn tonemap(c: vec3<f32>) -> vec3<f32> {
  var t = max(c, vec3<f32>(0.0));
  t = vec3<f32>(1.0) - exp(-t * 1.25);
  return pow(t, vec3<f32>(0.92));
}

@fragment
fn fs_main(varyings: VsOut) -> @location(0) vec4<f32> {
  var n = normalize(varyings.normal);
  let v = normalize(frame.cam_pos.xyz - varyings.world);
  if dot(n, v) < 0.0 {
    n = -n; // never light a surface from behind (culling is off for most objects)
  }

  var base = obj.color.rgb;
  if obj.params.z > 0.5 {
    base = mix(base, obj.floor_color.rgb, step(0.5, n.y));
    base = mix(base, obj.ceil_color.rgb, step(0.5, -n.y));
  }

  // Ambient + a hemisphere tint so downward faces read cooler than upward ones.
  let hemi = 0.5 + 0.5 * n.y;
  var lit = base * mix(vec3<f32>(0.045, 0.05, 0.062), vec3<f32>(0.15, 0.16, 0.19), hemi);

  var specular = vec3<f32>(0.0);
  let shininess = max(obj.params.x, 1.0);

  // i32 counter with a constant bound, as WebGPU requires for a bounded loop.
  for (var i: i32 = 0; i < LIGHT_COUNT; i = i + 1) {
    let light = frame.lights[i];
    let to_light = light.pos.xyz - varyings.world;
    let dist2 = dot(to_light, to_light);
    let l = to_light * inverseSqrt(max(dist2, 0.0001));

    let ndl = max(dot(n, l), 0.0);
    let atten = 1.0 / (1.0 + dist2 * 0.021);

    lit = lit + base * light.color.rgb * ndl * atten;

    let h = normalize(l + v);
    let ndh = max(dot(n, h), 0.0);
    specular = specular + light.color.rgb * pow(ndh, shininess) * ndl * atten;
  }

  lit = lit + specular * obj.params.y;

  // A touch of rim light keeps silhouette edges of dark shapes readable.
  let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
  lit = lit + rim * 0.055 * vec3<f32>(0.65, 0.78, 1.0);

  if obj.params.z > 0.5 {
    let dist = length(frame.cam_pos.xyz - varyings.world);
    let spacing = obj.params.w;
    let line = grid_mask(varyings.world.xz, spacing, 0.008 + dist * 0.0016);
    // Wall seams only read on vertical faces.
    let seam = grid_mask(varyings.world.xy - vec2<f32>(0.0, spacing), spacing, 0.006 + dist * 0.0014)
      * step(0.5, abs(n.x) + abs(n.z));
    lit = mix(lit, lit * 1.55 + vec3<f32>(0.02, 0.024, 0.03), max(line, seam * 0.5));
  }

  return vec4<f32>(tonemap(lit), 1.0);
}
