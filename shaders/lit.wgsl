// lit.wgsl — port of LIT_VERT + LIT_FRAG from
// /Users/work/Developer/bouncing-room/src/shaders.js
// (room + primitives: Blinn-Phong with three point lights, hemispheric ambient,
// rim light, room floor/ceiling colours, floor grid and wall seams.)
//
// Structural changes forced by WGSL / the port contract (see
// docs/PORT_CONTRACT.md); the maths is otherwise a line-for-line port:
//
//   * The GLSL's loose uniforms become one frame uniform block (group 0) plus
//     per-instance vertex attributes: i_color.rgb = base colour and
//     i_params = (gloss, specStrength, isRoom, gridSpacing).
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
//   * The GLSL's separate u_floorColor/u_ceilColor live in FrameUniforms.
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

// Layout comes from declaration order alone (naga 30 rejects the contract's
// @size/@offset attributes). FrameUniforms = 288 bytes. Keep this block
// byte-identical in lit.wgsl, flat.wgsl, line.wgsl and trail.wgsl.
//
// There is no per-object uniform any more. Everything that used to live in the
// contract's ObjectUniforms now arrives per *instance* (locations 2..7), which
// is what lets a few thousand bodies cost one draw call per shape instead of
// one per body. The room's floor and ceiling colours, the only per-object
// values with no instance slot, moved into FrameUniforms.
struct FrameUniforms {
  view: mat4x4<f32>,               // byte 0
  proj: mat4x4<f32>,               // byte 64
  lights: array<Light, 3>,         // byte 128, element stride 32
  cam_pos: vec4<f32>,              // byte 224
  floor_color: vec4<f32>,          // byte 240, read only by lit, for the room
  ceil_color: vec4<f32>,           // byte 256, read only by lit, for the room
  // (near, far, strength, bias) of the key light's shadow cube; strength 0
  // switches shadows off. Read only by lit.
  shadow: vec4<f32>,               // byte 272
}

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

// The key light's shadow: a depth cube rendered from the light, compared in the
// fragment shader. A cube rather than a single map because the key is a point
// light inside the room, and bodies shadow the walls as well as the floor.
@group(1) @binding(0)
var shadow_map: texture_depth_cube;
@group(1) @binding(1)
var shadow_sampler: sampler_comparison;

const LIGHT_COUNT: i32 = 3; // key / fill / orbiting accent

struct VsOut {
  @location(0) world: vec3<f32>,
  @location(1) normal: vec3<f32>,
  // base colour: the instance colour times the mesh's vertex colour (white for
  // every mesh that has none, which is all of them except a glTF import)
  @location(2) color: vec3<f32>,
  // (gloss, specStrength, isRoom, gridSpacing), constant across the instance
  @location(3) @interpolate(flat) params: vec4<f32>,
  @builtin(position) clip: vec4<f32>,
}

@vertex
fn vs_main(
  @location(0) a_pos: vec3<f32>,
  @location(1) a_nrm: vec3<f32>,
  @location(2) m0: vec4<f32>,
  @location(3) m1: vec4<f32>,
  @location(4) m2: vec4<f32>,
  @location(5) m3: vec4<f32>,
  @location(6) i_color: vec4<f32>,
  @location(7) i_params: vec4<f32>,
  @location(8) a_color: vec4<f32>,
) -> VsOut {
  var vs_out: VsOut;
  let model = mat4x4<f32>(m0, m1, m2, m3);
  let world = model * vec4<f32>(a_pos, 1.0);
  vs_out.world = world.xyz;
  // mat3(model) by columns; equivalent to the GLSL's u_rot.
  let rot = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
  vs_out.normal = rot * a_nrm;
  vs_out.color = i_color.rgb * a_color.rgb;
  vs_out.params = i_params;
  vs_out.clip = frame.proj * frame.view * world;
  return vs_out;
}

// How much of the key light reaches `world`: 1 lit, 0 shadowed, between at a
// shadow's edge. The cube stores hardware depth for a 90-degree face, and the
// face a direction lands on is the one its largest component points along, so
// the depth to compare against comes from that component alone.
fn key_shadow(world: vec3<f32>, n: vec3<f32>) -> f32 {
  let strength = frame.shadow.z;
  if strength <= 0.0 {
    return 1.0;
  }
  let near = frame.shadow.x;
  let far = frame.shadow.y;
  // Nudged along the normal so a surface does not shadow itself.
  let v = (world + n * 0.035) - frame.lights[0].pos.xyz;
  let a = abs(v);
  let z = max(a.x, max(a.y, a.z));
  let reference = far / (far - near) - (far * near / (far - near)) / z - frame.shadow.w;

  // Five taps: the centre and four around it, on top of the sampler's own
  // 2x2 comparison filter. Enough to take the stair-steps off an edge.
  var up = vec3<f32>(0.0, 1.0, 0.0);
  if a.y > 0.9 * length(v) {
    up = vec3<f32>(1.0, 0.0, 0.0);
  }
  let t1 = normalize(cross(v, up));
  let t2 = normalize(cross(v, t1));
  let spread = z * 0.0035;
  var lit = textureSampleCompareLevel(shadow_map, shadow_sampler, v, reference);
  lit = lit + textureSampleCompareLevel(shadow_map, shadow_sampler, v + (t1 + t2) * spread, reference);
  lit = lit + textureSampleCompareLevel(shadow_map, shadow_sampler, v + (t1 - t2) * spread, reference);
  lit = lit + textureSampleCompareLevel(shadow_map, shadow_sampler, v - (t1 + t2) * spread, reference);
  lit = lit + textureSampleCompareLevel(shadow_map, shadow_sampler, v - (t1 - t2) * spread, reference);
  return mix(1.0, lit * 0.2, strength);
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

  var base = varyings.color;
  if varyings.params.z > 0.5 {
    base = mix(base, frame.floor_color.rgb, step(0.5, n.y));
    base = mix(base, frame.ceil_color.rgb, step(0.5, -n.y));
  }

  // Ambient + a hemisphere tint so downward faces read cooler than upward ones.
  let hemi = 0.5 + 0.5 * n.y;
  var lit = base * mix(vec3<f32>(0.045, 0.05, 0.062), vec3<f32>(0.15, 0.16, 0.19), hemi);

  let shadow = key_shadow(varyings.world, n);
  var specular = vec3<f32>(0.0);
  let shininess = max(varyings.params.x, 1.0);

  // i32 counter with a constant bound, as WebGPU requires for a bounded loop.
  for (var i: i32 = 0; i < LIGHT_COUNT; i = i + 1) {
    let light = frame.lights[i];
    let to_light = light.pos.xyz - varyings.world;
    let dist2 = dot(to_light, to_light);
    let l = to_light * inverseSqrt(max(dist2, 0.0001));

    let ndl = max(dot(n, l), 0.0);
    var atten = 1.0 / (1.0 + dist2 * 0.021);
    // Only the key light casts shadows; the fill and the accent are there to
    // keep the shadowed side readable.
    if i == 0 {
      atten = atten * shadow;
    }

    lit = lit + base * light.color.rgb * ndl * atten;

    let h = normalize(l + v);
    let ndh = max(dot(n, h), 0.0);
    specular = specular + light.color.rgb * pow(ndh, shininess) * ndl * atten;
  }

  lit = lit + specular * varyings.params.y;

  // A touch of rim light keeps silhouette edges of dark shapes readable.
  let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
  lit = lit + rim * 0.055 * vec3<f32>(0.65, 0.78, 1.0);

  if varyings.params.z > 0.5 {
    let dist = length(frame.cam_pos.xyz - varyings.world);
    let spacing = varyings.params.w;
    let line = grid_mask(varyings.world.xz, spacing, 0.008 + dist * 0.0016);
    // Wall seams only read on vertical faces.
    let seam = grid_mask(varyings.world.xy - vec2<f32>(0.0, spacing), spacing, 0.006 + dist * 0.0014)
      * step(0.5, abs(n.x) + abs(n.z));
    lit = mix(lit, lit * 1.55 + vec3<f32>(0.02, 0.024, 0.03), max(line, seam * 0.5));
  }

  return vec4<f32>(tonemap(lit), 1.0);
}
