// flat.wgsl — port of FLAT_VERT + FLAT_FRAG from
// /Users/work/Developer/bouncing-room/src/shaders.js
// (soft radial discs: shadow blobs and the ceiling light panel.)
//
// Structural changes forced by WGSL / the port contract (docs/PORT_CONTRACT.md):
//
//   * Loose uniforms become the shared bind groups: FrameUniforms in group(0)
//     binding(0) (declared because the pipeline layout is shared, and because
//     the vertex stage needs view/proj) and ObjectUniforms in group(1)
//     binding(0), where obj.color.rgb is the disc colour, obj.params.x is alpha
//     and obj.params.y is the inner radius — the fraction of the radius out to
//     which the disc is fully opaque, before it fades to nothing at 1.
//     Nothing here reads cam_pos or the lights, but they stay in the struct so
//     the byte offsets match the contract, and group(0) is bound for this
//     pipeline too because the contract gives all three pipelines one layout.
//   * The `varying vec2 v_disc` becomes the @location(0) member of VsOut. It is
//     still the *local* xz of the mesh (quad_mesh is the unit XZ quad), so the
//     model matrix's scale controls the world radius exactly as before.
//   * Only @location(0) (position) is read in the vertex stage, as in the GLSL.
//     The shared 24-byte vertex layout also supplies a normal at @location(1);
//     WebGPU allows a pipeline attribute the shader does not consume.
//   * GLSL's reversed smoothstep — smoothstep(1.0, inner, d), with edge0 >
//     edge1 — is undefined per the WGSL spec (and per the SPIR-V/Metal builtins
//     this compiles down to). The mathematically identical, well-defined form
//     `1.0 - smoothstep(inner, 1.0, d)` is used instead; h(1-t) == 1-h(t) for
//     the Hermite curve, so the ramp is unchanged up to rounding. inner must
//     stay < 1.0, which the JS already guarantees (0.15 for blobs, 0.45/0.35
//     for the panel).
//   * `discard` exists in WGSL as a statement, unchanged; `gl_FragColor`
//     becomes an @location(0) return value; there is no `precision` qualifier.
//   * The pipeline is blended and depth-write-off (see the contract's pipeline
//     table); nothing about that changes the shader source.

// Kept byte-for-byte identical to lit.wgsl (contract: one bind-group layout for
// all three pipelines). `lights` is never read here, but it sits between proj
// and cam_pos, so dropping it would shift cam_pos and break the layout.
// The contract's @size/@offset attributes are rejected by naga 30 as unknown
// attributes, so layout comes from declaration order alone: FrameUniforms = 240
// bytes, ObjectUniforms = 128 bytes (widened by the integrator from the
// contract's 112 so floor/ceiling colours are uniforms; no member is dead).
// Keep this block byte-identical to the one in lit.wgsl and line.wgsl.
struct Light {
  pos: vec4<f32>,
  color: vec4<f32>,
}

struct FrameUniforms {
  view: mat4x4<f32>,      // byte 0
  proj: mat4x4<f32>,      // byte 64
  lights: array<Light, 3>, // byte 128, element stride 32
  cam_pos: vec4<f32>,     // byte 224
}

struct ObjectUniforms {
  model: mat4x4<f32>,      // byte 0
  color: vec4<f32>,        // byte 64
  params: vec4<f32>,       // byte 80
  floor_color: vec4<f32>,  // byte 96, read only by lit
  ceil_color: vec4<f32>,   // byte 112, read only by lit
}

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

@group(1) @binding(0)
var<uniform> obj: ObjectUniforms;

struct VsOut {
  @location(0) disc: vec2<f32>,
  @builtin(position) clip: vec4<f32>,
}

@vertex
fn vs_main(@location(0) a_pos: vec3<f32>) -> VsOut {
  var vs_out: VsOut;
  vs_out.disc = a_pos.xz;
  vs_out.clip = frame.proj * frame.view * obj.model * vec4<f32>(a_pos, 1.0);
  return vs_out;
}

@fragment
fn fs_main(varyings: VsOut) -> @location(0) vec4<f32> {
  let d = length(varyings.disc);
  let alpha = obj.params.x * (1.0 - smoothstep(obj.params.y, 1.0, d));
  if alpha <= 0.002 {
    discard;
  }
  return vec4<f32>(obj.color.rgb, alpha);
}
