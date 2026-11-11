// line.wgsl — port of LINE_VERT + LINE_FRAG from
// /Users/work/Developer/bouncing-room/src/shaders.js
// (wireframes, the room outline, impact rings: a constant colour with alpha.)
//
// Structural changes forced by WGSL / the port contract (docs/PORT_CONTRACT.md):
//
//   * The GLSL's u_model/u_view/u_proj/u_color uniforms become the contract's
//     two bind groups: FrameUniforms (240 bytes) in group(0) binding(0) and
//     ObjectUniforms (112 bytes) in group(1) binding(0). For this pipeline
//     obj.color.rgba is the line colour including alpha; obj.params is unused.
//     The struct declarations are identical in all three files so the three
//     pipelines can share one bind-group layout and one bind group each.
//   * There are no varyings at all, exactly as in the GLSL, so the vertex stage
//     returns @builtin(position) directly instead of a VsOut struct. (The
//     contract mentions the clip position could drive a size falloff; the GLSL
//     does not use one, so nothing was invented here.)
//   * Only @location(0) (position) is consumed, as in the GLSL. The shared
//     24-byte vertex layout also supplies a normal at @location(1); WebGPU lets
//     a pipeline carry an attribute the shader never reads.
//   * `gl_FragColor` becomes an @location(0) return value; the `precision
//     mediump float` line has no WGSL equivalent and was dropped.
//   * Note for the integrator (not a source issue): WebGPU has no line width —
//     all lines are 1 pixel. The WebGL build never called lineWidth either, so
//     both draw 1px lines and the look carries over unchanged; thicker rings
//     would need geometry, not a state change.
//   * Depth range is 0..1 in WebGPU: the CPU-side projection handles that, and
//     the shader does no z remapping.

struct Light {
  pos: vec4<f32>,
  color: vec4<f32>,
}

// Kept byte-for-byte identical to lit.wgsl (contract: one bind-group layout for
// all three pipelines). `lights`, `cam_pos` and `params` are never read here,
// but they keep the later members at the byte offsets the contract specifies.
// The contract's @size/@offset attributes are rejected by naga 30 as unknown
// attributes, so layout comes from declaration order alone: FrameUniforms = 240
// bytes, ObjectUniforms = 128 bytes (widened by the integrator from the
// contract's 112 so floor/ceiling colours are uniforms; no member is dead).
// Keep this block byte-identical to the ones in lit.wgsl and flat.wgsl.
struct FrameUniforms {
  view: mat4x4<f32>,       // byte 0
  proj: mat4x4<f32>,       // byte 64
  lights: array<Light, 3>, // byte 128, element stride 32
  cam_pos: vec4<f32>,      // byte 224
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

@vertex
fn vs_main(@location(0) a_pos: vec3<f32>) -> @builtin(position) vec4<f32> {
  return frame.proj * frame.view * obj.model * vec4<f32>(a_pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  return obj.color;
}
