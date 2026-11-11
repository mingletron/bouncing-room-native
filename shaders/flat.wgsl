// flat.wgsl — port of FLAT_VERT + FLAT_FRAG from the WebGL build's shaders.js:
// soft radial discs (ground shadow blobs, the ceiling light panel).
//
// Per-disc values arrive per instance rather than in a uniform: `i_color.rgb`
// is the disc colour and `i_params = (alpha, inner radius, -, -)`. The pipeline
// is blended and writes no depth.
//
// Kept byte-for-byte identical across lit.wgsl, flat.wgsl, line.wgsl and
// trail.wgsl: one frame bind group serves every pipeline. Members this shader
// never reads still have to be declared, or the ones after them would shift.
struct Light {
  pos: vec4<f32>,
  color: vec4<f32>,
}

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

struct VsOut {
  @location(0) disc: vec2<f32>,
  @location(1) @interpolate(flat) color: vec3<f32>,
  @location(2) @interpolate(flat) params: vec2<f32>,
  @builtin(position) clip: vec4<f32>,
}

@vertex
fn vs_main(
  @location(0) a_pos: vec3<f32>,
  @location(2) m0: vec4<f32>,
  @location(3) m1: vec4<f32>,
  @location(4) m2: vec4<f32>,
  @location(5) m3: vec4<f32>,
  @location(6) i_color: vec4<f32>,
  @location(7) i_params: vec4<f32>,
) -> VsOut {
  var vs_out: VsOut;
  let model = mat4x4<f32>(m0, m1, m2, m3);
  vs_out.disc = a_pos.xz;
  vs_out.color = i_color.rgb;
  vs_out.params = i_params.xy;
  vs_out.clip = frame.proj * frame.view * model * vec4<f32>(a_pos, 1.0);
  return vs_out;
}

@fragment
fn fs_main(varyings: VsOut) -> @location(0) vec4<f32> {
  let d = length(varyings.disc);
  let alpha = varyings.params.x * (1.0 - smoothstep(varyings.params.y, 1.0, d));
  if alpha <= 0.002 {
    discard;
  }
  return vec4<f32>(varyings.color, alpha);
}
