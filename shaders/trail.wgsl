// trail.wgsl — world-space line segments with a colour at each end: motion
// trails and impact sparks. Not part of the WebGL build; the geometry is
// rebuilt on the CPU every frame, so there is no model matrix and no instance.
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
  @location(0) color: vec4<f32>,
  @builtin(position) clip: vec4<f32>,
}

@vertex
fn vs_main(@location(0) a_pos: vec3<f32>, @location(1) a_color: vec4<f32>) -> VsOut {
  var vs_out: VsOut;
  vs_out.color = a_color;
  vs_out.clip = frame.proj * frame.view * vec4<f32>(a_pos, 1.0);
  return vs_out;
}

@fragment
fn fs_main(varyings: VsOut) -> @location(0) vec4<f32> {
  return varyings.color;
}
