// shadow.wgsl — depth-only pass for one face of the key light's shadow cube.
//
// Runs six times a frame, once per face, each with that face's view-projection
// chosen by a dynamic offset. Bodies are the only casters: the room is the
// shell the shadows land on. There is no fragment stage.
struct Face {
  view_proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> face: Face;

@vertex
fn vs_main(
  @location(0) a_pos: vec3<f32>,
  @location(2) m0: vec4<f32>,
  @location(3) m1: vec4<f32>,
  @location(4) m2: vec4<f32>,
  @location(5) m3: vec4<f32>,
) -> @builtin(position) vec4<f32> {
  let model = mat4x4<f32>(m0, m1, m2, m3);
  return face.view_proj * model * vec4<f32>(a_pos, 1.0);
}
