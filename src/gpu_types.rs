//! CPU-side mirrors of the WGSL uniform structs, plus the transform helpers the
//! renderer needs.
//!
//! The byte layouts here are load-bearing, and naga 30 rejects `@offset`/`@size`, so
//! layout in the WGSL copies follows declaration order alone. That makes one mistake
//! the real hazard: a field added, deleted, reordered or retyped in one copy but not
//! the others. Two mechanisms cover it.
//!
//! 1. The asserts at the bottom of this file pin **sizes and every field offset**,
//!    using `offset_of!`. They catch drift inside this file — including the swap of two
//!    same-typed fields, which no size check can see.
//! 2. `tests/gpu_layout.rs` parses each shader's copy out of its text, computes the
//!    offsets with WGSL's alignment rules, and compares both the member list and those
//!    offsets against `offset_of!` values from these structs. So the shaders are checked
//!    against Rust's actual layout, not against a hand-written transcription of it.
//!
//! What neither can do: check that a *name* means the same thing on both sides
//! (`shadow` here and `shadow` in a shader are only the same idea if someone kept them
//! that way), or see a shader struct that shrank below these sizes — that still
//! validates, because the bind group declares only `min_binding_size`, and the render
//! reads the wrong fields silently. And the attribute tables in `gfx.rs` restate
//! `Instance`/`LineVertex` offsets by hand, with nothing comparing the two: change an
//! offset in this file and its assert fires, but the copy in `gfx.rs` moves only because
//! whoever fixed the assert remembered. If a field is added, reordered or padded here,
//! it has to be added in the same position in every shader copy that mirrors the struct.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3, Vec4};

/// Number of point lights: key, fill and the orbiting accent.
pub const LIGHT_COUNT: usize = 3;

/// `Light` in the WGSL: 32 bytes, position with a padding slot and a colour.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub struct Light {
    pub pos: Vec4,
    pub color: Vec4,
}

impl Light {
    pub fn new(pos: Vec3, color: Vec3) -> Self {
        Light {
            pos: Vec4::new(pos.x, pos.y, pos.z, 0.0),
            color: Vec4::new(color.x, color.y, color.z, 1.0),
        }
    }
}

/// Per-frame uniforms: group 0, binding 0. 288 bytes.
///
/// The room's floor and ceiling colours live here rather than with the room's
/// instance because they are the only per-object values an instance has no slot
/// for, and there is only ever one room.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FrameUniforms {
    pub view: Mat4,
    pub proj: Mat4,
    pub lights: [Light; LIGHT_COUNT],
    pub cam_pos: Vec4,
    pub floor_color: Vec4,
    pub ceil_color: Vec4,
    /// `(near, far, strength, bias)` of the key light's shadow cube. A strength
    /// of 0 switches shadows off in the shader.
    pub shadow: Vec4,
}

/// One drawn thing: a body, the room, a disc, a ring. 96 bytes, read by the
/// shaders as per-instance vertex attributes at locations 2 to 7.
///
/// This replaced the contract's per-object uniform block. With a uniform per
/// body every body is its own draw call; with an instance per body, all the
/// bodies of one shape are a single call, which is what lets the room hold
/// thousands of them.
///
/// Interpretation depends on the pipeline:
///
/// * `lit`: `color.rgb` is the base colour and `params = (gloss, spec strength,
///   is room, grid spacing)`.
/// * `line`: `color.rgba` is the line colour including alpha; `params` unused.
/// * `flat`: `color.rgb` is the disc colour, `params = (alpha, inner radius)`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Instance {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
    pub params: [f32; 4],
}

impl Instance {
    pub fn new(model: Mat4, color: Vec4, params: Vec4) -> Self {
        Instance {
            model: model.to_cols_array_2d(),
            color: color.to_array(),
            params: params.to_array(),
        }
    }
}

/// One end of a world-space line segment with its own colour: what motion
/// trails and impact sparks are made of. 28 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<FrameUniforms>() == 288);
const _: () = assert!(std::mem::size_of::<Instance>() == 96);
const _: () = assert!(std::mem::size_of::<LineVertex>() == 28);
const _: () = assert!(std::mem::size_of::<Light>() == 32);

// Offsets too, because sizes alone cannot pin them. Swap `floor_color` and
// `ceil_color`, or `cam_pos` and `shadow` — pairs of the same type — and every size
// above still holds, this file still compiles, `tests/gpu_layout.rs` still passes, and
// the shaders spend the frame reading the camera position out of the shadow block.
// WGSL lays out by declaration order alone, the four shader copies were never touched,
// and nothing above notices: the only thing that changed is which slot holds what.
//
// `offset_of!` is the one way to ask a type where its own fields sit, so every field
// position the rest of the renderer depends on is pinned here — including the two
// vertex records, whose byte offsets `gfx.rs` repeats by hand in its attribute tables.
// `tests/gpu_layout.rs` compares each member parsed out of a shader against these same
// numbers, which is what makes the shaders checked against Rust's layout rather than
// against a third transcription of it.
const _: () = assert!(std::mem::offset_of!(FrameUniforms, view) == 0);
const _: () = assert!(std::mem::offset_of!(FrameUniforms, proj) == 64);
const _: () = assert!(std::mem::offset_of!(FrameUniforms, lights) == 128);
const _: () = assert!(std::mem::offset_of!(FrameUniforms, cam_pos) == 224);
const _: () = assert!(std::mem::offset_of!(FrameUniforms, floor_color) == 240);
const _: () = assert!(std::mem::offset_of!(FrameUniforms, ceil_color) == 256);
const _: () = assert!(std::mem::offset_of!(FrameUniforms, shadow) == 272);
const _: () = assert!(std::mem::offset_of!(Light, pos) == 0);
const _: () = assert!(std::mem::offset_of!(Light, color) == 16);
const _: () = assert!(std::mem::offset_of!(Instance, model) == 0);
const _: () = assert!(std::mem::offset_of!(Instance, color) == 64);
const _: () = assert!(std::mem::offset_of!(Instance, params) == 80);
const _: () = assert!(std::mem::offset_of!(LineVertex, pos) == 0);
const _: () = assert!(std::mem::offset_of!(LineVertex, color) == 12);

/// Model matrix for a body: translate, then apply the body's orthonormal
/// rotation, then a uniform scale. Mirrors `modelMatrix()` in the WebGL build,
/// and the uniform scale is what lets the shader use `mat3(model)` as the
/// normal matrix.
pub fn model_matrix(pos: Vec3, rot: Mat4, scale: f32) -> Mat4 {
    Mat4::from_translation(pos) * Mat4::from_scale(Vec3::splat(scale)) * rot
}

/// Maps the unit XZ quad in [-1,1] onto a disc lying in the plane whose normal
/// is `n`, nudged `lift` units along that normal so it does not z-fight the
/// surface it sits on. Port of `discMatrix()` from math.js.
pub fn disc_matrix(center: Vec3, n: Vec3, radius: f32, lift: f32) -> Mat4 {
    let up = if n.abs().y > 0.9 { Vec3::X } else { Vec3::Y };
    let mut t1 = n.cross(up);
    if t1.length_squared() < 1e-12 {
        t1 = Vec3::X;
    } else {
        t1 = t1.normalize();
    }
    let t2 = n.cross(t1);

    // Columns are the two in-plane axes and the normal, so the local X and Z of
    // the unit quad land in the plane and local Y (always 0 on that quad) is
    // irrelevant. (t1, t2, n) is right-handed, so the quad's winding is kept.
    Mat4::from_cols(
        (t1 * radius).extend(0.0),
        n.extend(0.0),
        (t2 * radius).extend(0.0),
        (center + n * lift).extend(1.0),
    )
}

/// `mat3` of a rotation matrix, for the normal matrix when a model matrix is
/// not wanted (the room draws with an identity model).
pub fn normal_matrix(rot: Mat4) -> Mat3 {
    Mat3::from_mat4(rot)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The basis `disc_matrix` happens to pick is arbitrary — all that matters is
    /// that it spans the plane, because the unit quad's in-plane axes are its
    /// local X and Z.
    fn assert_spans_the_plane(m: Mat4, n: Vec3, radius: f32) {
        let a = m.transform_vector3(Vec3::X);
        let b = m.transform_vector3(Vec3::Z);
        for axis in [a, b] {
            assert!(
                (axis.length() - radius).abs() < 1e-4,
                "in-plane axis {axis:?} must have length {radius}"
            );
            assert!(
                axis.dot(n).abs() < 1e-4,
                "in-plane axis {axis:?} must be perpendicular to the normal {n:?}"
            );
        }
        assert!(
            a.dot(b).abs() < 1e-4,
            "the two in-plane axes must be orthogonal: {a:?} vs {b:?}"
        );
        // x n b is along +n for the floor case below, i.e. the quad faces the
        // direction the camera has to be on to see it.
        assert!(a.cross(b).dot(n) > 0.0, "the disc must face +n, not -n");
    }

    #[test]
    fn disc_matrix_lays_a_unit_quad_flat_on_the_floor() {
        let m = disc_matrix(Vec3::ZERO, Vec3::Y, 2.0, 0.004);
        assert_spans_the_plane(m, Vec3::Y, 2.0);
        let centre = m.transform_point3(Vec3::ZERO);
        assert!(
            (centre - Vec3::new(0.0, 0.004, 0.0)).length() < 1e-5,
            "the lift must move the disc along the normal, got {centre:?}"
        );
        // Corners of the unit quad reach radius*sqrt(2) out, at 45 degrees.
        let corner = m.transform_point3(Vec3::new(1.0, 0.0, 1.0));
        let flat = Vec3::new(corner.x, 0.0, corner.z);
        let reach = 2.0 * std::f32::consts::SQRT_2;
        assert!((flat.length() - reach).abs() < 1e-4, "corner {corner:?}, expected reach {reach}");
    }

    #[test]
    fn disc_matrix_for_a_wall_is_perpendicular_to_it() {
        let m = disc_matrix(Vec3::new(3.0, 1.0, 0.0), Vec3::X, 1.0, 0.0);
        assert_spans_the_plane(m, Vec3::X, 1.0);
        let centre = m.transform_point3(Vec3::ZERO);
        assert!(
            (centre - Vec3::new(3.0, 1.0, 0.0)).length() < 1e-5,
            "the centre must land on the wall plane, got {centre:?}"
        );
    }

    #[test]
    fn model_matrix_scales_rotates_then_translates() {
        let m = model_matrix(Vec3::new(1.0, 2.0, 3.0), Mat4::IDENTITY, 2.0);
        assert!((m.transform_point3(Vec3::X) - Vec3::new(3.0, 2.0, 3.0)).length() < 1e-6);

        // The rotation sits between the scale and the translation.
        let quarter_turn = Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let m = model_matrix(Vec3::ZERO, quarter_turn, 2.0);
        let world = m.transform_point3(Vec3::X);
        assert!((world - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-5, "{world:?}");
    }
}
