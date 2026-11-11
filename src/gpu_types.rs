//! CPU-side mirrors of the WGSL uniform structs, plus the transform helpers the
//! renderer needs.
//!
//! The byte layouts here are load-bearing: they are asserted against the sizes
//! naga computes for `shaders/*.wgsl` (see `tests/gpu_layout.rs`, which compiles
//! the shaders and compares). If a field is added, reordered or padded here, it
//! has to be added there in the same position or the render reads garbage.

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

/// Per-frame uniforms: group 0, binding 0. 240 bytes.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FrameUniforms {
    pub view: Mat4,
    pub proj: Mat4,
    pub lights: [Light; LIGHT_COUNT],
    pub cam_pos: Vec4,
}

/// Per-object uniforms: group 1, binding 0. 128 bytes.
///
/// Interpretation depends on the pipeline, which is what lets all three
/// pipelines share one bind group layout:
///
/// * `lit`: `color.rgb` is the base colour, `params = (gloss, spec strength,
///   is room, grid spacing)`, and the two colour fields apply only to the room.
/// * `line`: `color.rgba` is the line colour including alpha; `params` unused.
/// * `flat`: `color.rgb` is the disc colour, `params = (alpha, inner radius)`.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ObjectUniforms {
    pub model: Mat4,
    pub color: Vec4,
    pub params: Vec4,
    pub floor_color: Vec4,
    pub ceil_color: Vec4,
}

impl Default for ObjectUniforms {
    fn default() -> Self {
        ObjectUniforms {
            model: Mat4::IDENTITY,
            color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            params: Vec4::ZERO,
            floor_color: Vec4::ZERO,
            ceil_color: Vec4::ZERO,
        }
    }
}

/// Uniform buffer slots must be a multiple of the device's
/// `min_uniform_buffer_offset_alignment` to be usable with dynamic offsets, and
/// 256 covers every backend we target. Kept as a constant so the object ring
/// buffer and the per-draw offsets stay in step.
pub const UNIFORM_SLOT_BYTES: usize = 256;

const _: () = assert!(std::mem::size_of::<FrameUniforms>() == 240);
const _: () = assert!(std::mem::size_of::<ObjectUniforms>() == 128);
const _: () = assert!(std::mem::size_of::<Light>() == 32);
const _: () = assert!(UNIFORM_SLOT_BYTES >= std::mem::size_of::<ObjectUniforms>());

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
