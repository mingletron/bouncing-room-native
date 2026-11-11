//! Orbit camera: a port of `src/controls.js`, minus the DOM.
//!
//! The pointer/wheel plumbing lives in `app.rs`; everything here is pure math,
//! which is what lets it be unit tested without a window. Pitch is clamped so
//! the up vector never flips when the camera passes over the room, and the
//! projection is the WebGPU flavour (`camera::rh::proj::directx`, depth 0..1,
//! Y-up) rather than the GL one the WebGL build used.
//!
//! glam 0.34 moved both out of `Mat4` associated functions and into the
//! `glam::camera` module, split by handedness and by NDC convention — the
//! contract's `Mat4::perspective_rh` is now `rh::proj::directx::perspective`,
//! and its `Mat4::look_at_rh` is `rh::view::look_at_mat4`.

use glam::{Mat4, Vec3};

pub const FOV_Y: f32 = 52.0f32.to_radians();
pub const NEAR: f32 = 0.1;
pub const FAR: f32 = 400.0;

#[derive(Copy, Clone, Debug)]
pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
    pub auto_orbit: bool,
    pub auto_speed: f32,
    pub min_distance: f32,
    pub max_distance: f32,
    pub min_pitch: f32,
    pub max_pitch: f32,
    /// Set while a pointer drag is active; suppresses auto-orbit.
    pub dragging: bool,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        OrbitCamera {
            yaw: 0.6,
            pitch: 0.42,
            distance: 18.0,
            target: Vec3::new(0.0, 5.4 * 0.45, 0.0),
            auto_orbit: true,
            auto_speed: 0.11,
            min_distance: 4.0,
            max_distance: 46.0,
            min_pitch: -0.22,
            max_pitch: 1.4,
            dragging: false,
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let cp = self.pitch.cos();
        self.target
            + Vec3::new(
                self.distance * cp * self.yaw.sin(),
                self.distance * self.pitch.sin(),
                self.distance * cp * self.yaw.cos(),
            )
    }

    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    pub fn projection(&self, width: u32, height: u32) -> Mat4 {
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        // WebGPU NDC depth is 0..1 with Y up, which is glam's `directx` flavour.
        // The `opengl` one would put geometry outside the depth range, and
        // `vulkan`'s would flip Y.
        glam::camera::rh::proj::directx::perspective(FOV_Y, aspect, NEAR, FAR)
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.006;
        self.pitch = (self.pitch + dy * 0.006).clamp(self.min_pitch, self.max_pitch);
    }

    pub fn dolly(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(self.min_distance, self.max_distance);
    }

    pub fn update(&mut self, dt: f32) {
        if self.auto_orbit && !self.dragging {
            self.yaw += self.auto_speed * dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_sits_at_the_orbit_distance_from_the_target() {
        let cam = OrbitCamera {
            target: Vec3::ZERO,
            distance: 10.0,
            ..Default::default()
        };
        assert!((cam.eye().length() - 10.0).abs() < 1e-4);
    }

    #[test]
    fn the_view_matrix_puts_the_camera_at_the_origin_and_the_target_at_negative_z() {
        let cam = OrbitCamera {
            target: Vec3::ZERO,
            distance: 5.0,
            yaw: 0.0,
            pitch: 0.0,
            ..Default::default()
        };
        let view = cam.view();
        let eye = view.transform_point3(cam.eye());
        assert!(eye.length() < 1e-4, "eye must map to the view origin, got {eye:?}");
        let ahead = view.transform_point3(Vec3::NEG_Z * -1.0); // world -z is in front here
        assert!(
            view.transform_point3(Vec3::ZERO).z < 0.0,
            "the target must sit in front of the camera in view space"
        );
        let _ = ahead;
    }

    #[test]
    fn world_up_stays_screen_up() {
        let cam = OrbitCamera {
            target: Vec3::ZERO,
            distance: 5.0,
            yaw: 0.0,
            pitch: 0.0,
            ..Default::default()
        };
        let up = cam.view().transform_vector3(Vec3::Y);
        assert!((up.y - 1.0).abs() < 1e-4 && up.x.abs() < 1e-4, "got {up:?}");
    }

    #[test]
    fn pitch_clamps_keep_the_up_vector_stable_at_the_poles() {
        let mut cam = OrbitCamera::default();
        for _ in 0..200 {
            cam.orbit(0.0, 1000.0);
        }
        assert!(cam.pitch <= cam.max_pitch + 1e-6);
        let view = cam.view();
        assert!(view.to_cols_array().iter().all(|v| v.is_finite()));

        let mut cam = OrbitCamera::default();
        for _ in 0..200 {
            cam.orbit(0.0, -1000.0);
        }
        assert!(cam.pitch >= cam.min_pitch - 1e-6);
        assert!(cam.view().to_cols_array().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dolly_is_clamped_both_ways() {
        let mut cam = OrbitCamera::default();
        for _ in 0..50 {
            cam.dolly(0.5);
        }
        assert!((cam.distance - cam.min_distance).abs() < 1e-6);
        for _ in 0..100 {
            cam.dolly(2.0);
        }
        assert!((cam.distance - cam.max_distance).abs() < 1e-6);
    }

    #[test]
    fn projection_maps_the_near_and_far_planes_into_zero_to_one_depth() {
        let cam = OrbitCamera::default();
        let proj = cam.projection(800, 600);
        // `transform_point3` assumes an affine transform — it returns the xyz of
        // the product and drops `w`, which for a projection matrix is precisely
        // the factor that turns z into a depth. So divide by hand.
        let depth_of = |p: Vec3| {
            let clip = proj * p.extend(1.0);
            clip.z / clip.w
        };
        let near_depth = depth_of(Vec3::NEG_Z * NEAR);
        let far_depth = depth_of(Vec3::NEG_Z * FAR);
        assert!(
            near_depth >= -1e-5 && near_depth <= 0.01,
            "near plane should map to depth ~0, got {near_depth}"
        );
        assert!(
            far_depth <= 1.0 + 1e-4 && far_depth > 0.9,
            "far plane should map to depth ~1, got {far_depth}"
        );
    }

    #[test]
    fn degenerate_frame_sizes_do_not_produce_nan() {
        let cam = OrbitCamera::default();
        assert!(cam.projection(0, 0).to_cols_array().iter().all(|v| v.is_finite()));
    }
}
