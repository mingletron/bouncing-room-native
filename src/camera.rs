//! Orbit camera: a port of `src/controls.js`, minus the DOM.
//!
//! Panning the orbit centre is the one thing here the browser build did not have:
//! `controls.js` could only orbit and dolly, so a pile that drifted into a corner
//! could not be followed. [`OrbitCamera::pan`] is the port's own addition, and
//! the room clamping that keeps it sane lives in `app.rs` next to the room
//! constants.
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

    /// The ray through a pixel: where it starts (the eye) and its unit
    /// direction. `x` and `y` are in the same pixels as `width` and `height`,
    /// with the origin at the top left, as the window reports the pointer.
    pub fn ray(&self, x: f32, y: f32, width: u32, height: u32) -> (Vec3, Vec3) {
        let ndc_x = 2.0 * x / width.max(1) as f32 - 1.0;
        let ndc_y = 1.0 - 2.0 * y / height.max(1) as f32;
        let inverse = (self.projection(width, height) * self.view()).inverse();
        let near = inverse.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
        let far = inverse.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
        (self.eye(), (far - near).normalize_or(Vec3::NEG_Z))
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.006;
        self.pitch = (self.pitch + dy * 0.006).clamp(self.min_pitch, self.max_pitch);
    }

    pub fn dolly(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(self.min_distance, self.max_distance);
    }

    /// Slide the orbit centre across the screen: a right-drag grabs the room and
    /// the room follows the pointer.
    ///
    /// `units_per_pixel` is what one pixel of drag is worth in world units, which
    /// only the caller knows — it depends on the window height, the field of view
    /// and how far the camera currently is. Getting that from the caller is what
    /// makes a grab still feel like a grab at both zoom limits.
    ///
    /// Only `target` moves: yaw, pitch and distance are untouched, so a pan is
    /// not a sneaky orbit and never dollies the camera in or out.
    pub fn pan(&mut self, dx: f32, dy: f32, units_per_pixel: f32) {
        // The two in-plane basis vectors, straight from yaw and pitch rather than
        // out of the view matrix: `right` is the camera's x axis in world space,
        // and `up` is perpendicular to that and to the forward direction. Both are
        // unit length at any pitch the clamps allow, so no normalising needed.
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let right = Vec3::new(cy, 0.0, -sy);
        let up = Vec3::new(-sy * sp, cp, -cy * sp);

        // Dragging right moves the room right, which means the camera — and with
        // it the point it is orbiting — moves left. Same logic downward.
        self.target += up * (dy * units_per_pixel) - right * (dx * units_per_pixel);
    }

    /// Restore the opening framing without touching anything the HUD owns.
    ///
    /// `auto_orbit` and `auto_speed` are deliberately left alone: a recenter that
    /// silently re-armed (or killed) the auto-orbit checkbox would be a settings
    /// change nobody asked for.
    pub fn recenter(&mut self) {
        let default = OrbitCamera::default();
        self.yaw = default.yaw;
        self.pitch = default.pitch;
        self.distance = default.distance;
        self.target = default.target;
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
    fn the_ray_through_the_middle_of_the_window_hits_the_orbit_centre() {
        let camera = OrbitCamera::default();
        let (origin, dir) = camera.ray(450.0, 300.0, 900, 600);
        assert!((origin - camera.eye()).length() < 1e-5);
        assert!((dir.length() - 1.0).abs() < 1e-5);
        let to_target = (camera.target - origin).normalize();
        assert!(dir.dot(to_target) > 0.9999, "{dir:?} vs {to_target:?}");

        // A pixel to the right of centre gives a ray to the camera's right, and
        // one above centre a ray that climbs — the two signs a picker can get
        // backwards and still look plausible.
        let (sy, cy) = camera.yaw.sin_cos();
        let right = Vec3::new(cy, 0.0, -sy);
        let (_, to_the_right) = camera.ray(800.0, 300.0, 900, 600);
        assert!(to_the_right.dot(right) > 0.1, "{to_the_right:?}");
        let (_, upward) = camera.ray(450.0, 20.0, 900, 600);
        assert!(upward.y > dir.y, "{upward:?} should climb more than {dir:?}");
    }

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
    fn a_right_drag_grabs_the_room_and_it_follows_the_pointer() {
        // The target itself is always at the screen centre, so the claim that the
        // room moved with the pointer has to be read off some other world point.
        let mut cam = OrbitCamera::default();
        let probe = Vec3::new(1.0, 2.0, -3.0);
        let on_screen = |cam: &OrbitCamera| cam.view().transform_point3(probe);

        let before = on_screen(&cam);
        cam.pan(40.0, 0.0, 0.05);
        assert!(
            on_screen(&cam).x > before.x,
            "a rightward drag must carry the room right"
        );

        let before = on_screen(&cam);
        cam.pan(0.0, 40.0, 0.05);
        assert!(
            on_screen(&cam).y < before.y,
            "and a downward drag (winit's y grows downwards) must carry it down"
        );
    }

    #[test]
    fn panning_never_orbits_or_dollies_on_the_way() {
        let mut cam = OrbitCamera::default();
        let (yaw, pitch, span) = (cam.yaw, cam.pitch, (cam.eye() - cam.target).length());
        cam.pan(120.0, -80.0, 0.05);
        assert_eq!(cam.yaw, yaw, "a pan is not a rotation");
        assert_eq!(cam.pitch, pitch);
        assert!(
            (cam.eye() - cam.target).length() - span < 1e-5,
            "nor a dolly: the eye rides along with the target"
        );
    }

    #[test]
    fn the_pan_step_is_the_pointer_delta_in_world_units() {
        let mut cam = OrbitCamera::default();
        let start = cam.target;
        cam.pan(10.0, 0.0, 0.04);
        let moved = cam.target - start;
        assert!(
            (moved.length() - 0.4).abs() < 1e-5,
            "10 px at 0.04 units per pixel is 0.4 world units, got {:?}",
            moved.length()
        );

        // Twice the pointer travel, twice the slide, same line.
        let mut doubled = OrbitCamera::default();
        doubled.pan(20.0, 0.0, 0.04);
        assert!(
            (doubled.target - start - moved * 2.0).length() < 1e-5,
            "got {:?}",
            doubled.target - start
        );
    }

    #[test]
    fn a_horizontal_drag_never_lifts_the_orbit_centre() {
        // `right` lies in the floor plane at every pitch, so a sideways grab must
        // not raise or drop the centre the camera swings around.
        for pitch in [-0.22, 0.0, 0.42, 0.9, 1.4] {
            let mut cam = OrbitCamera {
                pitch,
                yaw: 1.1,
                ..Default::default()
            };
            let start = cam.target;
            cam.pan(60.0, 0.0, 0.02);
            assert!(
                (cam.target.y - start.y).abs() < 1e-6,
                "pitch {pitch}: the centre moved {:.6} in y",
                cam.target.y - start.y
            );
        }
    }

    #[test]
    fn pan_stays_finite_at_the_pitch_poles() {
        let mut cam = OrbitCamera::default();
        cam.orbit(0.0, 1.0e4);
        cam.pan(60.0, 60.0, 0.1);
        assert!(cam.target.is_finite(), "got {:?}", cam.target);
        cam.orbit(0.0, -1.0e5);
        cam.pan(60.0, 60.0, 0.1);
        assert!(cam.target.is_finite(), "got {:?}", cam.target);
    }

    #[test]
    fn recenter_restores_the_framing_but_not_the_users_other_settings() {
        let mut cam = OrbitCamera::default();
        cam.auto_orbit = false;
        cam.auto_speed = 0.5;
        cam.orbit(500.0, 300.0);
        cam.dolly(3.0);
        cam.pan(120.0, -80.0, 0.05);

        cam.recenter();
        let fresh = OrbitCamera::default();
        assert_eq!((cam.yaw, cam.pitch, cam.distance), (fresh.yaw, fresh.pitch, fresh.distance));
        assert_eq!(cam.target, fresh.target);
        assert!(!cam.auto_orbit, "recentering must not re-arm auto-orbit");
        assert_eq!(cam.auto_speed, 0.5, "nor reset its speed");
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
