//! Convex-hull colliders.
//!
//! The simulation the port inherited models every body as a sphere, which is
//! honest for the generated primitives (each has a tuned radius) and wrong for
//! anything long or flat off disk: a spike bounced as a ball. A [`Hull`] is the
//! alternative — a convex polytope in the mesh's own normalised space, stored as
//! both its vertices and its face planes, because the two contact tests the
//! simulation needs each want one of them:
//!
//! * against a wall or another hull's face, the question is "which of my
//!   *vertices* is deepest along this direction" (a support query);
//! * against a sphere, it is "how far outside my *planes* is this point".
//!
//! What this is not: an exact convex hull. The vertices are the mesh's extreme
//! points along a fixed fan of directions, so a hull is a slightly *inner*
//! approximation of the true one, with a bounded vertex count however large the
//! mesh. And hull-against-hull tests only the face normals of the two hulls, not
//! the edge-against-edge axes a full separating-axis test adds, so two hulls
//! meeting edge to edge can register a touch a hair early. Both are the price of
//! a narrow phase that stays cheap enough to run every substep.

use glam::{Mat3, Vec3};

/// Directions the mesh is probed along, beyond the 26 axis and diagonal ones.
const FAN_DIRECTIONS: usize = 70;
/// Two support points closer than this are one vertex.
const VERTEX_MERGE: f32 = 1.0e-4;
/// How far behind a plane a point may sit and still count as on the hull's side
/// of it. Loose enough for a face made of several coplanar triangles.
const PLANE_SLACK: f32 = 1.0e-4;
/// Vertices within this distance of the deepest one share a contact: the
/// contact point is their centroid. This is what lets a box land on a face and
/// stay there, instead of being kicked about by whichever corner is a rounding
/// error lower than the others.
const CONTACT_BAND: f32 = 0.03;

/// A convex polytope in normalised model space (bounding radius at most 1).
#[derive(Clone, Debug)]
pub struct Hull {
    pub verts: Vec<Vec3>,
    /// `(n, d)`: the hull is the set of points with `n.dot(p) <= d` for every
    /// plane. Normals are unit length and point outward.
    pub planes: Vec<(Vec3, f32)>,
}

/// A contact between two bodies: `n` points from the first to the second.
#[derive(Copy, Clone, Debug)]
pub struct Contact {
    pub n: Vec3,
    pub depth: f32,
    pub point: Vec3,
}

/// A hull placed in the world: its body's centre, rotation and uniform scale.
#[derive(Copy, Clone, Debug)]
pub struct Placed<'a> {
    pub hull: &'a Hull,
    pub pos: Vec3,
    pub rot: Mat3,
    pub scale: f32,
}

fn probe_directions() -> Vec<Vec3> {
    let mut dirs = Vec::with_capacity(26 + FAN_DIRECTIONS);
    // The axes and diagonals first: they are what make a box's corners and an
    // axis-aligned part's faces come out exact.
    for x in -1i32..=1 {
        for y in -1i32..=1 {
            for z in -1i32..=1 {
                if (x, y, z) != (0, 0, 0) {
                    dirs.push(Vec3::new(x as f32, y as f32, z as f32).normalize());
                }
            }
        }
    }
    // Then an even spread over the sphere (a Fibonacci spiral).
    let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    for i in 0..FAN_DIRECTIONS {
        let y = 1.0 - 2.0 * (i as f32 + 0.5) / FAN_DIRECTIONS as f32;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let a = golden * i as f32;
        dirs.push(Vec3::new(a.cos() * r, y, a.sin() * r));
    }
    dirs
}

impl Hull {
    /// Build a hull around a point cloud, or `None` when the cloud is flat or
    /// too small to enclose any volume — the caller then keeps the sphere.
    pub fn from_points(points: &[Vec3]) -> Option<Hull> {
        if points.len() < 4 {
            return None;
        }
        let mut verts: Vec<Vec3> = Vec::new();
        for dir in probe_directions() {
            let mut best = points[0];
            let mut best_dot = f32::NEG_INFINITY;
            for &p in points {
                let d = p.dot(dir);
                if d > best_dot {
                    best_dot = d;
                    best = p;
                }
            }
            if !best.is_finite() {
                continue;
            }
            if verts.iter().all(|v| v.distance_squared(best) > VERTEX_MERGE * VERTEX_MERGE) {
                verts.push(best);
            }
        }
        if verts.len() < 4 {
            return None;
        }

        // Every triple of vertices spans a candidate plane; it is a face of the
        // hull exactly when all the other vertices sit on one side of it. With
        // the vertex count bounded by the direction fan this brute force is a
        // few million dot products, once per shape.
        let mut planes: Vec<(Vec3, f32)> = Vec::new();
        let n = verts.len();
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    let normal = (verts[j] - verts[i]).cross(verts[k] - verts[i]);
                    let len = normal.length();
                    if len < 1.0e-7 {
                        continue;
                    }
                    let normal = normal / len;
                    let d = normal.dot(verts[i]);
                    let (mut above, mut below) = (false, false);
                    for v in &verts {
                        let s = normal.dot(*v) - d;
                        above |= s > PLANE_SLACK;
                        below |= s < -PLANE_SLACK;
                        if above && below {
                            break;
                        }
                    }
                    let plane = match (above, below) {
                        (false, true) => (normal, d),
                        (true, false) => (-normal, -d),
                        // Both: it cuts through the hull. Neither: every vertex
                        // is in this plane, so there is no volume at all.
                        _ => continue,
                    };
                    let duplicate = planes
                        .iter()
                        .any(|(pn, pd)| pn.dot(plane.0) > 0.9999 && (pd - plane.1).abs() < 1.0e-3);
                    if !duplicate {
                        planes.push(plane);
                    }
                }
            }
        }
        if planes.len() < 4 {
            return None;
        }
        Some(Hull { verts, planes })
    }

    /// The lowest the hull reaches along `dir` (a unit vector in model space),
    /// and the centroid of the vertices that are down there.
    fn deepest(&self, dir: Vec3, band: f32) -> (f32, Vec3) {
        let mut low = f32::INFINITY;
        for v in &self.verts {
            low = low.min(v.dot(dir));
        }
        let mut sum = Vec3::ZERO;
        let mut count = 0.0;
        for v in &self.verts {
            if v.dot(dir) <= low + band {
                sum += *v;
                count += 1.0;
            }
        }
        (low, sum / count)
    }
}

impl Placed<'_> {
    fn to_world(&self, local: Vec3) -> Vec3 {
        self.pos + self.rot * (local * self.scale)
    }

    /// How far this hull reaches along world direction `-n`, as the smallest
    /// value of `n.dot(x)` over its vertices, with the world-space centroid of
    /// the vertices that reach it.
    pub fn lowest_along(&self, n: Vec3) -> (f32, Vec3) {
        let local = self.rot.transpose() * n;
        let band = CONTACT_BAND / self.scale.max(1.0e-3);
        let (low, centroid) = self.hull.deepest(local, band);
        (n.dot(self.pos) + self.scale * low, self.to_world(centroid))
    }

    /// Contact with a sphere, `n` pointing from the hull to the sphere.
    pub fn against_sphere(&self, centre: Vec3, radius: f32) -> Option<Contact> {
        let scale = self.scale.max(1.0e-3);
        let c = self.rot.transpose() * (centre - self.pos) / scale;
        let r = radius / scale;

        let mut worst = f32::NEG_INFINITY;
        let mut worst_n = Vec3::Y;
        for (n, d) in &self.hull.planes {
            let s = n.dot(c) - d;
            if s > worst {
                worst = s;
                worst_n = *n;
            }
        }
        if worst > r {
            return None;
        }
        let (normal, separation) = if worst <= 0.0 {
            // The centre is inside the hull: leave by the nearest face.
            (worst_n, worst)
        } else {
            // Outside: walk the point back onto the hull by projecting it onto
            // each plane it is beyond, a few rounds over. Near a face that lands
            // on the face; near an edge or corner it lands on the edge or corner,
            // which a single nearest-plane test would get wrong.
            let mut q = c;
            for _ in 0..6 {
                for (n, d) in &self.hull.planes {
                    let s = n.dot(q) - d;
                    if s > 0.0 {
                        q -= *n * s;
                    }
                }
            }
            let gap = c - q;
            let len = gap.length();
            if len < 1.0e-6 {
                (worst_n, worst)
            } else {
                (gap / len, len)
            }
        };
        let depth = (r - separation) * scale;
        if depth <= 0.0 {
            return None;
        }
        let n = (self.rot * normal).normalize_or_zero();
        Some(Contact {
            n,
            depth,
            point: centre - n * radius,
        })
    }

    /// Contact with another hull, `n` pointing from `self` to `other`.
    pub fn against_hull(&self, other: &Placed<'_>) -> Option<Contact> {
        let mut best: Option<Contact> = None;
        let mut best_sep = f32::NEG_INFINITY;
        // Each hull's faces against the other's vertices. A face found with the
        // other hull entirely beyond it means the two are apart.
        for (owner, target, flip) in [(self, other, 1.0f32), (other, self, -1.0f32)] {
            for (n_local, d) in &owner.hull.planes {
                let n = owner.rot * *n_local;
                let offset = n.dot(owner.pos) + owner.scale * d;
                let (low, centroid) = target.lowest_along(n);
                let sep = low - offset;
                if sep > 0.0 {
                    return None;
                }
                if sep > best_sep {
                    best_sep = sep;
                    best = Some(Contact {
                        n: n * flip,
                        depth: -sep,
                        point: centroid,
                    });
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube() -> Hull {
        let mut pts = Vec::new();
        for x in [-0.5f32, 0.5] {
            for y in [-0.5f32, 0.5] {
                for z in [-0.5f32, 0.5] {
                    pts.push(Vec3::new(x, y, z));
                }
            }
        }
        Hull::from_points(&pts).expect("a cube encloses volume")
    }

    fn placed(hull: &Hull, pos: Vec3) -> Placed<'_> {
        Placed {
            hull,
            pos,
            rot: Mat3::IDENTITY,
            scale: 1.0,
        }
    }

    #[test]
    fn a_cube_has_eight_corners_and_six_faces() {
        let hull = cube();
        assert_eq!(hull.verts.len(), 8, "{:?}", hull.verts);
        assert_eq!(hull.planes.len(), 6, "{:?}", hull.planes);
        for (n, d) in &hull.planes {
            assert!((n.length() - 1.0).abs() < 1e-5);
            assert!((d - 0.5).abs() < 1e-5, "each face is half a unit out, got {d}");
        }
    }

    #[test]
    fn interior_points_do_not_become_hull_vertices() {
        let mut pts: Vec<Vec3> = cube().verts;
        pts.push(Vec3::ZERO);
        pts.push(Vec3::new(0.1, -0.2, 0.3));
        let hull = Hull::from_points(&pts).unwrap();
        assert_eq!(hull.verts.len(), 8);
    }

    #[test]
    fn a_flat_cloud_is_refused() {
        let pts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        ];
        assert!(Hull::from_points(&pts).is_none());
    }

    #[test]
    fn a_cube_resting_on_a_face_touches_at_the_face_centre() {
        let hull = cube();
        let p = placed(&hull, Vec3::new(2.0, 0.4, 0.0));
        let (low, centroid) = p.lowest_along(Vec3::Y);
        assert!((low + 0.1).abs() < 1e-5, "the bottom face is at y = -0.1, got {low}");
        assert!((centroid - Vec3::new(2.0, -0.1, 0.0)).length() < 1e-5, "{centroid:?}");
    }

    #[test]
    fn a_sphere_beside_a_corner_is_not_touching_the_faces_it_is_not_near() {
        let hull = cube();
        let p = placed(&hull, Vec3::ZERO);
        // Diagonally off a corner: 0.3 beyond each of three faces, which is
        // 0.52 from the corner itself. A radius of 0.4 clears it; judging by the
        // nearest plane alone (0.3 away) would call that a touch.
        let centre = Vec3::splat(0.8);
        assert!(p.against_sphere(centre, 0.4).is_none());
        let hit = p.against_sphere(centre, 0.6).expect("0.6 reaches the corner");
        assert!((hit.n - Vec3::ONE.normalize()).length() < 1e-3, "{:?}", hit.n);
        assert!((hit.depth - (0.6 - 0.3 * 3.0f32.sqrt())).abs() < 1e-3, "{}", hit.depth);
    }

    #[test]
    fn two_cubes_overlapping_along_x_separate_along_x() {
        let hull = cube();
        let a = placed(&hull, Vec3::ZERO);
        let b = placed(&hull, Vec3::new(0.9, 0.05, 0.0));
        let hit = a.against_hull(&b).expect("they overlap by a tenth");
        assert!((hit.n - Vec3::X).length() < 1e-4, "{:?}", hit.n);
        assert!((hit.depth - 0.1).abs() < 1e-4, "{}", hit.depth);
        let apart = placed(&hull, Vec3::new(1.2, 0.0, 0.0));
        assert!(a.against_hull(&apart).is_none());
    }
}
