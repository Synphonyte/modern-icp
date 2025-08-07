use nalgebra::*;
use num_traits::{Float, Zero};
use statistical::standard_deviation;

use crate::{MaskedPointCloud, compute_centroid, demean_into_matrix};

/// Plane that is described by the equation `normal.dot(point_on_plane.coords) - constant == 0`
#[derive(Clone, Copy)]
pub struct Plane<T: Scalar + RealField + Copy, const D: usize> {
    pub normal: SVector<T, D>,
    pub constant: T,
}

impl<T: Scalar + RealField + Copy, const D: usize> Plane<T, D> {
    pub fn new(normal: &SVector<T, D>, constant: T) -> Self {
        Plane {
            normal: *normal,
            constant,
        }
    }

    pub fn from_normal_and_point(normal: &SVector<T, D>, coplanar_point: &Point<T, D>) -> Self {
        Plane {
            normal: *normal,
            constant: normal.dot(&coplanar_point.coords),
        }
    }

    /// Computes the signed distance from this plane to the given point.
    ///
    /// If the returned distance is positive the point is above the plane, i.e. in the direction of the normal.
    /// If the returned distance is negative the point is below the plane.
    pub fn distance_to_point(&self, point: &Point<T, D>) -> T {
        self.normal.dot(&point.coords) - self.constant
    }

    pub fn pivot(&self) -> Point<T, D> {
        Point::from(self.normal * self.constant)
    }

    pub fn set_pivot(&mut self, pivot: Point<T, D>) {
        self.constant = self.normal.dot(&pivot.coords);
    }

    /// Computes the standard deviation of the distances from the plane to the given points.
    pub fn points_dist_std_dev(&self, points: impl Iterator<Item = Point<T, D>>) -> T
    where
        T: Float,
    {
        let distances: Vec<T> = points
            .map(|p| Float::abs(self.distance_to_point(&p)))
            .collect();

        standard_deviation(distances.as_slice(), None)
    }

    /// Fit the plane to the given points using SVD.
    pub fn fit_to_points(points: impl Iterator<Item = Point<T, D>> + Clone) -> Self {
        let alignee_centroid = compute_centroid(points.clone());
        let demeaned_alignee = demean_into_matrix(points, &alignee_centroid);

        let SVD {
            u,
            v_t: _,
            singular_values: _,
        } = demeaned_alignee
            .try_svd(true, false, T::default_epsilon(), 10)
            .expect("SVD unsuccessful");

        let uw = u.unwrap();
        let normal = uw.column(2);

        Plane::from_normal_and_point(&normal.normalize(), &Point::from(alignee_centroid))
    }

    /// Fits a plane iteratively to the point cloud using SVD and then rejects points that are farther
    /// than `n_sigma` away from the plane.
    ///
    /// ## Parameters
    ///
    /// - `masked_point_cloud`: The masked point to fit the plane to.
    /// - `n_sigma`: The threshold for the standard deviation of the distances from the plane.
    /// - `max_iterations`: The maximum number of iterations of matching the plane to the cloud to perform.
    /// - `std_dev_threshold`: When the standard deviation of the distances from the plane is less than this value,
    ///   the plane is considered to be a good fit and the fitting algorithm terminates.
    ///
    /// ## Returns
    ///
    /// A tuple of the plane and the mask of which points were used to define the plane.
    /// Also, this returned function applies the mask to the masked point cloud.
    pub fn fit_to_point_cloud_wo_outliers<'a>(
        point_cloud: &mut MaskedPointCloud<'a, T, D>,
        n_sigma: T,
        max_iterations: usize,
        std_dev_threshold: T,
    ) -> (Self, Vec<bool>)
    where
        T: Zero + Float,
        f32: From<T>,
    {
        let len = point_cloud.len();

        let mut mask = vec![true; len];
        let mut plane = Self::new(&Vector::zero(), T::zero());

        for _ in 0..max_iterations {
            plane = Plane::fit_to_points(point_cloud.points_iter());

            #[cfg(feature = "rerun")]
            {
                point_cloud.log_rerun("fit_to_point_cloud_wo_outliers/point_cloud");
                plane.log_rerun("fit_to_point_cloud_wo_outliers/plane");
            }

            let distances: Vec<T> = point_cloud
                .iter()
                .map(|p| Float::abs(plane.distance_to_point(&p.pos)))
                .collect();

            let standard_deviation = standard_deviation(distances.as_slice(), None);

            if standard_deviation < std_dev_threshold {
                break;
            }

            let mut local_mask = vec![true; point_cloud.len()];

            for (i, d) in distances.into_iter().enumerate() {
                if d > standard_deviation * n_sigma {
                    local_mask[i] = false;

                    let mut j = 0;
                    for m in mask.iter_mut() {
                        if *m {
                            if j == i {
                                *m = false;
                                break;
                            }
                            j += 1;
                        }
                    }
                }
            }

            point_cloud.add_mask(&local_mask);
        }

        (plane, mask)
    }
}

pub type Plane3<T> = Plane<T, 3>;

#[cfg(feature = "rerun")]
impl<T, const D: usize> Plane<T, D>
where
    T: Scalar + RealField + Copy,
    f32: From<T>,
{
    pub fn log_rerun(&self, name: &str) {
        let pivot = self.pivot();

        crate::RR
            .log(
                name,
                &rerun::Ellipsoids3D::from_centers_and_half_sizes(
                    [(pivot[0].into(), pivot[1].into(), pivot[2].into())],
                    [(10.0, 10.0, 0.0)],
                )
                .with_quaternions([crate::unit_quat_array(
                    UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::face_towards(
                        &Vector3::new(self.normal[0], self.normal[1], self.normal[2]),
                        &Vector3::y(),
                    )),
                )]),
            )
            .unwrap();

        crate::RR
            .log(
                format!("{name}/normal"),
                &rerun::Arrows3D::from_vectors([crate::vec3_array(self.normal)])
                    .with_origins([crate::pt3_array(pivot)]),
            )
            .unwrap();
    }
}
