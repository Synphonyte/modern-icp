use nalgebra::*;

use crate::{compute_centroid, demean_into_matrix};

pub struct Plane<T: Scalar + RealField + Copy, const D: usize> {
    pub normal: SVector<T, D>,
    pub constant: T,
}

impl<T: Scalar + RealField + From<f32> + Copy, const D: usize> Plane<T, D> {
    pub fn new(normal: &SVector<T, D>, constant: T) -> Plane<T, D> {
        Plane {
            normal: *normal,
            constant,
        }
    }

    pub fn from_normal_and_point(
        normal: &SVector<T, D>,
        coplanar_point: &Point<T, D>,
    ) -> Plane<T, D> {
        Plane {
            normal: *normal,
            constant: normal.dot(&coplanar_point.coords),
        }
    }

    pub fn distance_to_point(&self, point: &Point<T, D>) -> T {
        self.normal.dot(&point.coords) - self.constant
    }

    pub fn fit_to_points(points: impl Iterator<Item = Point<T, D>> + Clone) -> Plane<T, D> {
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
}

pub type Plane3<T> = Plane<T, 3>;
