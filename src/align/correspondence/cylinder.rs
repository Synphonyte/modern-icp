use crate::correspondence::{CorrespondenceEstimator, Correspondences};
use crate::{MaskedPointCloud, PointCloud, PointCloudPoint};
use nalgebra::{Point3, RealField, Scalar, point, vector};
use num_traits::{Float, One, Zero};
use std::cell::{Cell, RefCell};

/// Computes the correspondences between the alignee point map and a cylinder of a given radius.
/// The mantle and the top of the cylinder are considered for correspondences between the alignee
/// points and the cylinder.
pub struct Cylinder<T>
where
    T: Scalar + RealField + Float + One + Zero + Copy,
{
    radius: Cell<T>,
    point_cloud: RefCell<PointCloud<T, 3>>,
}

type IntersectionAndDistance<T> = (PointCloudPoint<T, 3>, T);

enum CylinderIntersection<T: Scalar + Copy> {
    Mantle(IntersectionAndDistance<T>),
    Top(IntersectionAndDistance<T>),
}

impl<T> Cylinder<T>
where
    T: Scalar + RealField + Float + One + Zero + Copy,
{
    // https://www.mathcha.io/editor/NDjYxIYwiOphzkr933cVjGeBXcLoLJE8SLn8GzO
    fn compute_intersection_with_cylinder(&self, pos: &Point3<T>) -> CylinderIntersection<T> {
        let pos_2d = pos.coords.fixed_rows::<2>(0);
        let len = pos_2d.norm();

        let radius = self.radius.get();
        let mantle_distance = len - radius;

        if mantle_distance < -pos.z {
            let normalized_pos_2d = pos_2d / len;
            let scaled_pos_2d = normalized_pos_2d * radius;
            CylinderIntersection::Mantle((
                PointCloudPoint::from_pos_norm(
                    point![scaled_pos_2d[0], scaled_pos_2d[1], pos.z],
                    vector![normalized_pos_2d[0], normalized_pos_2d[1], T::one()],
                ),
                mantle_distance,
            ))
        } else {
            CylinderIntersection::Top((
                PointCloudPoint::from_pos_norm(
                    point![pos.x, pos.y, T::zero()],
                    vector![T::zero(), T::zero(), T::one()],
                ),
                pos.z,
            ))
        }
    }
}

impl<'a, T> CorrespondenceEstimator<'a, T, T, 3> for Cylinder<T>
where
    T: Scalar + RealField + Float + One + Zero,
{
    fn new(radius: &T) -> Self {
        Cylinder {
            radius: Cell::new(*radius),
            point_cloud: RefCell::new(vec![]),
        }
    }

    fn find_correspondences<'b, 't, FP>(
        &'t self,
        alignee: &'b PointCloud<T, 3>,
        _target: &'b T,
        filter_points: &mut FP,
    ) -> Correspondences<'b, 't, T, 3>
    where
        FP: FnMut(&PointCloudPoint<T, 3>) -> bool,
        'b: 't,
    {
        let mut point_cloud = self.point_cloud.borrow_mut();
        point_cloud.clear();

        let mut distances = Vec::with_capacity(alignee.len());

        let mut count = T::zero();
        let mut sum = T::zero();

        let mut mask = vec![false; alignee.len()];

        for (i, p) in alignee
            .iter()
            .enumerate()
            .filter(|(_, p)| filter_points(*p))
        {
            let cylinder_intersection = self.compute_intersection_with_cylinder(&p.pos);

            let (intersection, d) = match cylinder_intersection {
                CylinderIntersection::Mantle((i, d)) => {
                    sum += d;
                    count += T::one();
                    (i, d)
                }
                CylinderIntersection::Top(v) => v,
            };

            point_cloud.push(intersection);
            distances.push(Float::abs(d));

            mask[i] = true;
        }

        let mut alignee_cloud = MaskedPointCloud::new(alignee);
        alignee_cloud.add_mask(&mask);

        let target_cloud =
            unsafe { MaskedPointCloud::new(self.point_cloud.as_ptr().as_ref().unwrap()) };

        self.radius.set(self.radius.get() + sum / count);

        Correspondences::from_simple_one_way_correspondences(
            alignee_cloud,
            alignee,
            target_cloud,
            distances,
        )
    }
}
