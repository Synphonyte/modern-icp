use nalgebra::RealField;

use crate::{Plane, PointCloudPoint};

/// Returns a closure that checks if a point is above all given planes.
pub fn above_planes<'a, T, I, const D: usize>(
    planes: &'a I,
) -> impl Fn(&PointCloudPoint<T, D>) -> bool
where
    T: RealField + Copy + From<f32>,
    &'a I: IntoIterator<Item = &'a Plane<T, D>> + 'a,
{
    move |point| {
        planes
            .into_iter()
            .all(|plane| plane.distance_to_point(&point.pos) > T::zero())
    }
}
