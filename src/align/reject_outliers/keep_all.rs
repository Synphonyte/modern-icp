use crate::PointCloudIterator;
use nalgebra::{RealField, Scalar};
use num_traits::Float;

/// Doesn't reject any points. That means that all points are kept.
pub fn keep_all<T, const D: usize>(
    _: &mut PointCloudIterator<T, D>,
    _: &mut PointCloudIterator<T, D>,
    distances: &Vec<T>,
) -> Vec<bool>
where
    T: Scalar + RealField + From<f32> + Float,
{
    vec![true; distances.len()]
}
