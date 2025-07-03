use nalgebra::Scalar;

use crate::PointCloudPoint;

/// Accepts all points. Always returns `true`.
pub fn accept_all<T, const D: usize>(_point: &PointCloudPoint<T, D>) -> bool
where
    T: Scalar + Copy,
{
    true
}
