use crate::MaskedPointCloud;
use nalgebra::Scalar;

/// Doesn't reject any points. That means that all points are kept.
pub fn keep_all<T, const D: usize>(
    _: &mut MaskedPointCloud<T, D>,
    _: &mut MaskedPointCloud<T, D>,
    distances: &[T],
) -> Vec<bool>
where
    T: Scalar + Copy,
{
    vec![true; distances.len()]
}
