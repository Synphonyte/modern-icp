use crate::reject_outliers::keep_all::keep_all;
use crate::PointCloudIterator;
use nalgebra::{RealField, Scalar};
use num_traits::Float;
use statistical::standard_scores;

/// Reject outliers based on the 3-sigma distance.
///
/// Rejects points that are more than 3 standard deviations away from the mean.
pub fn reject_3_sigma_dist<T, const D: usize>(
    x: &mut PointCloudIterator<T, D>,
    y: &mut PointCloudIterator<T, D>,
    distances: &Vec<T>,
) -> Vec<bool>
where
    T: Scalar + RealField + From<f32> + Float,
{
    if distances.len() < 2 {
        return keep_all(x, y, distances);
    }

    let mask: Vec<bool> = standard_scores(distances.as_slice())
        .iter()
        .map(|s| s <= &3.0.into())
        .collect();

    x.add_mask(&mask);
    y.add_mask(&mask);

    mask
}
