use crate::reject_outliers::keep_all::keep_all;
use crate::MaskedPointCloud;
use nalgebra::{RealField, Scalar};
use num_traits::Float;
use statistical::standard_scores;

/// Reject outliers based on the n-sigma distance.
///
/// Rejects points that are more than n standard deviations away from the mean.
/// If you're unsure what value to use for n, a common choice is 3.
pub fn reject_n_sigma_dist<T, const D: usize>(
    n: T,
) -> impl Fn(&mut MaskedPointCloud<T, D>, &mut MaskedPointCloud<T, D>, &Vec<T>) -> Vec<bool>
where
    T: Scalar + RealField + Float + Copy + From<f32>,
{
    move |x: &mut MaskedPointCloud<T, D>, y: &mut MaskedPointCloud<T, D>, distances: &Vec<T>| {
        if distances.len() < 2 {
            return keep_all(x, y, distances);
        }

        let mask: Vec<bool> = standard_scores(distances.as_slice())
            .iter()
            .map(|s| s <= &n)
            .collect();

        x.add_mask(&mask);
        y.add_mask(&mask);

        mask
    }
}
