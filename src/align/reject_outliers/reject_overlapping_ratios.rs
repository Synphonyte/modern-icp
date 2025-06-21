use crate::{golden_section_search, sum_squared_distances, MaskedPointCloud};
use nalgebra::{RealField, Scalar};
use num_traits::{AsPrimitive, Float, One};
use std::cmp::Ordering;

/// Rejects based on the golden section search of the overlapping ratio.
///
/// See this [paper from Dong et al.](https://doi.org/10.1049/iet-cvi.2016.0058) for more details.
pub fn reject_overlapping_ratios<T, const D: usize>(
    x: &mut MaskedPointCloud<T, D>,
    y: &mut MaskedPointCloud<T, D>,
    distances: &[T],
) -> Vec<bool>
where
    T: Scalar + RealField + Float,
    usize: AsPrimitive<T>,
    f32: AsPrimitive<T>,
{
    let (sorted_indices, sorted_distances) = sort_by_distance(distances);

    let count: T = distances.len().as_();
    let ratio = calculate_ratio(distances);

    // TODO : this has to be sorted by distance!
    let is_within_ratio = |(i, _): (usize, &T)| i.as_() < count * ratio;

    let mask: Vec<bool> = sorted_distances
        .iter()
        .enumerate()
        .map(is_within_ratio)
        .collect();

    x.add_order(&sorted_indices);
    y.add_order(&sorted_indices);

    x.add_mask(&mask);
    y.add_mask(&mask);

    mask
}

fn calculate_ratio<T>(distances: &[T]) -> T
where
    T: Scalar + RealField + Float + One,
    f32: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    let lambda = T::one();

    let sum_squared_distances_ratio_fn =
        move |x: T| -> T { sum_squared_distances(distances, Some(x)) / Float::powf(x, lambda) };

    golden_section_search(&sum_squared_distances_ratio_fn, 0.68.as_(), 1.0.as_(), None)
}

fn sort_by_distance<T>(distances: &[T]) -> (Vec<usize>, Vec<T>)
where
    T: Scalar + RealField + Float + PartialOrd,
{
    let mut distances_clone = distances
        .iter()
        .enumerate()
        .map(|(i, d)| (i, *d))
        .collect::<Vec<_>>();

    distances_clone
        .sort_by(|(_, a), (_, b)| nalgebra::partial_cmp(a, b).unwrap_or(Ordering::Equal));

    distances_clone.iter().cloned().unzip()
}
