use crate::sum_squared_distances;
use nalgebra::{RealField, Scalar};
use num_traits::AsPrimitive;

/// Converge if the sum of the squared distances between the alignee and the target has
/// decreased by less than `epsilon`.
pub fn same_squared_distance_error<T, M>(
    epsilon: T,
) -> impl Fn(&[T], &[T], &M, &mut T, usize) -> bool
where
    T: Scalar + RealField + Copy,
    usize: AsPrimitive<T>,
{
    move |distances_target: &[T], distances_alignee: &[T], _: &M, error: &mut T, _: usize| {
        let sum_squared_distances_fn =
            |distances: &[T]| -> T { sum_squared_distances(distances, None) };

        let target = sum_squared_distances_fn(distances_target);
        let alignee = sum_squared_distances_fn(distances_alignee);
        let new_error = target + alignee;

        let is_small = (new_error - *error).abs() < epsilon;

        *error = new_error;

        is_small
    }
}
