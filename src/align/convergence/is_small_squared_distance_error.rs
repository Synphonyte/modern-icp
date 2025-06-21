use crate::sum_squared_distances;
use nalgebra::{RealField, Scalar};
use num_traits::AsPrimitive;

/// Converge if the sum of the squared distances between the alignee and the target has
/// decreased by only a small amount.
pub fn is_small_squared_distance_error<T, M>(
    distances_target: &Vec<T>,
    distances_alignee: &Vec<T>,
    _: &M,
    prev_value: &mut T,
) -> bool
where
    T: Scalar + RealField + Copy,
    f32: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    let sum_squared_distances_fn =
        |distances: &Vec<T>| -> T { sum_squared_distances(distances, None) };

    let target = sum_squared_distances_fn(distances_target);
    let alignee = sum_squared_distances_fn(distances_alignee);
    let target_add_alignee = target + alignee;

    if (*prev_value - target_add_alignee) < 1.5e-2.as_() && *prev_value != T::zero() {
        true
    } else {
        *prev_value = target_add_alignee;
        false
    }
}
