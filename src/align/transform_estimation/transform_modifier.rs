use crate::MaskedPointCloud;
use nalgebra::*;
use num_traits::{Float, One, Zero};
use std::ops::Mul;

/// Wraps the given transform estimator with a function that modifies the estimated transform before returning it.
///
/// ## Example
///
/// This uses the `translation_point_to_plane::estimate_translation` function to estimate the translation
/// but then modifies the estimation to only move along the y axis.
///
/// ```
/// # use modern_icp::transform_estimation::{
/// #     transform_modifier::build_modified_transform_estimator,
/// #     translation_point_to_plane::estimate_translation,
/// # };
/// # use nalgebra::{IsometryMatrix3, Translation3};
///
/// build_modified_transform_estimator(
///     estimate_translation,
///     |t| Some(IsometryMatrix3::from_parts(
///         Translation3::new(0.0, t.translation.y, 0.0),
///         t.rotation,
///     )),
/// );
/// ```
pub fn build_modified_transform_estimator<T, M, ET, MT>(
    mut estimate_transform: ET,
    mut modify_transform: MT,
) -> impl FnMut(&mut MaskedPointCloud<T, 3>, &mut MaskedPointCloud<T, 3>, usize) -> Option<M>
where
    T: Scalar + RealField + Float + One + Zero,
    f32: From<T>,
    M: One,
    for<'b> &'b M: Mul<Point3<T>, Output = Point3<T>>
        + Mul<Vector3<T>, Output = Vector3<T>>
        + Mul<M, Output = M>,
    ET: FnMut(&mut MaskedPointCloud<T, 3>, &mut MaskedPointCloud<T, 3>, usize) -> Option<M>
        + 'static,
    MT: FnMut(M) -> Option<M> + 'static,
{
    move |x: &mut MaskedPointCloud<T, 3>, y: &mut MaskedPointCloud<T, 3>, i: usize| {
        let transform = estimate_transform(x, y, i)?;
        modify_transform(transform)
    }
}

/// Builds an interlaced transform estimator.
///
/// This allos to use multiple transform estimators in a row.
/// The first estimator is used for the first iteration, the second estimator for the second iteration and so on.
pub fn build_interlaced_transform_estimator<T, M, const S: usize>(
    transform_estimators: &'static mut [impl FnMut(
        &mut MaskedPointCloud<T, 3>,
        &mut MaskedPointCloud<T, 3>,
        usize,
    ) -> Option<M>; S],
) -> impl FnMut(&mut MaskedPointCloud<T, 3>, &mut MaskedPointCloud<T, 3>, usize) -> Option<M>
where
    T: Scalar + RealField + Float + One + Zero,
    f32: From<T>,
    M: One,
    for<'b> &'b M: Mul<Point3<T>, Output = Point3<T>>
        + Mul<Vector3<T>, Output = Vector3<T>>
        + Mul<M, Output = M>,
{
    move |x: &mut MaskedPointCloud<T, 3>, y: &mut MaskedPointCloud<T, 3>, iteration: usize| {
        transform_estimators[iteration % S](x, y, iteration / S)
    }
}
