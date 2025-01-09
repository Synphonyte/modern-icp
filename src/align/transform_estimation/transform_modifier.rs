use crate::PointCloudIterator;
use nalgebra::*;
use num_traits::{Float, One, Zero};
use std::ops::Mul;

/// Wraps the give transform estimator with a function that modifies the estimated transform before returning it.
///
/// ## Example
///
/// This uses the `translation_point_to_plane::estimate_translation` function to estimate the translation
/// but then modifies the estimation to only move along the y axis.
///
/// ```
/// build_modified_transform_estimator(
///     translation_point_to_plane::estimate_translation,
///     |t| IsometryMatrix3::from_parts(
///         Translation3::new(0.0, t.translation.y, 0.0),
///         t.rotation,
///     ),
/// ),
/// ```
pub fn build_modified_transform_estimator<T, M, ET, MT>(
    mut estimate_transform: ET,
    mut modify_transform: MT,
) -> impl FnMut(&mut PointCloudIterator<T, 3>, &mut PointCloudIterator<T, 3>, usize) -> M
where
    T: Scalar + RealField + Float + One + Zero,
    f32: From<T>,
    M: One,
    for<'b> &'b M: Mul<Point3<T>, Output = Point3<T>>
        + Mul<Vector3<T>, Output = Vector3<T>>
        + Mul<M, Output = M>,
    ET: FnMut(&mut PointCloudIterator<T, 3>, &mut PointCloudIterator<T, 3>, usize) -> M + 'static,
    MT: FnMut(M) -> M + 'static,
{
    move |x: &mut PointCloudIterator<T, 3>, y: &mut PointCloudIterator<T, 3>, i: usize| {
        let transform = estimate_transform(x, y, i);
        modify_transform(transform)
    }
}

/// Builds an interlaced transform estimator.
///
/// This allos to use multiple transform estimators in a row.
/// The first estimator is used for the first iteration, the second estimator for the second iteration and so on.
pub fn build_interlaced_transform_estimator<T, M, const S: usize>(
    transform_estimators: &'static mut [impl FnMut(&mut PointCloudIterator<T, 3>, &mut PointCloudIterator<T, 3>, usize) -> M;
                     S],
) -> impl FnMut(&mut PointCloudIterator<T, 3>, &mut PointCloudIterator<T, 3>, usize) -> M
where
    T: Scalar + RealField + Float + One + Zero,
    f32: From<T>,
    M: One,
    for<'b> &'b M: Mul<Point3<T>, Output = Point3<T>>
        + Mul<Vector3<T>, Output = Vector3<T>>
        + Mul<M, Output = M>,
{
    move |x: &mut PointCloudIterator<T, 3>, y: &mut PointCloudIterator<T, 3>, iteration: usize| {
        transform_estimators[iteration % S](x, y, iteration / S)
    }
}
