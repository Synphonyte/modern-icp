use crate::correspondence::{CorrespondenceEstimator, Correspondences};
use crate::{transform_point_cloud, PointCloud, PointCloudIterator};
use nalgebra::*;
use num_traits::{Float, One, Zero};
use std::ops::Mul;

/// Estimates the transform that the alignee point cloud has to be transformed by to match the
/// target using the iterative closest point algorithm.
///
/// The `target` parameter can be any type that the `correspondence_estimator` can handle.
///
/// `max_iterations` is the maximum number of iterations to perform.
///
/// `correspondence_estimator` is used to find the correspondences between the alignee and the target.
/// Please see the [`CorrespondenceEstimator`] documentation for more information.
///
/// `reject_outliers` is used to reject outliers from the correspondences. It is simply a function
/// that takes the point iterators of one point cloud and the corresponding point iterators of the
/// other point cloud together with the distances between the points. It computes a mask vector
/// of booleans that is applied to both of the point iterators.
///
/// `estimate_step_transform` is used to estimate the transform that should be applied to the alignee
/// point cloud to match the target.
///
/// `is_converged` is used to check if the algorithm has converged.
pub fn estimate_transform<'a, T, M, TG, CE, RO, ET, IC>(
    alignee: &PointCloud<T, 3>,
    target: &'a TG,
    max_iterations: usize,
    correspondence_estimator: CE,
    mut reject_outliers: RO,
    mut estimate_step_transform: ET,
    mut is_converged: IC,
) -> M
where
    T: Scalar + RealField + Float + One + Zero,
    f32: From<T>,
    M: One,
    for<'b> &'b M: Mul<Point3<T>, Output = Point3<T>>
        + Mul<Vector3<T>, Output = Vector3<T>>
        + Mul<M, Output = M>,
    CE: CorrespondenceEstimator<'a, T, TG, 3>,
    RO: FnMut(&mut PointCloudIterator<T, 3>, &mut PointCloudIterator<T, 3>, &Vec<T>) -> Vec<bool>,
    ET: FnMut(&mut PointCloudIterator<T, 3>, &mut PointCloudIterator<T, 3>, usize) -> M,
    IC: FnMut(&Vec<T>, &Vec<T>, &M, &mut T) -> bool,
{
    let mut transform = M::one();

    let mut aligned = PointCloud::new();
    aligned.clone_from(alignee);

    let mut distance_error = T::zero();

    for i in 0..max_iterations {
        let Correspondences {
            mut alignee_points_iter,
            mut corresponding_target_points_iter,
            mut target_points_iter,
            mut corresponding_alignee_points_iter,
            alignee_to_target_distances,
            target_to_alignee_distances,
        } = correspondence_estimator.find_correspondences(&aligned, target);

        corresponding_target_points_iter.reset_iter();

        reject_outliers(
            &mut target_points_iter,
            &mut corresponding_alignee_points_iter,
            &target_to_alignee_distances,
        );
        reject_outliers(
            &mut alignee_points_iter,
            &mut corresponding_target_points_iter,
            &alignee_to_target_distances,
        );

        let mut masked_alignee = &mut alignee_points_iter;
        masked_alignee.reset_iter();
        masked_alignee.extend(&corresponding_alignee_points_iter);

        let mut masked_target = &mut corresponding_target_points_iter;
        masked_target.reset_iter();
        masked_target.extend(&target_points_iter);

        let step_transform = estimate_step_transform(&mut masked_alignee, &mut masked_target, i);

        transform_point_cloud(&mut aligned, &step_transform);

        transform = &step_transform * transform;

        if is_converged(
            &alignee_to_target_distances,
            &target_to_alignee_distances,
            &step_transform,
            &mut distance_error,
        ) {
            break;
        }
    }

    return transform;
}
