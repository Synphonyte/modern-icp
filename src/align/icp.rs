use crate::correspondence::{CorrespondenceEstimator, Correspondences};
use crate::{MaskedPointCloud, PointCloud, PointCloudPoint, transform_point_cloud};
use cfg_if::cfg_if;
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
/// The `filter_points` function is used to filter out points that are not considered for correspondence.
/// It takes a reference to a `PointCloudPoint` and returns a boolean which is `true` if the point should be included.
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
///
/// It returns the estimated transform and the distance error computed by `is_converged`.
pub fn estimate_transform<'a, T, M, TG, CE, FP, RO, ET, IC>(
    alignee: &PointCloud<T, 3>,
    target: &'a TG,
    max_iterations: usize,
    correspondence_estimator: CE,
    mut filter_points: FP,
    mut reject_outliers: RO,
    mut estimate_step_transform: ET,
    mut is_converged: IC,
) -> (M, T)
where
    T: Scalar + RealField + Float + One + Zero,
    f32: From<T>,
    M: One,
    for<'b> &'b M: Mul<Point3<T>, Output = Point3<T>>
        + Mul<Vector3<T>, Output = Vector3<T>>
        + Mul<M, Output = M>,
    CE: CorrespondenceEstimator<'a, T, TG, 3>,
    FP: FnMut(&PointCloudPoint<T, 3>) -> bool,
    RO: FnMut(&mut MaskedPointCloud<T, 3>, &mut MaskedPointCloud<T, 3>, &Vec<T>) -> Vec<bool>,
    ET: FnMut(&mut MaskedPointCloud<T, 3>, &mut MaskedPointCloud<T, 3>, usize) -> M,
    IC: FnMut(&Vec<T>, &Vec<T>, &M, &mut T) -> bool,
{
    let mut transform = M::one();

    let mut aligned = PointCloud::new();
    aligned.clone_from(alignee);

    cfg_if! {
        if #[cfg(feature = "rerun")]{
            use crate::{pt3_array, vec3_array};

            let mut timeline_step = 0;

            let mut next_step = || {
                timeline_step += 1;
                crate::RR.set_time_sequence("step", timeline_step);
            };

            crate::RR
                .log_static("/", &rerun::ViewCoordinates::RIGHT_HAND_Y_UP())
                .unwrap();

            crate::rr_log_cloud("alignee", &aligned);
        }
    }

    let mut distance_error = T::zero();

    for i in 0..max_iterations {
        #[cfg(feature = "rerun")]
        {
            next_step();
        }

        let Correspondences {
            mut alignee_point_cloud,
            mut corresponding_target_point_cloud,
            mut target_point_cloud,
            mut corresponding_alignee_point_cloud,
            alignee_to_target_distances,
            target_to_alignee_distances,
        } = correspondence_estimator.find_correspondences(&aligned, target, &mut filter_points);

        cfg_if! {
            if #[cfg(feature = "rerun")] {
                let mut alignee_points = vec![];
                let mut alignee_arrows = vec![];
                for (alignee_pt, corr_pt) in alignee_point_cloud
                    .iter()
                    .zip(corresponding_target_point_cloud.iter())
                {
                    alignee_points.push(pt3_array(alignee_pt.pos));
                    alignee_arrows.push(vec3_array(corr_pt.pos - alignee_pt.pos));
                }

                crate::RR
                    .log(
                        "corr/alignee-to-target",
                        &rerun::Arrows3D::from_vectors(&alignee_arrows).with_origins(&alignee_points),
                    )
                    .unwrap();

                let mut target_points = vec![];
                let mut target_arrows = vec![];
                for (target_pt, corr_pt) in target_point_cloud
                    .iter()
                    .zip(corresponding_alignee_point_cloud.iter())
                {
                    target_points.push(pt3_array(target_pt.pos));
                    target_arrows.push(vec3_array(corr_pt.pos - target_pt.pos));
                }

                if !target_points.is_empty() {
                    next_step();

                    crate::RR
                        .log(
                            "corr/target-to-alignee",
                            &rerun::Arrows3D::from_vectors(&target_arrows).with_origins(&target_points),
                        )
                        .unwrap();
                }
            }
        }

        reject_outliers(
            &mut target_point_cloud,
            &mut corresponding_alignee_point_cloud,
            &target_to_alignee_distances,
        );
        reject_outliers(
            &mut alignee_point_cloud,
            &mut corresponding_target_point_cloud,
            &alignee_to_target_distances,
        );

        #[cfg(feature = "rerun")]
        {
            let mut rejected_alignee_points = vec![];
            let mut rejected_alignee_arrows = vec![];
            let mut included_alignee_points = vec![];
            let mut included_alignee_arrows = vec![];

            for (alignee_pt, corr_pt) in alignee_point_cloud
                .iter()
                .zip(corresponding_target_point_cloud.iter())
            {
                included_alignee_points.push(pt3_array(alignee_pt.pos));
                included_alignee_arrows.push(vec3_array(corr_pt.pos - alignee_pt.pos));
            }

            for (i, pt) in alignee_points.iter().enumerate() {
                if !included_alignee_points.contains(&pt) {
                    rejected_alignee_points.push(pt.clone());
                    rejected_alignee_arrows.push(alignee_arrows[i].clone());
                }
            }

            let mut rejected_target_points = vec![];
            let mut rejected_target_arrows = vec![];
            let mut included_target_points = vec![];
            let mut included_target_arrows = vec![];

            for (target_pt, corr_pt) in target_point_cloud
                .iter()
                .zip(corresponding_alignee_point_cloud.iter())
            {
                included_target_points.push(pt3_array(target_pt.pos));
                included_target_arrows.push(vec3_array(corr_pt.pos - target_pt.pos));
            }

            for (i, pt) in target_points.iter().enumerate() {
                if !included_target_points.contains(&pt) {
                    rejected_target_points.push(pt.clone());
                    rejected_target_arrows.push(target_arrows[i].clone());
                }
            }

            next_step();

            crate::RR.log("corr", &rerun::Clear::new(true)).unwrap();

            crate::RR
                .log(
                    "corr/alignee-to-target/included",
                    &rerun::Arrows3D::from_vectors(&included_alignee_arrows)
                        .with_origins(&included_alignee_points)
                        .with_colors(included_alignee_points.iter().map(|_| (0, 170, 0))),
                )
                .unwrap();

            crate::RR
                .log(
                    "corr/alignee-to-target/excluded",
                    &rerun::Arrows3D::from_vectors(&rejected_alignee_arrows)
                        .with_origins(&rejected_alignee_points)
                        .with_colors(rejected_alignee_points.iter().map(|_| (170, 0, 0))),
                )
                .unwrap();

            if !included_target_points.is_empty() {
                next_step();

                crate::RR
                    .log(
                        "corr/target-to-alignee/included",
                        &rerun::Arrows3D::from_vectors(&included_target_arrows)
                            .with_origins(&included_target_points)
                            .with_colors(included_target_points.iter().map(|_| (0, 170, 0))),
                    )
                    .unwrap();

                crate::RR
                    .log(
                        "corr/target-to-alignee/excluded",
                        &rerun::Arrows3D::from_vectors(&rejected_target_arrows)
                            .with_origins(&rejected_target_points)
                            .with_colors(rejected_target_points.iter().map(|_| (170, 0, 0))),
                    )
                    .unwrap();
            }
        }

        let masked_alignee = &mut alignee_point_cloud;
        masked_alignee.extend(&corresponding_alignee_point_cloud);

        let masked_target = &mut corresponding_target_point_cloud;
        masked_target.extend(&target_point_cloud);

        let step_transform = estimate_step_transform(masked_alignee, masked_target, i);

        transform_point_cloud(&mut aligned, &step_transform);

        #[cfg(feature = "rerun")]
        {
            crate::RR.log("corr", &rerun::Clear::new(true)).unwrap();
            crate::rr_log_cloud("alignee", &aligned);
        }

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

    (transform, distance_error)
}
