use crate::convergence::ConvergenceCriterion;
use crate::correspondence::{CorrespondenceEstimator, Correspondences};
use crate::filter_points::{AcceptAll, PointFilter};
use crate::reject_outliers::{KeepAll, OutlierRejector};
use crate::transform_estimation::TransformEstimator;
use crate::{MaskedPointCloud, PointCloudPoint, ToPointCloud, transform_point_cloud};
use cfg_if::cfg_if;
use nalgebra::*;
use num_traits::{Float, One, Zero};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Mul;
use tracing::info;

/// The ICP algorithm for aligning two point clouds.
///
/// This is the main struct for this crate.
pub struct Icp<'a, T, M, TG, CE, ET, IC, const D: usize, FP = AcceptAll, RO = KeepAll>
where
    T: Scalar + RealField + Float + One + Zero + Debug,
    f32: From<T>,
    M: One
        + Clone
        + Mul<Point<T, D>, Output = Point<T, D>>
        + Mul<SVector<T, D>, Output = SVector<T, D>>
        + Mul<M, Output = M>,
    FP: PointFilter<T, D>,
    RO: OutlierRejector<T, D>,
{
    correspondence_estimator: CE,
    estimate_step_transform: ET,
    is_converged: IC,
    max_iterations: usize,
    filter_points: FP,
    reject_outliers: RO,

    _marker: PhantomData<(T, M, TG)>,
    _lt: &'a (),
}

impl<'a, T, M, const D: usize> Icp<'a, T, M, (), (), (), (), D>
where
    T: Scalar + RealField + Float + One + Zero + Debug,
    f32: From<T>,
    M: One
        + Clone
        + Mul<Point<T, D>, Output = Point<T, D>>
        + Mul<SVector<T, D>, Output = SVector<T, D>>
        + Mul<M, Output = M>,
{
    /// Creates a new incomplete [`Icp`] instance with some default values and some missing values that need to be set before calling [`Icp::estimate_transform`].
    ///
    /// You have to set the following fields before calling [`Icp::estimate_transform`]:
    /// - [`Icp::correspondence_estimator`]
    /// - [`Icp::estimate_step_transform`]
    /// - [`Icp::is_converged`]
    ///
    /// ## Example
    ///
    /// ```
    /// # use modern_icp::{Icp, PointCloud};
    /// # use modern_icp::correspondence::{CorrespondenceEstimator, NearestNeighbor};
    /// # use modern_icp::transform_estimation::point_to_plane_lls;
    /// # use modern_icp::convergence::same_squared_distance_error;
    /// #
    /// # let alignee_cloud = PointCloud::<f32, 3>::new();
    /// # let target_cloud = PointCloud::<f32, 3>::new();
    /// #
    /// let (alignee_transform, error_sum) = Icp::new()
    ///     .correspondence_estimator(NearestNeighbor::new(&target_cloud))
    ///     .estimate_step_transform(point_to_plane_lls::estimate_isometry)
    ///     .is_converged(same_squared_distance_error(0.1))
    ///     .estimate_transform(alignee_cloud, &target_cloud);
    /// ```
    #[allow(
        clippy::new_without_default,
        reason = "Default would be misleading as this creates an semi-uninitialized Icp instance that has to be completed using a builder pattern"
    )]
    pub fn new() -> Self {
        Self {
            max_iterations: 50,
            correspondence_estimator: (),
            estimate_step_transform: (),
            is_converged: (),
            filter_points: AcceptAll,
            reject_outliers: KeepAll,

            _marker: PhantomData,
            _lt: &(),
        }
    }
}

impl<'a, T, M, TG, CE, ET, IC, const D: usize, FP, RO> Icp<'a, T, M, TG, CE, ET, IC, D, FP, RO>
where
    T: Scalar + RealField + Float + One + Zero + Debug,
    f32: From<T>,
    M: One
        + Clone
        + Mul<Point<T, D>, Output = Point<T, D>>
        + Mul<SVector<T, D>, Output = SVector<T, D>>
        + Mul<M, Output = M>,
    CE: CorrespondenceEstimator<'a, T, TG, D>,
    ET: TransformEstimator<T, M, D>,
    IC: ConvergenceCriterion<T, M>,
    FP: PointFilter<T, D>,
    RO: OutlierRejector<T, D>,
{
    /// Estimates the transform that the `alignee` point cloud has to be transformed by to match the
    /// `target` using the iterative closest point algorithm.
    ///
    /// The `target` parameter can be any type that the `correspondence_estimator` can handle.
    ///
    /// It returns the estimated transform and the distance error computed by `is_converged`.
    ///
    /// > If you get a compiler error that this method is not available, make sure you have called at least the
    /// > following methods before calling this method:
    /// > - [`Icp::correspondence_estimator()`]
    /// > - [`Icp::estimate_step_transform()`]
    /// > - [`Icp::is_converged()`]
    ///
    /// ## Example
    ///
    /// ```
    /// # use modern_icp::{Icp, PointCloud};
    /// # use modern_icp::correspondence::{CorrespondenceEstimator, NearestNeighbor};
    /// # use modern_icp::transform_estimation::point_to_plane_lls;
    /// # use modern_icp::convergence::same_squared_distance_error;
    /// #
    /// # let alignee_cloud = PointCloud::<f32, 3>::new();
    /// # let target_cloud = PointCloud::<f32, 3>::new();
    /// #
    /// let (alignee_transform, error_sum) = Icp::new()
    ///     .correspondence_estimator(NearestNeighbor::new(&target_cloud))
    ///     .estimate_step_transform(point_to_plane_lls::estimate_isometry)
    ///     .is_converged(same_squared_distance_error(0.1))
    ///     .estimate_transform(alignee_cloud, &target_cloud);
    /// ```
    pub fn estimate_transform(
        &mut self,
        alignee: impl ToPointCloud<T, D>,
        target: &'a TG,
    ) -> (M, T) {
        let mut transform = M::one();

        let mut aligned = alignee.to_point_cloud();

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

        for i in 0..self.max_iterations {
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
            } = self.correspondence_estimator.find_correspondences(
                &aligned,
                target,
                &mut self.filter_points,
            );

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

            self.reject_outliers.reject(
                &mut target_point_cloud,
                &mut corresponding_alignee_point_cloud,
                &target_to_alignee_distances,
            );
            self.reject_outliers.reject(
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
                    if !included_alignee_points.contains(pt) {
                        rejected_alignee_points.push(*pt);
                        rejected_alignee_arrows.push(alignee_arrows[i]);
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
                    if !included_target_points.contains(pt) {
                        rejected_target_points.push(*pt);
                        rejected_target_arrows.push(target_arrows[i]);
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

            if masked_alignee.is_empty() || masked_target.is_empty() {
                info!("Correspondence estimation gave empty result. Terminating.");
                break;
            }

            let step_transform =
                match self
                    .estimate_step_transform
                    .estimate(masked_alignee, masked_target, i)
                {
                    Some(step_transform) => step_transform,
                    None => {
                        info!("Step transform estimation failed. Terminating.");
                        break;
                    }
                };

            transform_point_cloud(&mut aligned, step_transform.clone());

            transform = step_transform.clone() * transform;

            #[cfg(feature = "rerun")]
            {
                crate::RR.log("corr", &rerun::Clear::new(true)).unwrap();
                crate::rr_log_cloud("alignee", &aligned);
            }

            if self.is_converged.is_converged(
                &alignee_to_target_distances,
                &target_to_alignee_distances,
                &step_transform,
                &mut distance_error,
                i,
            ) {
                break;
            }
        }

        (transform, distance_error)
    }
}

impl<'a, T, M, TG, CE, ET, IC, const D: usize, FP, RO> Icp<'a, T, M, TG, CE, ET, IC, D, FP, RO>
where
    T: Scalar + RealField + Float + One + Zero + Debug,
    f32: From<T>,
    M: One
        + Clone
        + Mul<Point<T, D>, Output = Point<T, D>>
        + Mul<SVector<T, D>, Output = SVector<T, D>>
        + Mul<M, Output = M>,
    FP: PointFilter<T, D>,
    RO: OutlierRejector<T, D>,
{
    /// Sets the maximum number of iterations to perform.
    /// If the convergence criterion is not met within this number of iterations, the algorithm will stop.
    ///
    /// Defaults to `50`.
    pub fn max_iterations(self, max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..self
        }
    }

    /// The `filter_points` function is used to filter out points that are not considered for correspondence.
    /// It takes a reference to a `PointCloudPoint` and returns a boolean which is `true` if the point should be included.
    /// Use this to exclude points outside a bounding box, for example, or to filter out points that are too close to the sensor, etc.
    ///
    /// Defaults to [`crate::filter_points::accept_all`] which does not filter any points.
    ///
    /// Check the module [`crate::filter_points`] for built-in filters. It is very easy to implement your own filter.
    ///
    /// ## Example
    ///
    /// ```
    /// # use modern_icp::{Icp, PointCloud, PointCloudPoint};
    /// # use modern_icp::correspondence::{CorrespondenceEstimator, NearestNeighbor};
    /// # use modern_icp::transform_estimation::point_to_plane_lls;
    /// # use modern_icp::convergence::same_squared_distance_error;
    /// #
    /// # let alignee_cloud = PointCloud::<f32, 3>::new();
    /// # let target_cloud = PointCloud::<f32, 3>::new();
    /// #
    /// let (alignee_transform, error_sum) = Icp::new()
    ///     .correspondence_estimator(NearestNeighbor::new(&target_cloud))
    ///     .estimate_step_transform(point_to_plane_lls::estimate_isometry)
    ///     .is_converged(same_squared_distance_error(0.1))
    ///     .filter_points(|pt: &PointCloudPoint<f32, 3>| pt.pos.z > 0.0) // Filter: only use points above the xy-plane
    ///     .estimate_transform(alignee_cloud, &target_cloud);
    /// ```
    pub fn filter_points<NewFP: PointFilter<T, D>>(
        self,
        filter_points: NewFP,
    ) -> Icp<'a, T, M, TG, CE, ET, IC, D, NewFP, RO> {
        Icp {
            filter_points,
            max_iterations: self.max_iterations,
            correspondence_estimator: self.correspondence_estimator,
            reject_outliers: self.reject_outliers,
            estimate_step_transform: self.estimate_step_transform,
            is_converged: self.is_converged,

            _marker: PhantomData,
            _lt: self._lt,
        }
    }

    /// `reject_outliers` is used to reject outliers from the correspondences. It is simply a function
    /// that takes the point iterators of one point cloud and the corresponding point iterators of the
    /// other point cloud together with the distances between the points. It computes a mask vector
    /// of booleans that is applied to both of the point iterators.
    ///
    /// Defaults to [`crate::reject_outliers::keep_all`] which does not reject any outliers.
    ///
    /// If you want to reject outliers because of noise, or your point clouds are a bit dissimilar,
    /// a good place to start is with [`crate::reject_outliers::reject_n_sigma_dist`] as shown in the example below.
    /// Please refer to the module [`crate::reject_outliers`] to see all built-in outlier rejection functions.
    ///
    /// ## Example
    ///
    /// ```
    /// # use modern_icp::{Icp, PointCloud, PointCloudPoint};
    /// # use modern_icp::correspondence::{CorrespondenceEstimator, NearestNeighbor};
    /// # use modern_icp::transform_estimation::point_to_plane_lls;
    /// # use modern_icp::convergence::same_squared_distance_error;
    /// # use modern_icp::reject_outliers::reject_n_sigma_dist;
    /// #
    /// # let alignee_cloud = PointCloud::<f32, 3>::new();
    /// # let target_cloud = PointCloud::<f32, 3>::new();
    /// #
    /// let (alignee_transform, error_sum) = Icp::new()
    ///     .correspondence_estimator(NearestNeighbor::new(&target_cloud))
    ///     .estimate_step_transform(point_to_plane_lls::estimate_isometry)
    ///     .is_converged(same_squared_distance_error(0.1))
    ///     .reject_outliers(reject_n_sigma_dist(3.0))
    ///     .estimate_transform(alignee_cloud, &target_cloud);
    /// ```
    pub fn reject_outliers<NewRO>(
        self,
        reject_outliers: NewRO,
    ) -> Icp<'a, T, M, TG, CE, ET, IC, D, FP, NewRO>
    where
        NewRO: FnMut(&mut MaskedPointCloud<T, D>, &mut MaskedPointCloud<T, D>, &[T]) -> Vec<bool>,
    {
        Icp {
            filter_points: self.filter_points,
            max_iterations: self.max_iterations,
            correspondence_estimator: self.correspondence_estimator,
            reject_outliers,
            estimate_step_transform: self.estimate_step_transform,
            is_converged: self.is_converged,

            _marker: PhantomData,
            _lt: self._lt,
        }
    }
}

impl<'a, T, M, ET, IC, const D: usize, FP, RO> Icp<'a, T, M, (), (), ET, IC, D, FP, RO>
where
    T: Scalar + RealField + Float + One + Zero + Debug,
    f32: From<T>,
    M: One
        + Clone
        + Mul<Point<T, D>, Output = Point<T, D>>
        + Mul<SVector<T, D>, Output = SVector<T, D>>
        + Mul<M, Output = M>,
    FP: PointFilter<T, D>,
    RO: OutlierRejector<T, D>,
{
    /// `correspondence_estimator` is used to find the correspondences between the alignee and the target.
    /// This method is required to be called before [`Icp::estimate_transform`].
    ///
    /// Please see the [`CorrespondenceEstimator`] documentation for more information and check the module
    /// [`crate::correspondence`] for all built-in correspondence estimators.
    ///
    /// ## Example
    ///
    /// ```
    /// # use modern_icp::{Icp, PointCloud};
    /// # use modern_icp::correspondence::{CorrespondenceEstimator, NearestNeighbor};
    /// # use modern_icp::transform_estimation::point_to_plane_lls;
    /// # use modern_icp::convergence::same_squared_distance_error;
    /// #
    /// # let alignee_cloud = PointCloud::<f32, 3>::new();
    /// # let target_cloud = PointCloud::<f32, 3>::new();
    /// #
    /// let (alignee_transform, error_sum) = Icp::new()
    ///     .correspondence_estimator(NearestNeighbor::new(&target_cloud))
    ///     .estimate_step_transform(point_to_plane_lls::estimate_isometry)
    ///     .is_converged(same_squared_distance_error(0.1))
    ///     .estimate_transform(alignee_cloud, &target_cloud);
    /// ```
    pub fn correspondence_estimator<'b, CE, TG>(
        self,
        correspondence_estimator: CE,
    ) -> Icp<'b, T, M, TG, CE, ET, IC, D, FP, RO>
    where
        CE: CorrespondenceEstimator<'b, T, TG, D>,
    {
        Icp {
            max_iterations: self.max_iterations,
            correspondence_estimator,
            estimate_step_transform: self.estimate_step_transform,
            is_converged: self.is_converged,
            filter_points: self.filter_points,
            reject_outliers: self.reject_outliers,

            _marker: PhantomData,
            _lt: &(),
        }
    }
}

impl<'a, T, M, TG, CE, IC, const D: usize, FP, RO> Icp<'a, T, M, TG, CE, (), IC, D, FP, RO>
where
    T: Scalar + RealField + Float + One + Zero + Debug,
    f32: From<T>,
    M: One
        + Clone
        + Mul<Point<T, D>, Output = Point<T, D>>
        + Mul<SVector<T, D>, Output = SVector<T, D>>
        + Mul<M, Output = M>,
    FP: PointFilter<T, D>,
    RO: OutlierRejector<T, D>,
{
    /// `estimate_step_transform` is used to estimate the transform that should be applied to the alignee
    /// point cloud to match the target. This method is required to be called before [`Icp::estimate_transform`].
    ///
    /// Please check the module [`crate::transform_estimation`] for all built-in transform estimators.
    ///
    /// ## Example
    ///
    /// ```
    /// # use modern_icp::{Icp, PointCloud};
    /// # use modern_icp::correspondence::{CorrespondenceEstimator, NearestNeighbor};
    /// # use modern_icp::transform_estimation::point_to_plane_lls;
    /// # use modern_icp::convergence::same_squared_distance_error;
    /// #
    /// # let alignee_cloud = PointCloud::<f32, 3>::new();
    /// # let target_cloud = PointCloud::<f32, 3>::new();
    /// #
    /// let (alignee_transform, error_sum) = Icp::new()
    ///     .correspondence_estimator(NearestNeighbor::new(&target_cloud))
    ///     .estimate_step_transform(point_to_plane_lls::estimate_isometry)
    ///     .is_converged(same_squared_distance_error(0.1))
    ///     .estimate_transform(alignee_cloud, &target_cloud);
    /// ```
    pub fn estimate_step_transform<ET>(
        self,
        estimate_step_transform: ET,
    ) -> Icp<'a, T, M, TG, CE, ET, IC, D, FP, RO>
    where
        // ET: TransformEstimator<T, M, D>,
        ET: FnMut(&mut MaskedPointCloud<T, 3>, &mut MaskedPointCloud<T, 3>, usize) -> Option<M>,
    {
        Icp {
            max_iterations: self.max_iterations,
            correspondence_estimator: self.correspondence_estimator,
            estimate_step_transform,
            is_converged: self.is_converged,
            filter_points: self.filter_points,
            reject_outliers: self.reject_outliers,

            _marker: PhantomData,
            _lt: self._lt,
        }
    }
}

impl<'a, T, M, TG, CE, ET, const D: usize, FP, RO> Icp<'a, T, M, TG, CE, ET, (), D, FP, RO>
where
    T: Scalar + RealField + Float + One + Zero + Debug,
    f32: From<T>,
    M: One
        + Clone
        + Mul<Point<T, D>, Output = Point<T, D>>
        + Mul<SVector<T, D>, Output = SVector<T, D>>
        + Mul<M, Output = M>,
    FP: PointFilter<T, D>,
    RO: OutlierRejector<T, D>,
{
    /// This method sets the convergence criterion for the ICP algorithm. If the convergence criterion is met, the algorithm will stop iterating.
    /// Otherwise it runs for `max_iterations` before stopping. This method is required to be called before [`Icp::estimate_transform`].
    ///
    /// Please check the module [`crate::convergence`] for all built-in convergence criteria.
    ///
    /// ## Example
    ///
    /// ```
    /// # use modern_icp::{Icp, PointCloud};
    /// # use modern_icp::correspondence::{CorrespondenceEstimator, NearestNeighbor};
    /// # use modern_icp::transform_estimation::point_to_plane_lls;
    /// # use modern_icp::convergence::same_squared_distance_error;
    /// #
    /// # let alignee_cloud = PointCloud::<f32, 3>::new();
    /// # let target_cloud = PointCloud::<f32, 3>::new();
    /// #
    /// let (alignee_transform, error_sum) = Icp::new()
    ///     .correspondence_estimator(NearestNeighbor::new(&target_cloud))
    ///     .estimate_step_transform(point_to_plane_lls::estimate_isometry)
    ///     .is_converged(same_squared_distance_error(0.1))
    ///     .estimate_transform(alignee_cloud, &target_cloud);
    /// ```
    pub fn is_converged<IC>(self, is_converged: IC) -> Icp<'a, T, M, TG, CE, ET, IC, D, FP, RO>
    where
        IC: FnMut(&[T], &[T], &M, &mut T, usize) -> bool,
    {
        Icp {
            max_iterations: self.max_iterations,
            correspondence_estimator: self.correspondence_estimator,
            estimate_step_transform: self.estimate_step_transform,
            is_converged,
            filter_points: self.filter_points,
            reject_outliers: self.reject_outliers,

            _marker: PhantomData,
            _lt: self._lt,
        }
    }
}

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
#[allow(clippy::too_many_arguments)]
#[deprecated(since = "0.12.0", note = "use `Icp::new()` + builder methods instead")]
#[inline]
pub fn estimate_transform<'a, T, M, TG, CE, FP, RO, ET, IC>(
    alignee: impl ToPointCloud<T, 3>,
    target: &'a TG,
    max_iterations: usize,
    correspondence_estimator: CE,
    filter_points: FP,
    reject_outliers: RO,
    estimate_step_transform: ET,
    is_converged: IC,
) -> (M, T)
where
    T: Scalar + RealField + Float + One + Zero + Debug,
    f32: From<T>,
    M: One
        + Clone
        + Mul<Point<T, 3>, Output = Point<T, 3>>
        + Mul<SVector<T, 3>, Output = SVector<T, 3>>
        + Mul<M, Output = M>,
    CE: CorrespondenceEstimator<'a, T, TG, 3>,
    FP: FnMut(&PointCloudPoint<T, 3>) -> bool,
    RO: FnMut(&mut MaskedPointCloud<T, 3>, &mut MaskedPointCloud<T, 3>, &[T]) -> Vec<bool>,
    ET: FnMut(&mut MaskedPointCloud<T, 3>, &mut MaskedPointCloud<T, 3>, usize) -> Option<M>,
    IC: FnMut(&[T], &[T], &M, &mut T, usize) -> bool,
{
    Icp::new()
        .max_iterations(max_iterations)
        .correspondence_estimator(correspondence_estimator)
        .filter_points(filter_points)
        .reject_outliers(reject_outliers)
        .estimate_step_transform(estimate_step_transform)
        .is_converged(is_converged)
        .estimate_transform(alignee.to_point_cloud(), target)
}
