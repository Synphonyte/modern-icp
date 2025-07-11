mod bidirectional_distance;
mod cylinder;
mod nearest_neighbor;

use crate::{MaskedPointCloud, PointCloud, PointCloudPoint};
pub use bidirectional_distance::*;
pub use cylinder::*;
use kdtree::KdTree;
use kdtree::distance::squared_euclidean;
use nalgebra::*;
pub use nearest_neighbor::*;
use num_traits::{Float, One, Zero};
use std::fmt::Debug;

/// Contains all the correspondences found by the correspondence estimator.
pub struct Correspondences<'a, 't, T, const D: usize>
where
    T: Copy + PartialEq + Debug + 'static,
{
    /// Iterator over the points of the alignee that have a correspondence point in the target
    /// (found in the `corresponding_target_point_cloud`).
    pub alignee_point_cloud: MaskedPointCloud<'a, T, D>,

    /// Every point in this iterator corresponds to the point in the `alignee_point_cloud` with the same index.
    pub corresponding_target_point_cloud: MaskedPointCloud<'t, T, D>,

    /// Iterator over the points of the target that have a corresponcence point in the alignee.
    /// (found in the `corresponding_alignee_point_cloud`).
    /// This is only used by bidrectional distance correspondence estimators.
    pub target_point_cloud: MaskedPointCloud<'t, T, D>,

    /// Every point in this iterator corresponds to the point in the `target_point_cloud` with the same index.
    /// This is only used by bidrectional distance correspondence estimators.
    pub corresponding_alignee_point_cloud: MaskedPointCloud<'a, T, D>,

    /// Distances between the points of the alignee and the corresponding points in the target.
    /// Refers to the points in the `alignee_point_cloud` and `corresponding_target_point_cloud`.
    pub alignee_to_target_distances: Vec<T>,

    /// Distances between the points of the target and the corresponding points in the alignee.
    /// Refers to the points in the `target_point_cloud` and `corresponding_alignee_point_cloud`.
    /// This is only used by bidrectional distance correspondence estimators.
    pub target_to_alignee_distances: Vec<T>,
}

impl<'a, 't, T, const D: usize> Correspondences<'a, 't, T, D>
where
    T: Copy + PartialEq + Debug + 'static,
    'a: 't,
{
    /// Use this for simple one-way correspondence estimators.
    pub fn from_simple_one_way_correspondences(
        alignee_point_cloud: MaskedPointCloud<'a, T, D>,
        alignee: &'a PointCloud<T, D>,
        corresponding_target_point_cloud: MaskedPointCloud<'t, T, D>,
        alignee_to_target_distances: Vec<T>,
    ) -> Self {
        let mut empty_alignee_cloud = MaskedPointCloud::new(alignee);
        empty_alignee_cloud.set_empty();

        // please note that in non-empty cases this point cloud iterator needs to refer to target!
        let mut empty_target_cloud = MaskedPointCloud::new(alignee);
        empty_target_cloud.set_empty();

        Self {
            alignee_point_cloud,
            corresponding_target_point_cloud,
            target_point_cloud: empty_target_cloud,
            corresponding_alignee_point_cloud: empty_alignee_cloud,
            alignee_to_target_distances,
            target_to_alignee_distances: vec![],
        }
    }
}

/// Trait for correspondence estimators.
///
/// The goal of a correspondence estimator is to find corresponding points between the alignee and the target.
pub trait CorrespondenceEstimator<'a, T, TG, const D: usize>
where
    T: Scalar + RealField + Float + One + Zero,
{
    fn new(target: &'a TG) -> Self;

    /// Finds the correspondences between the alignee and the target.
    ///
    /// The target can be any type that the `CorrespondenceEstimator` implementation can handle.
    /// In many cases this will be another `PointCloud`.
    ///
    /// The `filter_points` function is used to filter out points that are not considered for correspondence.
    /// It takes a reference to a `PointCloudPoint` and returns a boolean which is `true` if the point should be included.
    ///
    /// See the [`Correspondences`] return type documentation for more information.
    fn find_correspondences<'b, 't, FP>(
        &'t self,
        alignee: &'b PointCloud<T, D>,
        target: &'b TG,
        filter_points: &mut FP,
    ) -> Correspondences<'b, 't, T, D>
    where
        FP: FnMut(&PointCloudPoint<T, 3>) -> bool,
        'b: 't;
}

/// For every point in `data_set_x` finds the nearest point in `data_set_y` using the KD-Tree `tree`.
/// Returns a masked point_cloud referencing `data_set_x` that contains all points allowed by `filter_points`
/// and another masked point_cloud referencing `data_set_y` that correspond to the points in the first
/// returned masked point_cloud.
/// Finally it returns a list of the distances between the points of the two returned point_clouds.
pub fn get_ordered_correspondences_and_distances_nn<'a, 't, T, FP>(
    tree: &KdTree<T, usize, Vec<T>>,
    data_set_x: &'a PointCloud<T, 3>,
    data_set_y: &'t PointCloud<T, 3>,
    filter_points: &mut FP,
) -> (
    MaskedPointCloud<'a, T, 3>,
    MaskedPointCloud<'t, T, 3>,
    Vec<T>,
)
where
    T: Scalar + RealField + Float + One + Zero,
    FP: FnMut(&PointCloudPoint<T, 3>) -> bool,
{
    let mut distances = vec![];
    let mut ordered_indices = vec![];

    let mut mask = vec![false; data_set_x.len()];

    for (i, p) in data_set_x
        .iter()
        .enumerate()
        .filter(|(_, p)| filter_points(*p))
    {
        let (distance, idx) = tree
            .nearest(p.pos.coords.as_slice(), 1, &squared_euclidean)
            .unwrap()[0];

        ordered_indices.push(*idx);
        distances.push(distance);
        mask[i] = true;
    }

    let mut point_cloud = MaskedPointCloud::new(data_set_x);
    point_cloud.add_mask(&mask);

    let mut corresponding_point_cloud = MaskedPointCloud::new(data_set_y);
    corresponding_point_cloud.add_order(&ordered_indices);

    (point_cloud, corresponding_point_cloud, distances)
}
