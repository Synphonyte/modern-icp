use crate::correspondence::{
    get_ordered_correspondences_and_distances_nn, CorrespondenceEstimator, Correspondences,
};
use crate::{kd_tree_of_point_cloud, PointCloud, PointCloudPoint, ToPointCloud};
use kdtree::KdTree;
use nalgebra::{RealField, Scalar};
use num_traits::{Float, One, Zero};

/// Computes the correspondences between the alignee and the target using the
/// Bidirectional Distance algorithm.
///
/// See this [paper from Dong et al.](https://doi.org/10.1049/iet-cvi.2016.0058)
pub struct BidirectionalDistance<T>
where
    T: Scalar + RealField + Float + One + Zero,
{
    target_tree: KdTree<T, usize, Vec<T>>,
    target_cloud: PointCloud<T, 3>,
}

impl<'a, T, PC> CorrespondenceEstimator<'a, T, PC, 3> for BidirectionalDistance<T>
where
    T: Scalar + RealField + Float + One + Zero,
    PC: ToPointCloud<T, 3>,
{
    fn new(target: &'a PC) -> Self {
        let target_cloud = target.to_point_cloud();

        BidirectionalDistance {
            target_tree: kd_tree_of_point_cloud(&target_cloud),
            target_cloud,
        }
    }

    fn find_correspondences<'b, 't, FP>(
        &'t self,
        alignee: &'b PointCloud<T, 3>,
        _target: &'b PC,
        filter_points: &mut FP,
    ) -> Correspondences<'b, 't, T, 3>
    where
        FP: FnMut(&PointCloudPoint<T, 3>) -> bool,
        'b: 't,
    {
        let (alignee_point_cloud, corresponding_target_point_cloud, alignee_to_target_distances) =
            get_ordered_correspondences_and_distances_nn(
                &self.target_tree,
                alignee,
                &self.target_cloud,
                filter_points,
            );

        let (target_point_cloud, corresponding_alignee_point_cloud, target_to_alignee_distances) =
            get_ordered_correspondences_and_distances_nn(
                &kd_tree_of_point_cloud(alignee),
                &self.target_cloud,
                alignee,
                filter_points,
            );

        Correspondences {
            alignee_point_cloud,
            corresponding_target_point_cloud,
            target_point_cloud,
            corresponding_alignee_point_cloud,
            alignee_to_target_distances,
            target_to_alignee_distances,
        }
    }
}
