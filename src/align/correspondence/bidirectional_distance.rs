use crate::correspondence::{
    CorrespondenceEstimator, Correspondences, get_ordered_correspondences_and_distances_nn,
};
use crate::{MaskedPointCloud, PointCloud, kd_tree_of_point_cloud};
use kdtree::KdTree;
use nalgebra::{RealField, Scalar};
use num_traits::{Float, One, Zero};

/// Computes the correspondences between the alignee and the target using the
/// Bidirectional Distance algorithm.
///
/// See this [paper from Dong et al.](https://doi.org/10.1049/iet-cvi.2016.0058)
pub struct BidirectionalDistance<'a, T>
where
    T: Scalar + RealField + Float + One + Zero,
{
    target_tree: KdTree<T, usize, &'a [T]>,
}

impl<'a, T> CorrespondenceEstimator<'a, T, PointCloud<T, 3>, 3> for BidirectionalDistance<'a, T>
where
    T: Scalar + RealField + Float + One + Zero,
{
    fn new(target: &'a PointCloud<T, 3>) -> Self {
        BidirectionalDistance {
            target_tree: kd_tree_of_point_cloud(&target),
        }
    }

    fn find_correspondences<'b>(
        &self,
        alignee: &'b PointCloud<T, 3>,
        target: &'b PointCloud<T, 3>,
    ) -> Correspondences<'b, T, 3> {
        let (corresponding_target_point_cloud, alignee_to_target_distances) =
            get_ordered_correspondences_and_distances_nn(&self.target_tree, alignee, target);

        let (corresponding_alignee_point_cloud, target_to_alignee_distances) =
            get_ordered_correspondences_and_distances_nn(
                &kd_tree_of_point_cloud(&alignee),
                target,
                alignee,
            );

        Correspondences {
            alignee_point_cloud: MaskedPointCloud::new(&alignee),
            corresponding_target_point_cloud,
            target_point_cloud: MaskedPointCloud::new(&target),
            corresponding_alignee_point_cloud,
            alignee_to_target_distances,
            target_to_alignee_distances,
        }
    }
}
