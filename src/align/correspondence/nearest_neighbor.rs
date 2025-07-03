use crate::correspondence::{
    CorrespondenceEstimator, Correspondences, get_ordered_correspondences_and_distances_nn,
};
use crate::{PointCloud, PointCloudPoint, kd_tree_of_point_cloud};
use kdtree::KdTree;
use nalgebra::{RealField, Scalar};
use num_traits::{Float, One, Zero};

pub struct NearestNeighbor<'a, T>
where
    T: Scalar + RealField + Float + One + Zero,
{
    tree: KdTree<T, usize, &'a [T]>,
}

impl<'a, T> CorrespondenceEstimator<'a, T, PointCloud<T, 3>, 3> for NearestNeighbor<'a, T>
where
    T: Scalar + RealField + Float + One + Zero,
{
    fn new(target: &'a PointCloud<T, 3>) -> Self {
        NearestNeighbor {
            tree: kd_tree_of_point_cloud(target),
        }
    }

    fn find_correspondences<'b, FP>(
        &self,
        alignee: &'b PointCloud<T, 3>,
        target: &'b PointCloud<T, 3>,
        filter_points: &mut FP,
    ) -> Correspondences<'b, T, 3>
    where
        FP: FnMut(&PointCloudPoint<T, 3>) -> bool,
    {
        let (alignee_point_cloud, corresponding_points_iter, distances) =
            get_ordered_correspondences_and_distances_nn(
                &self.tree,
                alignee,
                target,
                filter_points,
            );

        Correspondences::from_simple_one_way_correspondences(
            alignee_point_cloud,
            alignee,
            corresponding_points_iter,
            distances,
        )
    }
}
