use crate::correspondence::{
    CorrespondenceEstimator, Correspondences, get_ordered_correspondences_and_distances_nn,
};
use crate::{PointCloud, PointCloudPoint, ToPointCloud, kd_tree_of_point_cloud};
use kdtree::KdTree;
use nalgebra::{RealField, Scalar};
use num_traits::{Float, One, Zero};

pub struct NearestNeighbor<T>
where
    T: Scalar + RealField + Float + One + Zero,
{
    tree: KdTree<T, usize, Vec<T>>,
    target_cloud: PointCloud<T, 3>,
}

impl<'a, T, PC> CorrespondenceEstimator<'a, T, PC, 3> for NearestNeighbor<T>
where
    T: Scalar + RealField + Float + One + Zero,
    PC: ToPointCloud<T, 3>,
{
    fn new(target: &'a PC) -> Self {
        let target_cloud = target.to_point_cloud();

        NearestNeighbor {
            tree: kd_tree_of_point_cloud(&target_cloud),
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
        let (alignee_point_cloud, corresponding_points_iter, distances) =
            get_ordered_correspondences_and_distances_nn(
                &self.tree,
                alignee,
                &self.target_cloud,
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
