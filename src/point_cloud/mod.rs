mod iterator;
mod masked;
mod point;
mod traits;

pub use iterator::*;
use kdtree::KdTree;
pub use masked::*;
use nalgebra::{Point3, RealField, Scalar};
use num_traits::{Float, One, Zero};
pub use point::*;
pub use traits::*;

pub type PointCloud<T, const D: usize> = Vec<PointCloudPoint<T, D>>;

pub fn point_cloud_from_position_slice<T: Scalar + Copy>(slice: &[T]) -> PointCloud<T, 3> {
    let mut cloud = PointCloud::with_capacity(slice.len() / 3);

    for i in (0..slice.len()).step_by(3) {
        cloud.push(PointCloudPoint {
            pos: Point3::new(slice[i], slice[i + 1], slice[i + 2]),
            norm: None,
        })
    }

    cloud
}

pub fn kd_tree_of_point_cloud<T, const D: usize>(
    point_cloud: &PointCloud<T, D>,
) -> KdTree<T, usize, Vec<T>>
where
    T: Scalar + RealField + Float + One + Zero,
{
    let mut kd_tree = KdTree::new(D);

    for (i, p) in point_cloud.iter().enumerate() {
        kd_tree.add(p.pos.coords.as_slice().to_owned(), i).unwrap();
    }

    kd_tree
}
