use nalgebra::{Point, Scalar};

use super::PointCloud;

pub trait ToPointCloud<T, const D: usize>
where
    T: Scalar + Copy,
{
    fn to_point_cloud(&self) -> PointCloud<T, D>;
}

impl<T, const D: usize> ToPointCloud<T, D> for PointCloud<T, D>
where
    T: Scalar + Copy,
{
    fn to_point_cloud(&self) -> PointCloud<T, D> {
        self.clone()
    }
}

impl<T, const D: usize> ToPointCloud<T, D> for Vec<Point<T, D>>
where
    T: Scalar + Copy,
{
    fn to_point_cloud(&self) -> PointCloud<T, D> {
        self.iter().map(|point| (*point).into()).collect()
    }
}
