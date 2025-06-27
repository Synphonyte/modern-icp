use nalgebra::{Point, SVector, Scalar};

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointCloudPoint<T: Scalar + Copy, const D: usize> {
    pub pos: Point<T, D>,
    pub norm: Option<SVector<T, D>>,
}

impl<T: Scalar + Copy, const D: usize> From<Point<T, D>> for PointCloudPoint<T, D> {
    fn from(p: Point<T, D>) -> Self {
        PointCloudPoint { pos: p, norm: None }
    }
}

impl<T: Scalar + Copy, const D: usize> From<SVector<T, D>> for PointCloudPoint<T, D> {
    fn from(p: SVector<T, D>) -> Self {
        PointCloudPoint {
            pos: Point::from(p),
            norm: None,
        }
    }
}
