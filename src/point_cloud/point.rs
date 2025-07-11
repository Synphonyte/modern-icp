use nalgebra::{Point, SVector, Scalar};
use num_traits::One;

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointCloudPoint<T: Scalar + Copy, const D: usize> {
    pub pos: Point<T, D>,
    pub norm: Option<SVector<T, D>>,
    pub weight: T,
}

impl<T: Scalar + Copy + One, const D: usize> PointCloudPoint<T, D> {
    pub fn from_pos(pos: Point<T, D>) -> Self {
        PointCloudPoint {
            pos,
            norm: None,
            weight: T::one(),
        }
    }

    pub fn from_pos_norm(pos: Point<T, D>, norm: SVector<T, D>) -> Self {
        PointCloudPoint {
            pos,
            norm: Some(norm),
            weight: T::one(),
        }
    }
}

impl<T: Scalar + Copy + One, const D: usize> From<Point<T, D>> for PointCloudPoint<T, D> {
    fn from(p: Point<T, D>) -> Self {
        PointCloudPoint {
            pos: p,
            norm: None,
            weight: T::one(),
        }
    }
}

impl<T: Scalar + Copy + One, const D: usize> From<SVector<T, D>> for PointCloudPoint<T, D> {
    fn from(p: SVector<T, D>) -> Self {
        PointCloudPoint {
            pos: Point::from(p),
            norm: None,
            weight: T::one(),
        }
    }
}
