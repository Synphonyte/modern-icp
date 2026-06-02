mod above_planes;
mod accept_all;

pub use above_planes::*;
pub use accept_all::*;
use nalgebra::Scalar;

use crate::PointCloudPoint;

pub trait PointFilter<T, const D: usize>
where
    T: Scalar + Copy,
{
    fn filter(&mut self, point: &PointCloudPoint<T, D>) -> bool;
}

impl<F, T, const D: usize> PointFilter<T, D> for F
where
    F: FnMut(&PointCloudPoint<T, D>) -> bool,
    T: Scalar + Copy,
{
    fn filter(&mut self, point: &PointCloudPoint<T, D>) -> bool {
        self(point)
    }
}

/// Default implementation that accepts all points.
///
/// Use [`accept_all`] instead if you want to use this explicitly in your own code.
pub struct AcceptAll;

impl<T, const D: usize> PointFilter<T, D> for AcceptAll
where
    T: Scalar + Copy,
{
    fn filter(&mut self, point: &PointCloudPoint<T, D>) -> bool {
        accept_all(point)
    }
}
