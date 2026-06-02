mod keep_all;
mod reject_n_sigma_dist;
mod reject_overlapping_ratios;

pub use keep_all::*;
pub use reject_n_sigma_dist::*;
pub use reject_overlapping_ratios::*;

use nalgebra::Scalar;
use std::fmt::Debug;

use crate::MaskedPointCloud;

pub trait OutlierRejector<T, const D: usize>
where
    T: Debug + Scalar + Copy,
{
    fn reject(
        &mut self,
        x: &mut MaskedPointCloud<T, D>,
        y: &mut MaskedPointCloud<T, D>,
        distances: &[T],
    ) -> Vec<bool>;
}

impl<F, T, const D: usize> OutlierRejector<T, D> for F
where
    F: FnMut(&mut MaskedPointCloud<T, D>, &mut MaskedPointCloud<T, D>, &[T]) -> Vec<bool>,
    T: Debug + Scalar + Copy,
{
    fn reject(
        &mut self,
        x: &mut MaskedPointCloud<T, D>,
        y: &mut MaskedPointCloud<T, D>,
        distances: &[T],
    ) -> Vec<bool> {
        self(x, y, distances)
    }
}

/// Default implementation that doesn't reject any points.
///
/// Use [`keep_all`] instead if you want to use this explicitly in your own code.
pub struct KeepAll;

impl<T, const D: usize> OutlierRejector<T, D> for KeepAll
where
    T: Debug + Scalar + Copy,
{
    fn reject(
        &mut self,
        x: &mut MaskedPointCloud<T, D>,
        y: &mut MaskedPointCloud<T, D>,
        distances: &[T],
    ) -> Vec<bool> {
        keep_all(x, y, distances)
    }
}
