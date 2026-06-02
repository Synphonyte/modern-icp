use std::fmt::Debug;

use nalgebra::Scalar;

use crate::MaskedPointCloud;

pub mod affine_transformation;
pub mod point_to_plane_lls;
pub mod point_to_plane_lls_weighted;
pub mod scale;
pub mod svd;
pub mod transform_modifier;
pub mod translation_point_to_plane;

pub trait TransformEstimator<T, M, const D: usize>
where
    T: Debug + Scalar + Copy,
{
    fn estimate<'a, 'b, 'c>(
        &'a mut self,
        alignee: &'b mut MaskedPointCloud<T, D>,
        target: &'c mut MaskedPointCloud<T, D>,
        step: usize,
    ) -> Option<M>;
}

impl<F, T, M, const D: usize> TransformEstimator<T, M, D> for F
where
    F: FnMut(&mut MaskedPointCloud<T, D>, &mut MaskedPointCloud<T, D>, usize) -> Option<M>,
    T: Debug + Scalar + Copy,
{
    fn estimate<'a, 'b, 'c>(
        &'a mut self,
        alignee: &'b mut MaskedPointCloud<T, D>,
        target: &'c mut MaskedPointCloud<T, D>,
        step: usize,
    ) -> Option<M> {
        self(alignee, target, step)
    }
}
