use nalgebra::*;
use crate::PointCloudIterator;

/// Estimates the non-uniform scale between the alignee and the target.
///
/// See [the math](https://www.mathcha.io/editor/2OV22T3mi5DtZw4yWZFV82O7DcDQvJYBhQWyNl0?embedded=true)
pub fn estimate_scale_vector<T, const D: usize>(
    alignee: &mut PointCloudIterator<T, D>,
    target: &mut PointCloudIterator<T, D>,
) -> SVector<T, D>
where
    T: Scalar + RealField + Copy,
{
    alignee.reset_iter();
    target.reset_iter();

    let numerator = alignee
        .zip(target)
        .fold(SVector::<T, D>::zeros(), |r, (a, t)| {
            r + a.pos.coords.component_mul(&t.pos.coords)
        });

    alignee.reset_iter();

    let denominator = alignee.fold(SVector::<T, D>::zeros(), |r, a| {
        r + a.pos.coords.component_mul(&a.pos.coords)
    });

    numerator.component_div(&denominator)
}

pub fn estimate_scale<T, const D: usize>(
    alignee: &mut PointCloudIterator<T, D>,
    target: &mut PointCloudIterator<T, D>,
    _: usize,
) -> SMatrix<T, D, D>
where
    T: Scalar + RealField + Copy,
{
    let vec = estimate_scale_vector(alignee, target);
    SMatrix::from_diagonal(&vec)
}
