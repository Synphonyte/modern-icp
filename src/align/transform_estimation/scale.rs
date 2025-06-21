use crate::MaskedPointCloud;
use nalgebra::*;

/// Estimates the non-uniform scale between the alignee and the target.
///
/// See [the math](https://www.mathcha.io/editor/2OV22T3mi5DtZw4yWZFV82O7DcDQvJYBhQWyNl0?embedded=true)
pub fn estimate_scale_vector<T, const D: usize>(
    alignee: &mut MaskedPointCloud<T, D>,
    target: &mut MaskedPointCloud<T, D>,
) -> SVector<T, D>
where
    T: Scalar + RealField + Copy,
{
    let numerator = alignee
        .iter()
        .zip(target.iter())
        .fold(SVector::<T, D>::zeros(), |r, (a, t)| {
            r + a.pos.coords.component_mul(&t.pos.coords)
        });

    let denominator = alignee.iter().fold(SVector::<T, D>::zeros(), |r, a| {
        r + a.pos.coords.component_mul(&a.pos.coords)
    });

    numerator.component_div(&denominator)
}

pub fn estimate_scale<T, const D: usize>(
    alignee: &mut MaskedPointCloud<T, D>,
    target: &mut MaskedPointCloud<T, D>,
    _: usize,
) -> SMatrix<T, D, D>
where
    T: Scalar + RealField + Copy,
{
    let vec = estimate_scale_vector(alignee, target);
    SMatrix::from_diagonal(&vec)
}
