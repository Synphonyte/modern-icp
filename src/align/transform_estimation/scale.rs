use crate::{MaskedPointCloud, compute_centroid};
use nalgebra::{allocator::Allocator, *};

/// Estimates the non-uniform scale between the alignee and the target.
///
/// See [the math](https://www.mathcha.io/editor/2OV22T3mi5DtZw4yWZFV82O7DcDQvJYBhQWyNl0?embedded=true)
pub fn estimate_scale_vector<T, const D: usize>(
    alignee: &mut MaskedPointCloud<T, D>,
    target: &mut MaskedPointCloud<T, D>,
) -> Option<SVector<T, D>>
where
    T: Scalar + RealField + Copy,
{
    if alignee.is_empty() || target.is_empty() {
        return None;
    }

    let numerator = alignee
        .iter()
        .zip(target.iter())
        .fold(SVector::<T, D>::zeros(), |r, (a, t)| {
            r + a.pos.coords.component_mul(&t.pos.coords)
        });

    let denominator = alignee.iter().fold(SVector::<T, D>::zeros(), |r, a| {
        r + a.pos.coords.component_mul(&a.pos.coords)
    });

    Some(numerator.component_div(&denominator))
}

/// Estimates the non-uniform scale and translation between the alignee and the target.
///
/// Solves per-component linear regression: `t_i ≈ s * a_i + b`, minimizing `‖s * a_i + b - t_i‖²`.
/// Returns `(scale, translation)` as `(SVector<T, D>, SVector<T, D>)`.
pub fn estimate_scale_translation_vector<T, const D: usize>(
    alignee: &mut MaskedPointCloud<T, D>,
    target: &mut MaskedPointCloud<T, D>,
) -> Option<(SVector<T, D>, SVector<T, D>)>
where
    T: Scalar + RealField + Copy,
{
    if alignee.is_empty() || target.is_empty() {
        return None;
    }

    let n = T::from_usize(alignee.len())?;

    let alignee_centroid = compute_centroid(alignee.points_iter());
    let target_centroid = compute_centroid(target.points_iter());

    // numerator:   Σ a_i · t_i  -  n * ā · t̄
    // denominator: Σ a_i · a_i  -  n * ā · ā
    let (sum_at, sum_aa) = alignee.iter().zip(target.iter()).fold(
        (SVector::<T, D>::zeros(), SVector::<T, D>::zeros()),
        |(at, aa), (a, t)| {
            (
                at + a.pos.coords.component_mul(&t.pos.coords),
                aa + a.pos.coords.component_mul(&a.pos.coords),
            )
        },
    );

    let numerator = sum_at - alignee_centroid.component_mul(&target_centroid) * n;
    let denominator = sum_aa - alignee_centroid.component_mul(&alignee_centroid) * n;

    // Avoid division by zero per component
    let scale = SVector::<T, D>::from_fn(|i, _| {
        let d = denominator[i];
        if d == T::zero() {
            T::one()
        } else {
            numerator[i] / d
        }
    });

    let translation = target_centroid - scale.component_mul(&alignee_centroid);

    Some((scale, translation))
}

/// Estimates the non-uniform scale and translation between the alignee and the target,
/// returned as a `Transform<T, TAffine, D>` (homogeneous matrix).
///
/// The homogeneous matrix encodes `x ↦ diag(s) * x + b`:
/// scale on the diagonal, translation in the last column.
#[inline]
pub fn estimate_scale_translation<T, const D: usize>(
    alignee: &mut MaskedPointCloud<T, D>,
    target: &mut MaskedPointCloud<T, D>,
    _: usize,
) -> Option<Transform<T, TAffine, D>>
where
    T: Scalar + RealField + Copy,
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    let (scale, translation) = estimate_scale_translation_vector(alignee, target)?;

    let mut mat = OMatrix::<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>::identity();

    for i in 0..D {
        mat[(i, i)] = scale[i];
        mat[(i, D)] = translation[i];
    }

    Some(Transform::from_matrix_unchecked(mat))
}

#[inline]
pub fn estimate_scale<T, const D: usize>(
    alignee: &mut MaskedPointCloud<T, D>,
    target: &mut MaskedPointCloud<T, D>,
    _: usize,
) -> Option<SMatrix<T, D, D>>
where
    T: Scalar + RealField + Copy,
{
    estimate_scale_vector(alignee, target).map(|vec| SMatrix::from_diagonal(&vec))
}
