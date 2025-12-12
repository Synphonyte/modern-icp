use crate::{MaskedPointCloud, PointCloudPoint, compute_centroid};
use nalgebra::*;

/// Estimates the affine transformation between the alignee and the target.
///
/// See this [paper from Dong et al.](https://doi.org/10.1049/iet-cvi.2016.0058)
pub fn estimate_affine<T>(
    alignee: &mut MaskedPointCloud<T, 3>,
    target: &mut MaskedPointCloud<T, 3>,
    _: usize,
) -> Option<Affine3<T>>
where
    T: Scalar + RealField + From<f32> + Copy,
{
    let subtract_mean = |mean_value: Point3<T>| move |p: &PointCloudPoint<T, 3>| p.pos - mean_value;

    let mean_value_alignee = Point3::from(compute_centroid(alignee.points_iter()));
    let demeaned_alignee: Vec<Vector3<T>> = alignee
        .iter()
        .map(subtract_mean(mean_value_alignee))
        .collect();

    let mean_value_target = Point3::from(compute_centroid(target.points_iter()));
    let demeaned_target: Vec<Vector3<T>> = target
        .iter()
        .map(subtract_mean(mean_value_target))
        .collect();

    let vec_sum2: Matrix3<T> = demeaned_alignee
        .iter()
        .zip(demeaned_target.iter())
        .fold(Matrix3::zeros(), |m, (a, b)| m + b * a.transpose());
    let matrix_sum_inv: Matrix3<T> = demeaned_alignee
        .iter()
        .fold(Matrix3::zeros(), |m, a| m + a * a.transpose());

    matrix_sum_inv.try_inverse().map(|matrix_sum| {
        #[allow(non_snake_case)]
        let A = matrix_sum * vec_sum2;

        let count = T::from(alignee.len() as f32);

        let translation = mean_value_target
            - alignee
                .iter()
                .fold(Vector3::zeros(), |v, pcp| v + A * pcp.pos.coords)
                * T::one()
                / count;

        #[allow(non_snake_case)]
        let mut M = A
            .insert_fixed_rows::<1>(3, T::zero())
            .insert_fixed_columns::<1>(3, T::zero());

        M[12] = translation.x;
        M[13] = translation.y;
        M[14] = translation.z;
        M[15] = T::one();

        Affine3::from_matrix_unchecked(M)
    })
}
