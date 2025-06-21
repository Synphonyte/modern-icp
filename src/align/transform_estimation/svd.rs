use crate::{compute_centroid, demean_into_matrix, MaskedPointCloud};
use nalgebra::*;

/// Estimates the isometry between the alignee and the target using the SVD algorithm.
///
/// See this [implementation of the algorithm from PointCloudLibrary](https://github.com/PointCloudLibrary/pcl/blob/d242fcbdbb53efc7de48c9159343432a2194a27c/registration/include/pcl/registration/impl/transformation_estimation_svd.hpp)
pub fn estimate_isometry<T>(
    alignee: &mut MaskedPointCloud<T, 3>,
    target: &mut MaskedPointCloud<T, 3>,
    _: usize,
) -> IsometryMatrix3<T>
where
    T: Scalar + RealField + Copy,
{
    let alignee_centroid = compute_centroid(alignee.points_iter());
    let target_centroid = compute_centroid(target.points_iter());

    let demeaned_alignee = demean_into_matrix(alignee.points_iter(), &alignee_centroid);
    let demeaned_target = demean_into_matrix(target.points_iter(), &target_centroid);

    let covariant_matrix = demeaned_alignee * demeaned_target.transpose();

    let SVD {
        u,
        v_t,
        singular_values: _,
    } = covariant_matrix.svd(true, true);

    let u = u.unwrap();
    let mut v = v_t.unwrap().transpose();

    if u.determinant() * v.determinant() < T::zero() {
        let mut column = v.column_mut(2);
        column *= -T::one();
    }

    let rotation_matrix = v * u.transpose();
    let rotation = Rotation3::from_matrix_unchecked(rotation_matrix);

    let translation = Translation3::from(target_centroid - rotation_matrix * alignee_centroid);

    IsometryMatrix3::from_parts(translation, rotation)
}

pub fn estimate_similarity<T>(
    source: &mut MaskedPointCloud<T, 3>,
    target: &mut MaskedPointCloud<T, 3>,
) -> SimilarityMatrix3<T>
where
    T: Scalar + RealField + Copy,
    // &'a T: Mul<&'a T, Output=&'a T>,
{
    let source_centroid = compute_centroid(source.points_iter());
    let target_centroid = compute_centroid(target.points_iter());

    let demeaned_source = demean_into_matrix(source.points_iter(), &source_centroid);
    let demeaned_target = demean_into_matrix(target.points_iter(), &target_centroid);

    let covariant_matrix = &demeaned_source * demeaned_target.transpose();

    let SVD {
        u,
        v_t,
        singular_values: _,
    } = covariant_matrix.svd(true, true);

    let u = u.unwrap();
    let mut v = v_t.unwrap().transpose();

    if u.determinant() * v.determinant() < T::zero() {
        let mut column = v.column_mut(2);
        column *= -T::one();
    }

    let rotation_matrix = v * u.transpose();

    let rotated_source = rotation_matrix * &demeaned_source;

    let mut sum_ss = T::zero();
    let mut sum_tt = T::zero();

    for ((dem_src, dem_tgt), rot_src) in demeaned_source
        .iter()
        .zip(demeaned_target.iter())
        .zip(rotated_source.iter())
    {
        sum_ss += (*dem_src) * (*dem_src);
        sum_tt += (*dem_tgt) * (*rot_src);
    }

    let scale = sum_tt / sum_ss;

    let rotation = Rotation3::from_matrix_unchecked(rotation_matrix);

    let translation =
        Translation3::from(target_centroid - rotation_matrix * source_centroid * scale);

    SimilarityMatrix3::from_parts(translation, rotation, scale)
}
