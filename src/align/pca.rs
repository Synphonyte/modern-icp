use crate::{compute_centroid, demean_into_matrix, PointCloudIterator};
use nalgebra::{partial_cmp, Const, RealField, Scalar, SymmetricEigen, Vector3};

/// Computes the principal component analysis of the point cloud.
///
/// For a detailed explanation of PCA, please see [this article](https://www.baeldung.com/cs/principal-component-analysis).
pub fn compute_principal_component_analysis<T>(
    mut point_cloud_iter: &mut PointCloudIterator<T, 3>,
) -> Vec<Vector3<T>>
where
    T: Scalar + RealField + Copy,
{
    let centroid = compute_centroid(&mut point_cloud_iter);

    compute_principal_component_analysis_with_centroid(&mut point_cloud_iter, &centroid)
}

pub fn compute_principal_component_analysis_with_centroid<T>(
    mut point_cloud_iter: &mut PointCloudIterator<T, 3>,
    centroid: &Vector3<T>,
) -> Vec<Vector3<T>>
where
    T: Scalar + RealField + Copy,
{
    let demeaned = demean_into_matrix(&mut point_cloud_iter, &centroid);

    // TODO : this is a symmetric matrix and eigen decomp only looks at one half (see nalgebra docs) => only compute that half.
    let covariant_matrix = &demeaned * &demeaned.transpose();

    let eigen_decomp = SymmetricEigen::<T, Const<3>>::new(covariant_matrix);

    let mut eigen_vv: Vec<_> = eigen_decomp
        .eigenvectors
        .column_iter()
        .zip(eigen_decomp.eigenvalues.iter())
        .collect();

    eigen_vv.sort_unstable_by(|(_, v1), (_, v2)| partial_cmp(*v2, *v1).unwrap());

    eigen_vv
        .iter()
        .map(|(vec, _)| Vector3::from(*vec))
        .collect()
}
