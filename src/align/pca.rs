use crate::{compute_centroid, demean_into_matrix};
use nalgebra::{
    Const, DefaultAllocator, DimSub, OVector, Point, RealField, Scalar, SymmetricEigen, ToTypenum,
    allocator::Allocator, partial_cmp,
};

/// Computes the principal component analysis of the point cloud.
///
/// For a detailed explanation of PCA, please see [this article](https://www.baeldung.com/cs/principal-component-analysis).
pub fn compute_principal_component_analysis<T, const D: usize>(
    points: impl Iterator<Item = Point<T, D>> + Clone,
) -> Vec<OVector<T, Const<D>>>
where
    T: Scalar + RealField + Copy,
    Const<D>: ToTypenum + DimSub<Const<1>>,
    DefaultAllocator: Allocator<<Const<D> as DimSub<Const<1>>>::Output>,
{
    let centroid = compute_centroid(points.clone());

    compute_principal_component_analysis_with_centroid(points, &centroid)
}

pub fn compute_principal_component_analysis_with_centroid<T, const D: usize>(
    points: impl Iterator<Item = Point<T, D>>,
    centroid: &OVector<T, Const<D>>,
) -> Vec<OVector<T, Const<D>>>
where
    T: Scalar + RealField + Copy,
    Const<D>: ToTypenum + DimSub<Const<1>>,
    DefaultAllocator: Allocator<<Const<D> as DimSub<Const<1>>>::Output>,
{
    let demeaned = demean_into_matrix(points, &centroid);

    // TODO : this is a symmetric matrix and eigen decomp only looks at one half (see nalgebra docs) => only compute that half.
    let covariant_matrix = &demeaned * &demeaned.transpose();

    let eigen_decomp = SymmetricEigen::<T, Const<D>>::new(covariant_matrix);

    let mut eigen_vv: Vec<_> = eigen_decomp
        .eigenvectors
        .column_iter()
        .zip(eigen_decomp.eigenvalues.iter())
        .collect();

    eigen_vv.sort_unstable_by(|(_, v1), (_, v2)| partial_cmp(*v2, *v1).unwrap());

    eigen_vv.into_iter().map(|(vec, _)| vec.into()).collect()
}
