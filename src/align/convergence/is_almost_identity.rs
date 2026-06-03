use nalgebra::{
    Const, DefaultAllocator, Dim, DimName, DimNameAdd, DimNameSum, Matrix, OMatrix, Owned,
    RealField, Scalar, Storage, TAffine, Transform, U1, allocator::Allocator,
};

/// Converge if the step affine transform matrix (4x4) is almost the identity matrix.
///
/// Compares each matrix column component-wise with the x/y/z/w basis vector respectively.
pub fn is_almost_identity_affine<T, const D: usize>(
    epsilon: T,
) -> impl Fn(&[T], &[T], &Transform<T, TAffine, D>, &mut T, usize) -> bool
where
    T: Scalar + RealField + Copy,
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    let check_matrix = is_almost_identity_matrix::<
        T,
        Owned<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
        DimNameSum<Const<D>, U1>,
    >(epsilon);

    move |a: &[T], b: &[T], transform: &Transform<T, TAffine, D>, t: &mut T, i: usize| {
        check_matrix(a, b, transform.matrix(), t, i)
    }
}

#[allow(clippy::type_complexity)]
/// Converge if the step transform matrix (3x3) is almost equal to the identity matrix (component-wise within a given epsilon).
pub fn is_almost_identity_matrix<T, S, D>(
    epsilon: T,
) -> impl Fn(&[T], &[T], &Matrix<T, D, D, S>, &mut T, usize) -> bool
where
    T: Scalar + RealField + Copy,
    S: Storage<T, D, D>,
    D: Dim + DimName,
    DefaultAllocator: Allocator<D, D>,
{
    let eps_mat = OMatrix::<T, D, D>::from_element(epsilon);

    move |_: &[T], _: &[T], transform: &Matrix<T, D, D, S>, _: &mut T, _: usize| {
        let diff = (transform - OMatrix::<T, D, D>::identity()).abs();
        diff < eps_mat
    }
}
