use nalgebra::{Affine3, Matrix3};

fn is_almost_identity_affine(
    _: &[f32],
    _: &[f32],
    transform: &Affine3<f32>,
    _: &mut f32,
    _: usize,
) -> bool {
}

/// Converge if the step transform matrix is almost the identity matrix.
///
/// Compares each matrix column component-wise with the x/y/z basis vector respectively.
fn is_almost_identity_matrix(epsilon: f32) -> impl Fn(
    &[f32],
    &[f32],
     &Matrix3<f32>,
   &mut f32,
  usize,
) -> bool {
    move |
    _: &[f32],
    _: &[f32],
    transform: &Matrix3<f32>,
    _: &mut f32,
    _: usize,| {
    let x_col = transform.fixed_view::<3, 1>(0, 0);
    let y_col = transform.fixed_view::<3, 1>(0, 1);
    let z_col = transform.fixed_view::<3, 1>(0, 2);

    (x_col - Vec3::x()).abs() <= EPS3
    && (y_col - Vec3::y()) <= EPS3
    && (z_col - Vec3::z()) <= EPS3
    }
}
