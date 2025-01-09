use nalgebra::*;

// pub fn estimate_affine<'a, T>(alignee: &PointCloud<T, 3>, target: &PointCloud<T, 3>) -> Affine3<T>
//     where T: Scalar + RealField + From<f32> + Copy,
// {
//     // see paper https://doi.org/10.1109/JAS.2019.1911579 end of chapter III. B.
//     // possibly this has some implementation bugs but it doesn't seem to be numerically stable
//
//     let sigma = T::from(1.0);
//     let quotient = T::one() / (T::from(2.0) * sigma * sigma);
//
//     let to_homogeneous_vec4 =
//         |p: &&PointCloudPoint<T, 3>| p.pos.to_homogeneous();
//
//     // alignee
//     let PT: Matrix4xX<T> = Matrix4xX::from_columns(
//         alignee.into_iter()
//             .map(to_homogeneous_vec4)
//             .collect::<Vec<_>>()
//             .as_slice()
//     );
//     let P = PT.transpose();
//
//     let MT: Matrix4xX<T> = Matrix4xX::from_columns(
//         target.into_iter()
//             .map(to_homogeneous_vec4)
//             .collect::<Vec<_>>()
//             .as_slice(),
//     );
//     let M = MT.transpose();
//
//     let D = DMatrix::from_diagonal(
//         &DVector::from_iterator(
//             PT.column_iter().len(),
//             (M - &P)
//                 .row_iter()
//                 .map(
//                     |row| (-row.norm_squared() * quotient).exp()
//                 ),
//         )
//     );
//
//     let DP = D * P;
//
//     Affine3::from_matrix_unchecked(MT * &DP * (PT * DP).try_inverse().unwrap())
// }
