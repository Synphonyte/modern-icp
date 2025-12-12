use nalgebra::*;

use crate::{MaskedPointCloud, Plane};

/// Estimates the isometry between the alignee and the target using the Point-to-Plane-LLS algorithm.
///
/// See this [implementation of the algorithm from PointCloudLibrary](https://github.com/PointCloudLibrary/pcl/blob/3f19fc83cfa3850e13d5f833895871d6a92221e2/registration/include/pcl/registration/impl/transformation_estimation_point_to_plane_lls_weighted.hpp#L192)
/// See also this [paper from Low](https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf)
#[allow(non_snake_case)]
pub fn estimate_isometry<T>(
    alignee: &mut MaskedPointCloud<T, 3>,
    target: &mut MaskedPointCloud<T, 3>,
    _: usize,
) -> Option<Isometry3<T>>
where
    T: Scalar + RealField + Copy,
{
    let mut ATA = Matrix6::<T>::zeros();
    let mut ATb = Vector6::<T>::zeros();

    for (a, t) in alignee.iter().zip(target.iter()) {
        let pos_align = a.pos;

        let pos_target = t.pos;
        let norm_target = t.norm.unwrap() * (a.weight * t.weight);

        let ax = pos_align.x;
        let ay = pos_align.y;
        let az = pos_align.z;

        let tx = pos_target.x;
        let ty = pos_target.y;
        let tz = pos_target.z;

        let nx = norm_target.x;
        let ny = norm_target.y;
        let nz = norm_target.z;

        let a = nz * ay - ny * az;
        let b = nx * az - nz * ax;
        let c = ny * ax - nx * ay;

        ATA[(0, 0)] += a * a;
        ATA[(0, 1)] += a * b;
        ATA[(0, 2)] += a * c;
        ATA[(0, 3)] += a * nx;
        ATA[(0, 4)] += a * ny;
        ATA[(0, 5)] += a * nz;

        ATA[(1, 1)] += b * b;
        ATA[(1, 2)] += b * c;
        ATA[(1, 3)] += b * nx;
        ATA[(1, 4)] += b * ny;
        ATA[(1, 5)] += b * nz;

        ATA[(2, 2)] += c * c;
        ATA[(2, 3)] += c * nx;
        ATA[(2, 4)] += c * ny;
        ATA[(2, 5)] += c * nz;

        ATA[(3, 3)] += nx * nx;
        ATA[(3, 4)] += nx * ny;
        ATA[(3, 5)] += nx * nz;

        ATA[(4, 4)] += ny * ny;
        ATA[(4, 5)] += ny * nz;

        ATA[(5, 5)] += nz * nz;

        let d = nx * tx + ny * ty + nz * tz - nx * ax - ny * ay - nz * az;

        ATb[0] += a * d;
        ATb[1] += b * d;
        ATb[2] += c * d;
        ATb[3] += nx * d;
        ATb[4] += ny * d;
        ATb[5] += nz * d;
    }

    ATA[(1, 0)] = ATA[(0, 1)];

    ATA[(2, 0)] = ATA[(0, 2)];
    ATA[(2, 1)] = ATA[(1, 2)];

    ATA[(3, 0)] = ATA[(0, 3)];
    ATA[(3, 1)] = ATA[(1, 3)];
    ATA[(3, 2)] = ATA[(2, 3)];

    ATA[(4, 0)] = ATA[(0, 4)];
    ATA[(4, 1)] = ATA[(1, 4)];
    ATA[(4, 2)] = ATA[(2, 4)];
    ATA[(4, 3)] = ATA[(3, 4)];

    ATA[(5, 0)] = ATA[(0, 5)];
    ATA[(5, 1)] = ATA[(1, 5)];
    ATA[(5, 2)] = ATA[(2, 5)];
    ATA[(5, 3)] = ATA[(3, 5)];
    ATA[(5, 4)] = ATA[(4, 5)];

    ATA.try_inverse().map(|ATA_inv| {
        // Solve A*x = b
        let x = ATA_inv * ATb;

        // based on https://github.com/pglira/simpleICP/blob/236dfe918ab8e2af53e71d9963816e3adf8f0b76/python/simpleicp.py#L78
        let alpha = x[0];
        let beta = x[1];
        let gamma = x[2];

        Isometry3::from_parts(
            Translation3::new(x[3], x[4], x[5]),
            UnitQuaternion::from_matrix(&matrix![
            T::one(), -gamma, beta;
            gamma, T::one(), -alpha;
            -beta, alpha, T::one()]),
        )
    })
}

pub fn estimate_scale_point_to_plane<'a, T>(
    alignee: &mut MaskedPointCloud<'a, T, 3>,
    target: &mut MaskedPointCloud<'a, T, 3>,
) -> Matrix3<T>
where
    T: Scalar + RealField + Copy + From<f32>,
    f64: From<T>,
{
    let mut scale = Vector3::zeros();

    for (a, t) in alignee.iter().zip(target.iter()) {
        let a_mul_n = a.pos.coords.component_mul(&t.norm.unwrap());

        let x = a_mul_n.x;
        let y = a_mul_n.y;
        let z = a_mul_n.z;

        let c = Plane::from_normal_and_point(&t.norm.unwrap(), &t.pos).constant;

        scale += Vector3::new((c - y - z) / x, (c - x - z) / y, (c - y - x) / z);
    }

    Matrix3::from_diagonal(&scale)
}
