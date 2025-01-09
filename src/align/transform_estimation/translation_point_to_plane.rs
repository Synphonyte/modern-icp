use crate::{Plane, PointCloudIterator};
use nalgebra::*;

/// Estimates exclusively the translation between the alignee and the target using the Point-to-Plane algorithm.
pub fn estimate_translation<T, const D: usize>(
    alignee: &mut PointCloudIterator<T, D>,
    target: &mut PointCloudIterator<T, D>,
    _: usize,
) -> Isometry<T, Rotation<T, D>, D>
where
    T: Scalar + RealField + Copy + From<f32>,
{
    let translation_vec = alignee
        .zip(target.map(|t| {
            Plane::from_normal_and_point(
                &t.norm.expect("Target point cloud must have normals"),
                &t.pos,
            )
        }))
        .map(|(a, t)| t.normal * t.distance_to_point(&a.pos))
        .sum::<SVector<T, D>>()
        / -T::from(alignee.len() as f32);

    Isometry::from_parts(Translation::from(translation_vec), Rotation::identity())
}
