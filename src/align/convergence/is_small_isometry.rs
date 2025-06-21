use nalgebra::{IsometryMatrix3, RealField, Scalar};

const MIN_TRANSLATION_THRESHOLD: f32 = 0.001;
const MIN_ANGLE_THRESHOLD: f32 = 0.01;

/// Converge if the translation and the rotation of the alignee to the target transformation
/// are too small.
pub fn is_small_isometry<T>(
    _: &Vec<T>,
    _: &Vec<T>,
    isometry: &IsometryMatrix3<T>,
    _: &mut T,
) -> bool
where
    T: Scalar + RealField + From<f32>,
{
    isometry.translation.vector.magnitude_squared() < MIN_TRANSLATION_THRESHOLD.into()
        && isometry.rotation.angle() < MIN_ANGLE_THRESHOLD.into()
}
