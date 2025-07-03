use nalgebra::{Isometry3, RealField, Scalar};

/// Converge if the translation and the rotation of the alignee to the target transformation
/// are smaller than the given thresholds `translation_threshold` and `angle_threshold`.
///
/// Values to start experimenting with could be `translation_threshold = 0.001` and `angle_threshold = 0.01`.
pub fn is_small_isometry<T>(
    translation_threshold: T,
    angle_threshold: T,
) -> impl Fn(&Vec<T>, &Vec<T>, &Isometry3<T>, &mut T) -> bool
where
    T: Scalar + RealField + Copy,
{
    move |_: &Vec<T>, _: &Vec<T>, isometry: &Isometry3<T>, _: &mut T| {
        isometry.translation.vector.magnitude_squared() < translation_threshold.into()
            && isometry.rotation.angle() < angle_threshold.into()
    }
}
