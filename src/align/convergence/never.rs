use nalgebra::{RealField, Scalar};

/// Never converges. Always returns false.
pub fn never<T: Scalar + RealField, M>(_: &[T], _: &[T], _: &M, _: &mut T, _: usize) -> bool {
    false
}
