use nalgebra::{RealField, Scalar};

/// Never converges. Always returns false.
pub fn never<T: Scalar + RealField, M>(_: &Vec<T>, _: &Vec<T>, _: &M, _: &mut T) -> bool {
    false
}
