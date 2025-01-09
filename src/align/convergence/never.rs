use nalgebra::{RealField, Scalar};

/// Never converges. Always returns false.
pub fn never<'a, T: Scalar + RealField, M>(_: &Vec<T>, _: &Vec<T>, _: &M, _: &mut T) -> bool {
    false
}
