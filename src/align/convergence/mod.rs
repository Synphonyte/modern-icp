mod is_almost_identity;
mod is_small_isometry;
mod never;
mod same_squared_distance_error;

pub use is_almost_identity::*;
pub use is_small_isometry::*;
pub use never::*;
pub use same_squared_distance_error::*;

pub trait ConvergenceCriterion<T, M> {
    fn is_converged(
        &mut self,
        target_distances: &[T],
        alignee_distances: &[T],
        current_transform: &M,
        error: &mut T,
        step: usize,
    ) -> bool;
}

impl<F, T, M> ConvergenceCriterion<T, M> for F
where
    F: FnMut(&[T], &[T], &M, &mut T, usize) -> bool,
{
    fn is_converged(
        &mut self,
        target_distances: &[T],
        alignee_distances: &[T],
        current_transform: &M,
        error: &mut T,
        step: usize,
    ) -> bool {
        self(
            target_distances,
            alignee_distances,
            current_transform,
            error,
            step,
        )
    }
}
