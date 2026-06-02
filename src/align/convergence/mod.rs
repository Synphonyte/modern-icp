mod is_small_isometry;
mod never;
mod same_squared_distance_error;

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
    ) -> bool;
}

impl<F, T, M> ConvergenceCriterion<T, M> for F
where
    F: FnMut(&[T], &[T], &M, &mut T) -> bool,
{
    fn is_converged(
        &mut self,
        target_distances: &[T],
        alignee_distances: &[T],
        current_transform: &M,
        error: &mut T,
    ) -> bool {
        self(
            target_distances,
            alignee_distances,
            current_transform,
            error,
        )
    }
}
