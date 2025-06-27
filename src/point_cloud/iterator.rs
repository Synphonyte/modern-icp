use nalgebra::Scalar;

use super::{MaskedPointCloud, PointCloudPoint};

#[derive(Debug, Copy)]
pub struct PointCloudIterator<'a, T: Scalar + Copy, const D: usize> {
    pub(super) target: &'a MaskedPointCloud<'a, T, D>,
    pub(super) cur_index: usize,
}

impl<'a, T: Scalar + Copy, const D: usize> Clone for PointCloudIterator<'a, T, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: Scalar + Copy, const D: usize> Iterator for PointCloudIterator<'a, T, D> {
    type Item = &'a PointCloudPoint<T, D>;

    fn next(&mut self) -> Option<Self::Item> {
        let res = if self.cur_index >= self.target.masked_and_ordered_to_plain_index.len() {
            None
        } else {
            let index = self.target.masked_and_ordered_to_plain_index[self.cur_index];
            Some(&self.target.point_cloud[index])
        };

        self.cur_index += 1;
        res
    }
}

impl<'a, T: Scalar + Copy, const D: usize> ExactSizeIterator for PointCloudIterator<'a, T, D> {
    fn len(&self) -> usize {
        self.target.masked_and_ordered_to_plain_index.len()
    }
}
