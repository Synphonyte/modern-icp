use nalgebra::{Point, Scalar};

use super::{PointCloud, PointCloudIterator, PointCloudPoint};

#[derive(Debug)]
pub struct MaskedPointCloud<'a, T, const D: usize>
where
    T: Scalar + Copy,
{
    pub point_cloud: &'a PointCloud<T, D>,
    pub masked_and_ordered_to_plain_index: Vec<usize>,
}

impl<'a, T: Scalar + Copy, const D: usize> Clone for MaskedPointCloud<'a, T, D> {
    fn clone(&self) -> Self {
        Self {
            point_cloud: self.point_cloud,
            masked_and_ordered_to_plain_index: self.masked_and_ordered_to_plain_index.clone(),
        }
    }
}

impl<'a, T: Scalar + Copy, const D: usize> MaskedPointCloud<'a, T, D> {
    pub fn new(point_cloud: &'a PointCloud<T, D>) -> Self {
        Self {
            point_cloud,
            masked_and_ordered_to_plain_index: (0..point_cloud.len()).collect(),
        }
    }

    pub fn with_mask(point_cloud: &'a PointCloud<T, D>, mask: &[bool]) -> Self {
        let mut iter = Self::new(point_cloud);
        iter.add_mask(mask);
        iter
    }

    pub fn add_mask(&mut self, mask: &[bool]) {
        self.masked_and_ordered_to_plain_index = mask
            .iter()
            .zip(self.masked_and_ordered_to_plain_index.iter())
            .filter_map(|(m, i)| if *m { Some(*i) } else { None })
            .collect();
    }

    pub fn len(&self) -> usize {
        self.masked_and_ordered_to_plain_index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.masked_and_ordered_to_plain_index.is_empty()
    }

    // pub fn iter_with_mask(&self) -> impl Iterator<Item = (&PointCloudPoint<T, D>, bool)> {
    //     self.point_cloud
    //         .iter()
    //         .enumerate()
    //         .map(|(i, p)| (p, self.masked_and_ordered_to_plain_index.contains(&i)))
    // }

    pub fn set_empty(&mut self) {
        self.masked_and_ordered_to_plain_index = vec![];
    }

    pub fn add_order(&mut self, indices: &[usize]) {
        self.masked_and_ordered_to_plain_index = indices
            .iter()
            .map(|i| self.masked_and_ordered_to_plain_index[*i])
            .collect();
    }

    pub fn extend(&mut self, other: &MaskedPointCloud<T, D>) {
        // TODO : check if targets are the same
        self.masked_and_ordered_to_plain_index
            .extend(&mut other.masked_and_ordered_to_plain_index.iter());
    }

    pub fn decompose(self) -> Vec<usize> {
        self.masked_and_ordered_to_plain_index
    }

    pub fn compose(
        point_cloud: &'a PointCloud<T, D>,
        masked_and_ordered_to_plain_index: Vec<usize>,
    ) -> Self {
        Self {
            point_cloud,
            masked_and_ordered_to_plain_index,
        }
    }

    pub fn sort_by_key<F, K>(&mut self, mut f: F)
    where
        F: FnMut(&PointCloudPoint<T, D>) -> K,
        K: Ord,
    {
        let mut points_and_indices = self
            .masked_and_ordered_to_plain_index
            .iter()
            .map(|i| (&self.point_cloud[*i], i))
            .collect::<Vec<_>>();

        points_and_indices.sort_by_key(|(p, _)| f(p));

        self.masked_and_ordered_to_plain_index =
            points_and_indices.iter().map(|(_, i)| **i).collect();
    }

    pub fn iter(&'a self) -> PointCloudIterator<'a, T, D> {
        PointCloudIterator {
            target: self,
            cur_index: 0,
        }
    }

    pub fn points_iter(&'a self) -> impl Iterator<Item = Point<T, D>> + Clone + use<'a, T, D> {
        self.iter().map(|p| p.pos)
    }
}

impl<'a, T: Scalar + Copy, const D: usize> From<&'a PointCloud<T, D>>
    for MaskedPointCloud<'a, T, D>
{
    fn from(point_cloud: &'a PointCloud<T, D>) -> Self {
        Self {
            point_cloud,
            masked_and_ordered_to_plain_index: Vec::new(),
        }
    }
}

#[cfg(feature = "rerun")]
impl<'a, T, const D: usize> MaskedPointCloud<'a, T, D>
where
    T: Scalar + Copy,
    f32: From<T>,
{
    pub fn log_rerun(&self, name: &str) {
        let mut included_points = vec![];
        let mut excluded_points = vec![];

        for (i, p) in self.point_cloud.iter().enumerate() {
            let arr = crate::pt3_array(p.pos);
            if self.masked_and_ordered_to_plain_index.contains(&i) {
                included_points.push(arr);
            } else {
                excluded_points.push(arr);
            }
        }

        crate::RR
            .log(
                format!("{name}/included"),
                &rerun::Points3D::new(included_points),
            )
            .unwrap();
        crate::RR
            .log(
                format!("{name}/excluded"),
                &rerun::Points3D::new(excluded_points),
            )
            .unwrap();
    }
}
