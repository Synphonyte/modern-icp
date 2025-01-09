use kdtree::KdTree;
use nalgebra::{Point, Point3, RealField, SVector, Scalar};
use num_traits::{Float, One, Zero};

#[derive(Clone)]
pub struct PointCloudPoint<T: Scalar + Copy, const D: usize> {
    pub pos: Point<T, D>,
    pub norm: Option<SVector<T, D>>,
}

impl<T: Scalar + Copy, const D: usize> From<Point<T, D>> for PointCloudPoint<T, D> {
    fn from(p: Point<T, D>) -> Self {
        PointCloudPoint { pos: p, norm: None }
    }
}

impl<T: Scalar + Copy, const D: usize> From<SVector<T, D>> for PointCloudPoint<T, D> {
    fn from(p: SVector<T, D>) -> Self {
        PointCloudPoint {
            pos: Point::from(p),
            norm: None,
        }
    }
}

pub type PointCloud<T, const D: usize> = Vec<PointCloudPoint<T, D>>;

pub struct PointCloudIterator<'a, T: Scalar + Copy, const D: usize> {
    target: &'a PointCloud<T, D>,
    masked_and_ordered_to_plain_index: Vec<usize>,
    cur_index: usize,
}

impl<'a, T: Scalar + Copy, const D: usize> PointCloudIterator<'a, T, D> {
    pub fn new(target: &'a PointCloud<T, D>) -> Self {
        PointCloudIterator {
            target,
            masked_and_ordered_to_plain_index: (0..target.len()).collect(),
            cur_index: 0,
        }
    }

    pub fn with_mask(target: &'a PointCloud<T, D>, mask: &[bool]) -> Self {
        let mut iter = Self::new(&target);
        iter.add_mask(&mask);
        iter
    }

    pub fn add_mask(&mut self, mask: &[bool]) {
        self.masked_and_ordered_to_plain_index = mask
            .iter()
            .zip(self.masked_and_ordered_to_plain_index.iter())
            .filter_map(|(m, i)| if *m { Some(*i) } else { None })
            .collect();
    }

    pub fn set_empty(&mut self) {
        self.masked_and_ordered_to_plain_index = vec![];
    }

    pub fn add_order(&mut self, indices: &[usize]) {
        self.masked_and_ordered_to_plain_index = indices
            .into_iter()
            .map(|i| self.masked_and_ordered_to_plain_index[*i])
            .collect();
    }

    #[inline]
    pub fn reset_iter(&mut self) {
        self.cur_index = 0;
    }

    pub fn extend(&mut self, other_iter: &PointCloudIterator<'a, T, D>) {
        // TODO : check if targets are the same
        self.masked_and_ordered_to_plain_index
            .extend(&mut other_iter.masked_and_ordered_to_plain_index.iter());
    }

    pub fn decompose(self) -> Vec<usize> {
        self.masked_and_ordered_to_plain_index
    }

    pub fn compose(
        target: &'a PointCloud<T, D>,
        masked_and_ordered_to_plain_index: Vec<usize>,
    ) -> Self {
        PointCloudIterator {
            target,
            masked_and_ordered_to_plain_index,
            cur_index: 0,
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
            .map(|i| (&self.target[*i], i))
            .collect::<Vec<_>>();

        points_and_indices.sort_by_key(|(p, _)| f(p));

        self.masked_and_ordered_to_plain_index =
            points_and_indices.iter().map(|(_, i)| **i).collect();
    }
}

impl<'a, T: Scalar + Copy, const D: usize> Iterator for PointCloudIterator<'a, T, D> {
    type Item = &'a PointCloudPoint<T, D>;

    fn next(&mut self) -> Option<Self::Item> {
        let res = if self.cur_index >= self.masked_and_ordered_to_plain_index.len() {
            None
        } else {
            let index = self.masked_and_ordered_to_plain_index[self.cur_index];
            Some(&self.target[index])
        };

        self.cur_index += 1;
        res
    }
}

impl<'a, T: Scalar + Copy, const D: usize> ExactSizeIterator for PointCloudIterator<'a, T, D> {
    fn len(&self) -> usize {
        self.masked_and_ordered_to_plain_index.len()
    }
}

pub fn point_cloud_from_position_slice<T: Scalar + Copy>(slice: &[T]) -> PointCloud<T, 3> {
    let mut cloud = PointCloud::with_capacity(slice.len() / 3);

    for i in (0..slice.len()).step_by(3) {
        cloud.push(PointCloudPoint {
            pos: Point3::new(slice[i], slice[i + 1], slice[i + 2]),
            norm: None,
        })
    }

    cloud
}

pub fn kd_tree_of_point_cloud<T, const D: usize>(
    point_cloud: &PointCloud<T, D>,
) -> KdTree<T, usize, &[T]>
where
    T: Scalar + RealField + Float + One + Zero,
{
    let mut kd_tree = KdTree::new(D as usize);

    for (i, p) in point_cloud.iter().enumerate() {
        kd_tree.add(p.pos.coords.as_slice(), i).unwrap();
    }

    kd_tree
}
