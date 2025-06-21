use kdtree::KdTree;
use nalgebra::{Point, Point3, RealField, SVector, Scalar};
use num_traits::{Float, One, Zero};

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[derive(Debug)]
pub struct MaskedPointCloud<'a, T, const D: usize>
where
    T: Scalar + Copy,
{
    pub point_cloud: &'a PointCloud<T, D>,
    pub masked_and_ordered_to_plain_index: Vec<usize>,
}

#[derive(Debug, Copy)]
pub struct PointCloudIterator<'a, T: Scalar + Copy, const D: usize> {
    target: &'a MaskedPointCloud<'a, T, D>,
    cur_index: usize,
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
    let mut kd_tree = KdTree::new(D);

    for (i, p) in point_cloud.iter().enumerate() {
        kd_tree.add(p.pos.coords.as_slice(), i).unwrap();
    }

    kd_tree
}
