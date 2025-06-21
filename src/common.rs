use crate::PointCloud;
use nalgebra::*;
use num_traits::AsPrimitive;
use std::ops::{Div, Mul};

/// Computes the centroid (geometric center) of the point cloud.
pub fn compute_centroid<T, const D: usize>(
    points: impl Iterator<Item = Point<T, D>>,
) -> SVector<T, D>
where
    T: RealField + Copy,
    SVector<T, D>: Div<T, Output = SVector<T, D>>,
{
    let mut centroid = SVector::zeros();

    let mut count = T::zero();
    for pt in points {
        centroid += pt.coords;
        count += T::one();
    }

    if count == T::zero() {
        return centroid;
    }

    centroid / count
}

/// Demeans the point cloud by subtracting the centroid from every point and returns the demeaned
/// point cloud as a matrix.
pub fn demean_into_matrix<T, const D: usize>(
    points: impl Iterator<Item = Point<T, D>>,
    centroid: &SVector<T, D>,
) -> OMatrix<T, Const<D>, Dyn>
where
    T: Scalar + RealField + Copy,
{
    let mut demeaned_cloud = vec![];

    for pt in points {
        demeaned_cloud.push(pt.coords - centroid);
    }

    Matrix::from_columns(demeaned_cloud.as_slice())
}

/// Transforms every point in the point cloud using the given transform.
pub fn transform_point_cloud<T, M, const D: usize>(cloud: &mut PointCloud<T, D>, transform: M)
where
    T: Scalar + RealField + Copy,
    M: Mul<Point<T, D>, Output = Point<T, D>> + Mul<SVector<T, D>, Output = SVector<T, D>> + Copy,
{
    for point in cloud.iter_mut() {
        point.pos = transform * point.pos;

        point.norm = point.norm.map(|norm| transform * norm);
    }
}

/// See the [paper from Dong et al.](https://doi.org/10.1049/iet-cvi.2016.0058)
pub fn golden_section_search<T>(f: &dyn Fn(T) -> T, a: T, b: T, tol: Option<T>) -> T
where
    T: Scalar + RealField + Copy,
    f32: AsPrimitive<T>,
{
    let gr = (5.0.as_().sqrt() + T::one()) * 0.5.as_();

    let mut new_a = a;
    let mut new_b = b;

    let mut c = new_b - (new_b - new_a) / gr;
    let mut d = new_a + (new_b - new_a) / gr;

    while (new_b - new_a).abs() > tol.unwrap_or(1e-5.as_()) {
        if f(c) < f(d) {
            new_b = d;
        } else {
            new_a = c;
        }

        // We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = new_b - (new_b - new_a) / gr; // new_b - (gr - 1)(new_a - new_b)
        d = new_a + (new_b - new_a) / gr; // new_a - (gr - 1)(new_a - new_b)
    }

    new_a // (new_b + new_a) * T::from(0.5);
}

pub fn sum_squared_distances<T>(distances: &[T], x: Option<T>) -> T
where
    T: Scalar + RealField + Copy,
    usize: AsPrimitive<T>,
{
    let d_length = distances.len().as_() * x.unwrap_or(T::one());

    let mut sum = T::zero();
    for (i, d) in distances.iter().rev().enumerate() {
        if i.as_() > d_length {
            break;
        }
        sum += *d
    }

    sum
}
