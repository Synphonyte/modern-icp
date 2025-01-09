use nalgebra::*;

pub struct Plane<T: Scalar + RealField + Copy, const D: usize> {
    pub normal: SVector<T, D>,
    pub constant: T,
}

impl<T: Scalar + RealField + From<f32> + Copy, const D: usize> Plane<T, D> {
    pub fn new(normal: &SVector<T, D>, constant: T) -> Plane<T, D> {
        Plane {
            normal: *normal,
            constant,
        }
    }

    pub fn from_normal_and_point(normal: &SVector<T, D>, coplanar_point: &Point<T, D>) -> Plane<T, D> {
        Plane {
            normal: *normal,
            constant: normal.dot(&coplanar_point.coords),
        }
    }

    pub fn distance_to_point(&self, point: &Point<T, D>) -> T {
        self.normal.dot(&point.coords) - self.constant
    }
}

pub type Plane3<T> = Plane<T, 3>;
