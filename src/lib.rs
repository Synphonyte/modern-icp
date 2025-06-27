//! A modern modular pure Rust implementation of the Iterative Closest Point algorithm.
//!
//! ## Example
//!
//! ```
//! use modern_icp::{};
//!
//!
//! ```
mod align;
mod common;
mod plane;
mod point_cloud;

pub use align::*;
pub use common::*;
pub use plane::*;
pub use point_cloud::*;

#[cfg(feature = "rerun")]
lazy_static::lazy_static! {
    pub static ref RR: rerun::RecordingStream = rerun::RecordingStreamBuilder::new("shrink_to_fit_mesh").spawn().unwrap();
}

#[cfg(feature = "rerun")]
fn pt3_array<T>(pt: nalgebra::Point3<T>) -> [f32; 3]
where
    T: Clone + PartialEq + nalgebra::Scalar,
    f32: From<T>,
{
    [
        pt.x.clone().into(),
        pt.y.clone().into(),
        pt.z.clone().into(),
    ]
}

#[cfg(feature = "rerun")]
fn vec3_array<T>(vec: nalgebra::Vector3<T>) -> [f32; 3]
where
    T: Clone + PartialEq + nalgebra::Scalar,
    f32: From<T>,
{
    [
        vec.x.clone().into(),
        vec.y.clone().into(),
        vec.z.clone().into(),
    ]
}

#[cfg(feature = "rerun")]
pub fn rr_log_cloud<T>(name: &str, pt_cloud: &PointCloud<T, 3>)
where
    T: Copy + Clone + PartialEq + nalgebra::Scalar,
    f32: From<T>,
{
    crate::RR
        .log(
            name,
            &rerun::Points3D::new(
                pt_cloud
                    .iter()
                    .map(|pt| pt3_array(pt.pos))
                    .collect::<Vec<_>>(),
            ),
        )
        .unwrap();
}
