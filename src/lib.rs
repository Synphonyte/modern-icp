//! A modern modular pure Rust implementation of the Iterative Closest Point algorithm.
//!
//! ## Example
//!
//! ```
//! # use modern_icp::PointCloudPoint;
//! # use modern_icp::icp::estimate_transform;
//! # use modern_icp::correspondence::BidirectionalDistance;
//! # use modern_icp::transform_estimation::point_to_plane_lls;
//! # use modern_icp::convergence::same_squared_distance_error;
//! # use modern_icp::reject_outliers::reject_3_sigma_dist;
//! # use modern_icp::filter_points::accept_all;
//! # use crate::modern_icp::correspondence::CorrespondenceEstimator;
//! # use nalgebra::{Point3, Vector3};
//! #
//! # // Generate random point clouds
//! # let mut alignee_cloud = Vec::with_capacity(100);
//! # let mut target_cloud = Vec::with_capacity(100);
//! #
//! # // Generate 100 random points for alignee cloud
//! # for _ in 0..100 {
//! #
//! #     let pos = Point3::new(
//! #         rand::random::<f32>() * 10.0,
//! #         rand::random::<f32>() * 10.0,
//! #         rand::random::<f32>() * 10.0,
//! #     );
//! #     let norm = Vector3::new(
//! #         rand::random::<f32>() * 2.0 - 1.0,
//! #         rand::random::<f32>() * 2.0 - 1.0,
//! #         rand::random::<f32>() * 2.0 - 1.0,
//! #     ).normalize();
//! #     alignee_cloud.push(PointCloudPoint { pos, norm: Some(norm) });
//! # }
//! #
//! # // Generate 100 random points for target cloud
//! # for _ in 0..100 {
//! #     let pos = Point3::new(
//! #         rand::random::<f32>() * 10.0,
//! #         rand::random::<f32>() * 10.0,
//! #         rand::random::<f32>() * 10.0,
//! #     );
//! #     let norm = Vector3::new(
//! #         rand::random::<f32>() * 2.0 - 1.0,
//! #         rand::random::<f32>() * 2.0 - 1.0,
//! #         rand::random::<f32>() * 2.0 - 1.0,
//! #     ).normalize();
//! #     target_cloud.push(PointCloudPoint { pos, norm: Some(norm) });
//! # }
//! #
//! let (alignee_transform, error_sum) = estimate_transform(
//!     &alignee_cloud,
//!     &target_cloud,
//!     20, // max iterations
//!     BidirectionalDistance::new(&target_cloud),
//!     accept_all,
//!     reject_3_sigma_dist,
//!     point_to_plane_lls::estimate_isometry,
//!     same_squared_distance_error(1.0),
//! );
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
