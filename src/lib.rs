//! A modern modular pure Rust implementation of the Iterative Closest Point algorithm.
//!
//! ## Example
//!
//! ```
//! # use modern_icp::{Icp, PointCloud, PointCloudPoint};
//! # use modern_icp::correspondence::{BidirectionalDistance, NearestNeighbor};
//! # use modern_icp::transform_estimation::point_to_plane_lls;
//! # use modern_icp::convergence::{never, same_squared_distance_error};
//! # use modern_icp::reject_outliers::reject_n_sigma_dist;
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
//! #     alignee_cloud.push(PointCloudPoint::from_pos_norm(pos, norm));
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
//! #     target_cloud.push(PointCloudPoint::from_pos_norm(pos, norm));
//! # }
//! #
//! // Basic usage
//! let (alignee_transform, error_sum) = Icp::new()
//!     .correspondence_estimator(NearestNeighbor::new(&target_cloud))
//!     .estimate_step_transform(point_to_plane_lls::estimate_isometry)
//!     .is_converged(same_squared_distance_error(0.1))
//!     .estimate_transform(alignee_cloud, &target_cloud);
//!
//! # let alignee_cloud = PointCloud::<f32, 3>::new();
//! # let target_cloud = PointCloud::<f32, 3>::new();
//! #
//! // With outliers rejection and point filtering
//! let (alignee_transform, error_sum) = Icp::new()
//!     .correspondence_estimator(BidirectionalDistance::new(&target_cloud))
//!     .estimate_step_transform(point_to_plane_lls::estimate_isometry)
//!     .is_converged(never) // run for 20 iterations without convergence check
//!     .max_iterations(20) // stop after 20 iterations
//!     .filter_points(|pt: &PointCloudPoint<f32, 3>| pt.pos.z > 0.0) // only use points above the xy-plane
//!     .reject_outliers(reject_n_sigma_dist(3.0))
//!     .estimate_transform(alignee_cloud, &target_cloud);
//! ```
//!
//! ## Integrations
//!
//! ### Serde
//!
//! A serde integration is provided so you can serialize and deserialize `PointCloud` and `PointCloudPoint` using the `serde` crate.
//! Enable the `serde` feature to use it.
//!
//! ### Modelz
//!
//! An integrations with the modelz crate is provided so you can use `Model3D` with the `estimate_transform` function.
//! This allows for easy loading of 3D models from disk and using them with the ICP algorithm. Enable the `modelz` feature to use it.
//!
//! ```
//! # use modern_icp::PointCloudPoint;
//! # use modern_icp::Icp;
//! # use modern_icp::correspondence::NearestNeighbor;
//! # use modern_icp::transform_estimation::point_to_plane_lls;
//! # use modern_icp::convergence::same_squared_distance_error;
//! # use modern_icp::reject_outliers::reject_n_sigma_dist;
//! # use modern_icp::filter_points::accept_all;
//! # use crate::modern_icp::correspondence::CorrespondenceEstimator;
//! use modelz::Model3D;
//!
//! let Ok(alignee) = Model3D::load("alignee.gltf") else { return; };
//! let Ok(target) = Model3D::load("target.stl") else { return; };
//!
//! let (alignee_transform, error_sum) = Icp::new()
//!     .correspondence_estimator(NearestNeighbor::new(&target))
//!     .estimate_step_transform(point_to_plane_lls::estimate_isometry)
//!     .is_converged(same_squared_distance_error(0.1))
//!     .estimate_transform(alignee, &target);
//! ```
//!
//! ### Rerun
//!
//! An integration with the rerun crate is provided so you can visualize the ICP process.
//! With [the `rerun` viewer installed](https://rerun.io/docs/getting-started/install-rerun/viewer) simply enable the `rerun` Cargo feature.
//! When you run the ICP algorithm you'll see the alignment process visualized in real-time.
mod align;
mod common;
mod integrations;
mod plane;
mod point_cloud;

pub use align::*;
pub use common::*;
#[allow(unused_imports)]
pub use integrations::*;
pub use plane::*;
pub use point_cloud::*;

#[cfg(feature = "rerun")]
lazy_static::lazy_static! {
    pub static ref RR: rerun::RecordingStream = rerun::RecordingStreamBuilder::new("modern-icp").spawn().unwrap();
}

#[cfg(feature = "rerun")]
fn pt3_array<T, const D: usize>(pt: nalgebra::Point<T, D>) -> [f32; 3]
where
    T: Clone + PartialEq + nalgebra::Scalar,
    f32: From<T>,
{
    [
        pt[0].clone().into(),
        pt[1].clone().into(),
        pt[2].clone().into(),
    ]
}

#[cfg(feature = "rerun")]
fn vec3_array<T, const D: usize>(vec: nalgebra::OVector<T, nalgebra::Const<D>>) -> [f32; 3]
where
    T: Clone + PartialEq + nalgebra::Scalar,
    f32: From<T>,
{
    [
        vec[0].clone().into(),
        vec[1].clone().into(),
        vec[2].clone().into(),
    ]
}

#[cfg(feature = "rerun")]
pub fn unit_quat_array<T>(quat: nalgebra::UnitQuaternion<T>) -> [f32; 4]
where
    T: Copy + PartialEq + nalgebra::SimdValue + nalgebra::Scalar + nalgebra::RealField,
    f32: From<T>,
{
    let quat = quat.quaternion();
    [quat.i.into(), quat.j.into(), quat.k.into(), quat.w.into()]
}

#[cfg(feature = "rerun")]
pub fn rr_log_cloud<T, const D: usize>(name: &str, pt_cloud: &PointCloud<T, D>)
where
    T: Copy + Clone + PartialEq + nalgebra::Scalar,
    f32: From<T>,
{
    if D == 3 {
        crate::RR
            .log(
                format!("{}/points", name),
                &rerun::Points3D::new(pt_cloud.iter().map(|pt| pt3_array(pt.pos))),
            )
            .unwrap();

        if pt_cloud[0].norm.is_some() {
            crate::RR
                .log(
                    format!("{}/normals", name),
                    &rerun::Arrows3D::from_vectors(
                        pt_cloud.iter().map(|pt| vec3_array(pt.norm.unwrap())),
                    )
                    .with_origins(pt_cloud.iter().map(|pt| pt3_array(pt.pos))),
                )
                .unwrap();
        }
    }
}
