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
mod types;

pub use align::*;
pub use common::*;
pub use plane::*;
pub use types::*;

#[cfg(feature = "rerun")]
lazy_static::lazy_static! {
    pub static ref RR: rerun::RecordingStream = rerun::RecordingStreamBuilder::new("shrink_to_fit_mesh").spawn().unwrap();
}

#[cfg(feature = "rerun")]
fn pt3_array<T>(pt: nalgebra::Point3<T>) -> [T; 3]
where
    T: Clone + PartialEq + nalgebra::Scalar,
{
    [pt.x, pt.y, pt.z]
}

#[cfg(feature = "rerun")]
fn vec3_array<T>(vec: nalgebra::Vector3<T>) -> [T; 3]
where
    T: Clone + PartialEq + nalgebra::Scalar,
{
    [vec.x, vec.y, vec.z]
}
