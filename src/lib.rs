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
