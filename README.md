# Modern ICP

[![Crates.io](https://img.shields.io/crates/v/modern-icp.svg)](https://crates.io/crates/modern-icp)
[![Docs](https://docs.rs/modern-icp/badge.svg)](https://docs.rs/modern-icp/)
[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/synphonyte/modern-icp#license)
[![Build Status](https://github.com/synphonyte/modern-icp/actions/workflows/cd.yml/badge.svg)](https://github.com/synphonyte/modern-icp/actions/workflows/cd.yml)

<!-- cargo-rdme start -->

A modern modular pure Rust implementation of the Iterative Closest Point algorithm.

### Example

```rust
// Basic usage
let (alignee_transform, error_sum) = Icp::new()
    .correspondence_estimator(NearestNeighbor::new(&target_cloud))
    .estimate_step_transform(point_to_plane_lls::estimate_isometry)
    .is_converged(same_squared_distance_error(0.1))
    .estimate_transform(alignee_cloud, &target_cloud);

// With outliers rejection and point filtering
let (alignee_transform, error_sum) = Icp::new()
    .correspondence_estimator(BidirectionalDistance::new(&target_cloud))
    .estimate_step_transform(point_to_plane_lls::estimate_isometry)
    .is_converged(never) // run for 20 iterations without convergence check
    .max_iterations(20) // stop after 20 iterations
    .filter_points(|pt: &PointCloudPoint<f32, 3>| pt.pos.z > 0.0) // only use points above the xy-plane
    .reject_outliers(reject_n_sigma_dist(3.0))
    .estimate_transform(alignee_cloud, &target_cloud);
```

### Integrations

#### Serde

A serde integration is provided so you can serialize and deserialize `PointCloud` and `PointCloudPoint` using the `serde` crate.
Enable the `serde` feature to use it.

#### Modelz

An integrations with the modelz crate is provided so you can use `Model3D` with the `estimate_transform` function.
This allows for easy loading of 3D models from disk and using them with the ICP algorithm. Enable the `modelz` feature to use it.

```rust
use modelz::Model3D;

let Ok(alignee) = Model3D::load("alignee.gltf") else { return; };
let Ok(target) = Model3D::load("target.stl") else { return; };

let (alignee_transform, error_sum) = Icp::new()
    .correspondence_estimator(NearestNeighbor::new(&target))
    .estimate_step_transform(point_to_plane_lls::estimate_isometry)
    .is_converged(same_squared_distance_error(0.1))
    .estimate_transform(alignee, &target);
```

#### Rerun

An integration with the rerun crate is provided so you can visualize the ICP process.
With [the `rerun` viewer installed](https://rerun.io/docs/getting-started/install-rerun/viewer) simply enable the `rerun` Cargo feature.
When you run the ICP algorithm you'll see the alignment process visualized in real-time.

<!-- cargo-rdme end -->
