# Modern ICP

[![Crates.io](https://img.shields.io/crates/v/modern-icp.svg)](https://crates.io/crates/modern-icp)
[![Docs](https://docs.rs/modern-icp/badge.svg)](https://docs.rs/modern-icp/)
[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/synphonyte/modern-icp#license)
[![Build Status](https://github.com/synphonyte/modern-icp/actions/workflows/cd.yml/badge.svg)](https://github.com/synphonyte/modern-icp/actions/workflows/cd.yml)

<!-- cargo-rdme start -->

A modern modular pure Rust implementation of the Iterative Closest Point algorithm.

### Example

```rust
let (alignee_transform, error_sum) = estimate_transform(
    alignee_cloud,
    &target_cloud,
    20, // max iterations
    BidirectionalDistance::new(&target_cloud),
    accept_all,
    reject_3_sigma_dist,
    point_to_plane_lls::estimate_isometry,
    same_squared_distance_error(1.0),
);
```

### Integrations

An integrations with the modelz crate is provided so you can use `Model3D` with the `estimate_transform` function.

```rust
use modelz::Model3D;

if let (Ok(alignee), Ok(target)) = (Model3D::load("alignee.gltf"), Model3D::load("target.stl")) {
    let (transform, error_sum) = estimate_transform(
        alignee,
        &target,
        20, // max iterations
        BidirectionalDistance::new(&target),
        accept_all,
        reject_3_sigma_dist,
        point_to_plane_lls::estimate_isometry,
        same_squared_distance_error(1.0),
    );
}
```

<!-- cargo-rdme end -->
