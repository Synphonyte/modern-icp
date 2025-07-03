# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-07-02

- Renamed `is_small_squared_distance_error` to `same_squared_distance_error`
- `estimate_transform` now returns the final alignment error together with the transform
- Principal Component Analysis now returns the Eigenvalues together with principal axes
- ICP now accepts a filter to filter out points to be considered for alignment
  - Two predefined filters are provided: `accept_all` and `above_planes`

## [0.2.0] - 2025-06-26

- Some small cleanups
- point_to_plane_lls::estimate_isometry now returns an Isometry3 instead of a IsometryMatrix3
- rerun visualization works

## [0.1.0] - 2025-06-21

- Implementation of various variations of ICP which can be combined modularly
