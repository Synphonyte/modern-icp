use crate::{MaskedPointCloud, Plane};
use statistical::standard_deviation;

/// Reject outliers based on plane distance.
///
/// Fits a plane iteratively to the point cloud using SVD and then rejects points that are farther
/// than `std_dev_threshold` away from the plane.
///
/// ## Parameters
///
/// - `Std_dev_threshold`: The threshold for the standard deviation of the distances from the plane.
/// - `Max_iterations`: The maximum number of iterations of matching the plane to the cloud to perform.
///
/// ## Returns
///
/// A function that takes a masked point cloud and returns a mask indicating which points to keep.
/// Also, this returned function applies the mask to the masked point cloud.
pub fn reject_outliers_plane_dist(
    std_dev_threshold: f32,
    max_iterations: usize,
) -> impl Fn(&mut MaskedPointCloud<f32, 3>, &mut MaskedPointCloud<f32, 3>, &Vec<f32>) -> Vec<bool> {
    move |alignee: &mut MaskedPointCloud<f32, 3>,
          target: &mut MaskedPointCloud<f32, 3>,
          _: &Vec<f32>|
          -> Vec<bool> {
        let len = alignee.len();

        let mut mask = vec![true; len];

        for _ in 0..max_iterations {
            let plane = Plane::fit_to_points(alignee.points_iter());

            let distances: Vec<f32> = alignee
                .iter()
                .map(|p| plane.distance_to_point(&p.pos).abs())
                .collect();

            if standard_deviation(distances.as_slice(), None) < std_dev_threshold {
                break;
            }

            let mut max_dist = 0.0;
            let mut max_idx = 0;

            for (i, d) in distances.into_iter().enumerate() {
                if d > max_dist {
                    max_dist = d;
                    max_idx = i;
                }
            }

            let mut local_mask = vec![true; alignee.len()];
            local_mask[max_idx] = false;

            let mut i = 0;
            for m in mask.iter_mut() {
                if *m {
                    if i == max_idx {
                        *m = false;
                        break;
                    }
                    i += 1;
                }
            }

            alignee.add_mask(&local_mask);
            target.add_mask(&local_mask);
        }

        mask
    }
}
