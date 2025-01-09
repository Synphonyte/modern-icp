use crate::{compute_centroid, demean_into_matrix, Plane3, PointCloudIterator};
use nalgebra::{Point3, SVD};
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
/// A function that takes a point cloud iterator and returns a mask indicating which points to keep.
/// Also, this returned function applies the mask to the point cloud iterators.
pub fn reject_outliers_plane_dist(
    std_dev_threshold: f32,
    max_iterations: usize,
) -> impl Fn(&mut PointCloudIterator<f32, 3>, &mut PointCloudIterator<f32, 3>, &Vec<f32>) -> Vec<bool>
{
    move |mut alignee: &mut PointCloudIterator<f32, 3>,
          target: &mut PointCloudIterator<f32, 3>,
          _: &Vec<f32>|
          -> Vec<bool> {
        let len = alignee.len();

        let mut mask = vec![true; len];

        for _ in 0..max_iterations {
            let alignee_centroid = compute_centroid(&mut alignee);

            alignee.reset_iter();

            let demeaned_alignee = demean_into_matrix(&mut alignee, &alignee_centroid);

            let SVD {
                u,
                v_t: _,
                singular_values: _,
            } = demeaned_alignee
                .try_svd(true, false, f32::EPSILON, 10)
                .expect("SVD unsuccessful");

            let uw = u.unwrap();

            let normal = uw.column(2);

            let plane =
                Plane3::from_normal_and_point(&normal.normalize(), &Point3::from(alignee_centroid));

            alignee.reset_iter();

            let distances: Vec<f32> = alignee
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
