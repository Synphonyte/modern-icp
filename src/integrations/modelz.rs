use modelz::Model3D;
use nalgebra::{Point3, Vector3};

use crate::{PointCloud, PointCloudPoint, ToPointCloud};

impl ToPointCloud<f32, 3> for Model3D {
    fn to_point_cloud(&self) -> PointCloud<f32, 3> {
        let vert_count = self.meshes.iter().map(|m| m.vertices.len()).sum();
        let mut point_cloud = PointCloud::with_capacity(vert_count);

        for mesh in &self.meshes {
            for vertex in &mesh.vertices {
                point_cloud.push(PointCloudPoint {
                    pos: Point3::new(vertex.position[0], vertex.position[1], vertex.position[2]),
                    norm: vertex.normal.map(|n| Vector3::new(n[0], n[1], n[2])),
                    weight: 1.0,
                });
            }
        }

        point_cloud
    }
}
