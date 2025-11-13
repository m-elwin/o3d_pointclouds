import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2
import numpy as np
import open3d as o3d

class TableFind(Node):
    def __init__(self):
        """
        Create a ROS Node that extracts a table (or other flat surface) from a point cloud
        using Open3D instead of PCL.
        """
        super().__init__("table_find")

        # ROS interfaces
        self._sub = self.create_subscription(PointCloud2, "pcl_handler", self.pcl_handler, 10)
        self._cropped = self.create_publisher(PointCloud2, "pcl_cropped", 10)
        self._voxel = self.create_publisher(PointCloud2, "pcl_voxel", 10)
        self._inplane = self.create_publisher(PointCloud2, "pcl_inplane", 10)
        self._outplane = self.create_publisher(PointCloud2, "pcl_outplane", 10)

    def pcl_handler(self, pcl_msg: PointCloud2):
        """Receive a PointCloud2 message, process with Open3D, and publish results."""
        # --- Convert ROS2 PointCloud2 to an Open3D point cloud
        pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                sensor_msgs_py.point_cloud2.read_points_numpy(pcl_msg, ["x", "y", "z"])))

        # Crop the points based on a bounding box defined by minimum and maximum points
        crop_min = np.array([-0.75, -0.6, 0.1])
        crop_max = np.array([0.5, 0.1, 2.0])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_min, max_bound=crop_max)
        pcd_cropped = pcd.crop(bbox)

        # Publish the cropped pointcloud
        cropped_msg = sensor_msgs_py.point_cloud2.create_cloud_xyz32(
            pcl_msg.header, np.asarray(np.asarray(pcd_cropped.points)))
        self._cropped.publish(cropped_msg)

        # --- Downsample with voxel grid
        pcd_voxel = pcd_cropped.voxel_down_sample(voxel_size=0.01)
        voxel_points = np.asarray(pcd_voxel.points)
        voxel_msg = sensor_msgs_py.point_cloud2.create_cloud_xyz32(
            pcl_msg.header, voxel_points
        )
        self._voxel.publish(voxel_msg)

        # --- Plane segmentation
        ransac_n = 3
        if len(pcd_voxel.points) < ransac_n:
            return

        plane_model, inliers = pcd_voxel.segment_plane(
            distance_threshold=0.02,
            ransac_n=ransac_n,
            num_iterations=1000,
        )

        if len(inliers) == 0:
            self.get_logger().warn("No plane found.")
            return

        # Extract inliers (points lying on the plane)
        pcd_inplane = pcd_voxel.select_by_index(inliers)
        inplane_points = np.asarray(pcd_inplane.points)
        inplane_msg = sensor_msgs_py.point_cloud2.create_cloud_xyz32(
            pcl_msg.header, inplane_points
        )
        self._inplane.publish(inplane_msg)

        # Extract outliers (points not on the plane)
        pcd_outplane = pcd_voxel.select_by_index(inliers, invert=True)
        outplane_msg = sensor_msgs_py.point_cloud2.create_cloud_xyz32(
            pcl_msg.header, np.asarray(pcd_outplane.points))
        self._outplane.publish(outplane_msg)


def table_entry(args=None):
    rclpy.init(args=args)
    node = TableFind()
    rclpy.spin(node)
    rclpy.shutdown()
