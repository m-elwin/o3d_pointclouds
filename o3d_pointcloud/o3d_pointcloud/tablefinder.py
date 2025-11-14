"""
Use a 3D camera and Open3D to detect objects that are on a table.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Vector3
import sensor_msgs_py.point_cloud2
from visualization_msgs.msg import Marker
import numpy as np
import open3d as o3d

def ros_to_o3d(pcl_msg : PointCloud2):
    """Convert A ros2 PointCloud2 message into an Open3D message.

    Args:
    -----
    pcl_msg - A ROS 2 PointCloud2 message to convert

    Returns:
    --------
    A Open3D PointCloud
    """
    return o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(
            sensor_msgs_py.point_cloud2.read_points_numpy(pcl_msg, ["x", "y", "z"])))

def o3d_to_ros(pcd : o3d.geometry.PointCloud, header : Header):
    """Convert An Open3D PointCloud to a ROS 2 PointCloud2 message

    Args:
    -----
    pcd - the Open3D pointcloud
    header - The frame and timestamp of the point cloud data

    Returns:
    --------
    pcl_msg - A ROS 2 PointCloud2 message
    """
    return sensor_msgs_py.point_cloud2.create_cloud_xyz32(
        header, np.asarray(pcd.points))

class TableFind(Node):
    def __init__(self):
        """
        Create a ROS Node that extracts a table (or other flat surface) from a point cloud
        using Open3D instead of PCL.

        Subscribers
        -----------
        pcl_handler (sensor_msgs/msg/PointCloud2) The input point cloud

        Publishers
        ----------
        pcl_cropped (sensor_msgs/msg/PointCloud2) The input points, after cropping
        pcl_voxel (sensor_msgs/msg/PointCloud2)   The voxelized points from the cropped point cloud
        pcl_inplane (sensor_msgs/msg/PointCloud2) The points that are in the table
        pcl_outplane (sensor_msgs/msg/PointCloud2) The points that are not in the table

        Parameters
        ----------
        xmin (float) - The minimum x coordinate for cropping
        xmax (float) - The maximum x coordinate for cropping
        ymin (float) - The minimum y coordinate for cropping
        ymay (float) - The mayimum y coordinate for cropping
        zmin (float) - The minimum z coordinate for cropping
        zmaz (float) - The mazimum z coordinate for cropping
        """
        super().__init__("table_find")

        # Parameters for the various filters
        self._xmin = self.declare_parameter("xmin", -1.0).value
        self._xmax = self.declare_parameter("xmax", 1.0).value
        self._ymin = self.declare_parameter("ymin", -1.0).value
        self._ymax = self.declare_parameter("ymax", 1.0).value
        self._zmin = self.declare_parameter("zmin", -1.0).value
        self._zmax = self.declare_parameter("zmax", 1.0).value

        self.add_on_set_parameters_callback(self.update_parameters)

        self._sub = self.create_subscription(PointCloud2, "pcl_handler", self.pcl_handler, 10)
        self._cropped = self.create_publisher(PointCloud2, "pcl_cropped", 10)
        self._voxel = self.create_publisher(PointCloud2, "pcl_voxel", 10)
        self._inplane = self.create_publisher(PointCloud2, "pcl_inplane", 10)
        self._outplane = self.create_publisher(PointCloud2, "pcl_outplane", 10)

        self._marker = self.create_publisher(Marker,
                                             'visualization_marker',
                                             QoSProfile(depth=10,
                                                        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))

    def draw_cropbox(self, frame):
        """Draw the outline of a cropbox as a Marker.

        Args:
        -----
        frame (Str) - the name of the frame in which to draw the marker
        """
        # define the box's vertices
        v1 = Point(x=self._xmin, y=self._ymin, z=self._zmin)
        v2 = Point(x=self._xmax, y=self._ymin, z=self._zmin)
        v3 = Point(x=self._xmax, y=self._ymin, z=self._zmax)
        v4 = Point(x=self._xmin, y=self._ymin, z=self._zmax)
        v5 = Point(x=self._xmin, y=self._ymax, z=self._zmin)
        v6 = Point(x=self._xmax, y=self._ymax, z=self._zmin)
        v7 = Point(x=self._xmax, y=self._ymax, z=self._zmax)
        v8 = Point(x=self._xmin, y=self._ymax, z=self._zmax)

        self._marker.publish(
            Marker(
                header=Header(frame_id=frame, stamp=self.get_clock().now().to_msg()),
                ns="table_finder",
                id=0,
                type=Marker.LINE_LIST,
                action=Marker.ADD,
                scale=Vector3(x=0.01),
                points=[v1, v2,
                        v2, v3,
                        v3, v4,
                        v4, v1,
                        v5, v6,
                        v6, v7,
                        v7, v8,
                        v8, v5,
                        v1, v5,
                        v2, v6,
                        v3, v7,
                        v4, v8],
                color=ColorRGBA(r=0.5, g=0.0, b=0.5, a=1.0),
                frame_locked=True))


    def update_parameters(self, params):
        """Update parameters. All parameters are listed as self._(paramname) in this class."""
        for p in params:
            setattr(self, f"_{p.name}", p.value)
        return SetParametersResult(successful=True)

    def pcl_handler(self, pcl_msg: PointCloud2):
        """Receive a PointCloud2 message, process with Open3D, and publish results."""
        # --- Convert ROS2 PointCloud2 to an Open3D point cloud
        pcd = ros_to_o3d(pcl_msg)

        if len(pcd.points) < 1:
            self.get_logger().warn("Not enough points for cropping.")
            return

        # Crop the points based on a bounding box defined by minimum and maximum points
        crop_min = np.array([self._xmin, self._ymin, self._zmin])
        crop_max = np.array([self._xmax, self._ymax, self._zmax])
        # Visualize the cropbox
        self.draw_cropbox(pcl_msg.header.frame_id)

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_min, max_bound=crop_max)
        pcd_cropped = pcd.crop(bbox)

        # Publish the results of the cropping.
        # Since this is a transformation of the original pointcloud, we use
        # The same timestamp and frame as the original pcl message
        self._cropped.publish(o3d_to_ros(pcd_cropped, pcl_msg.header))

        # Downsample with voxel grid
        pcd_voxel = pcd_cropped.voxel_down_sample(voxel_size=0.01)
        self._voxel.publish(o3d_to_ros(pcd_voxel, pcl_msg.header))

        # Plane segmentation
        # we need at least some points for the segmentation to work well
        if len(pcd_voxel.points) < 10:
            self.get_logger().warn("Not enough points for segmentation.")
            return

        plane_model, inliers = pcd_voxel.segment_plane(
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=1000,
        )

        if len(inliers) == 0:
            self.get_logger().warn("No plane found.")
            return

        # Extract inliers (points lying on the plane)
        pcd_inplane = pcd_voxel.select_by_index(inliers)
        self._inplane.publish(o3d_to_ros(pcd_inplane, pcl_msg.header))

        # Extract outliers (points not on the plane)
        pcd_outplane = pcd_voxel.select_by_index(inliers, invert=True)
        self._outplane.publish(o3d_to_ros(pcd_outplane, pcl_msg.header))


def table_entry(args=None):
    rclpy.init(args=args)
    node = TableFind()
    rclpy.spin(node)
    rclpy.shutdown()
