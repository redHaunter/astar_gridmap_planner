#!/usr/bin/env python3
import rospy
import random
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class PathCollector:
    def __init__(self):
        rospy.init_node("path_collector")

        # Subscribe to the incoming Path topic
        rospy.Subscriber("/planned_path", Path, self.path_callback)

        # Publisher for MarkerArray to visualize all stored paths
        self.marker_pub = rospy.Publisher("/path_array", MarkerArray, queue_size=10)

        # Store received paths and their colors
        self.paths_with_colors = []

        # Timer to publish paths at regular intervals
        rospy.Timer(rospy.Duration(1.0), self.publish_paths)  # 1 Hz

    def path_callback(self, msg):
        rospy.loginfo("Received new path with %d poses", len(msg.poses))

        # Assign a random color once when the path is received
        color = self.get_random_color()

        # Store as a tuple (path, color)
        self.paths_with_colors.append((msg, color))

    def get_random_color(self):
        """Generate a random RGB color with full alpha."""
        return [random.random() for _ in range(3)] + [1.0]  # [r, g, b, a]

    def publish_paths(self, event):
        marker_array = MarkerArray()

        for idx, (path, color) in enumerate(self.paths_with_colors):
            marker = Marker()
            marker.header = path.header
            marker.ns = "paths"
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.2

            # Use stored color
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]

            # Convert path poses to marker points
            marker.points = [pose.pose.position for pose in path.poses]

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

if __name__ == "__main__":
    try:
        PathCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
