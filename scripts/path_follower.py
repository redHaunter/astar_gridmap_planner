#!/usr/bin/env python3

import rospy
import math
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry

try:
    from scipy.interpolate import splprep, splev
    has_scipy = True
except ImportError:
    has_scipy = False


class PathFollower:
    def __init__(self):
        rospy.init_node('path_follower_node', anonymous=True)

        # Publishers and Subscribers
        self.path_sub = rospy.Subscriber("/planned_path", Path, self.path_callback)
        self.odom_sub = rospy.Subscriber("/spot/odometry", Odometry, self.odom_callback)
        self.cmd_vel_pub = rospy.Publisher("/spot/cmd_vel", Twist, queue_size=10)
        self.smooth_path_pub = rospy.Publisher("/smooth_path", Path, queue_size=1)

        # Robot state
        self.robot_position = None
        self.robot_orientation = None
        self.path = []
        self.is_following_path = False
        self.aligning_to_goal_direction = False

        # Control parameters
        self.linear_velocity = 0.2
        self.angular_velocity = 0.5

        self.rate = rospy.Rate(10)

    def path_callback(self, msg):
        if self.is_following_path:
            rospy.loginfo("Currently following a path. Ignoring new one.")
            return

        if not self.robot_position:
            rospy.loginfo("Robot position unknown. Ignoring path.")
            return

        # Convert to simple list of x, y
        raw_points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        raw_points = self.skip_loops_in_path(raw_points)

        if len(raw_points) < 2:
            rospy.logwarn("Received path too short to smooth.")
            return

        # Smooth the path
        smooth_points = self.smooth_path(raw_points)

        # Check if robot is already at the last goal
        last_x, last_y = smooth_points[-1]
        dist_to_goal = self.get_distance_xy(self.robot_position.x, self.robot_position.y, last_x, last_y)
        if dist_to_goal < 0.2:
            rospy.loginfo("Robot is already at the goal. Ignoring path.")
            return

        # Convert to Path message
        smooth_path_msg = Path()
        smooth_path_msg.header = msg.header
        for x, y in smooth_points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0
            smooth_path_msg.poses.append(pose)

        self.smooth_path_pub.publish(smooth_path_msg)  # Publish for RViz

        # Store path and start following
        self.path = smooth_path_msg.poses
        self.aligning_to_goal_direction = True
        self.is_following_path = True
        rospy.loginfo("Started following smoothed path with {} points.".format(len(self.path)))

    # def smooth_path(self, points, angle_threshold_deg=20):
    #     if len(points) < 3:
    #         return points  # Not enough points to smooth

    #     def angle_between(p1, p2):
    #         return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

    #     smoothed = [points[0]]
    #     prev_angle = angle_between(points[0], points[1])

    #     for i in range(1, len(points) - 1):
    #         current_angle = angle_between(points[i], points[i + 1])
    #         angle_diff = abs(current_angle - prev_angle)
    #         angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]

    #         if abs(angle_diff) * 180 / math.pi > angle_threshold_deg:
    #             smoothed.append(points[i])  # Keep point only if angle changes too much

    #         prev_angle = current_angle

    #     smoothed.append(points[-1])  # Always include last point
    #     return smoothed

    def smooth_path(self, points, smoothing=1.0, resolution=100):
        if has_scipy and len(points) >= 3:
            try:
                x, y = zip(*points)
                tck, _ = splprep([x, y], s=smoothing)
                u_fine = [i * 1.0 / resolution for i in range(resolution + 1)]
                x_fine, y_fine = splev(u_fine, tck)
                return list(zip(x_fine, y_fine))
            except Exception as e:
                rospy.logwarn("Spline smoothing failed: {}. Falling back to raw path.".format(str(e)))
                return points
        else:
            rospy.loginfo("Scipy not available or path too short, using linear interpolation.")
            return points

    def skip_loops_in_path(self, points, distance_threshold=0.1, min_index_gap=3):
        """
        Detect loops and skip them: if point i is close to point j > i + gap,
        skip the intermediate points and jump from i to j.
        """
        i = 0
        n = len(points)
        new_path = []

        while i < n:
            xi, yi = points[i]
            new_path.append((xi, yi))

            # Look ahead to find close future point
            jumped = False
            for j in range(i + min_index_gap, n):
                xj, yj = points[j]
                dist = math.hypot(xi - xj, yi - yj)
                if dist < distance_threshold:
                    rospy.logwarn(f"Loop detected from point {i} to {j}, skipping intermediate points.")
                    i = j  # Jump to j
                    jumped = True
                    break

            if not jumped:
                i += 1  # Move forward normally

        return new_path


    def align_with_goal_direction(self, goal_pose):
        """
        Align robot to face the direction from current position to final goal point.
        """
        dx = goal_pose.position.x - self.robot_position.x
        dy = goal_pose.position.y - self.robot_position.y
        target_angle = math.atan2(dy, dx)

        angle_diff = target_angle - self.robot_orientation
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]

        if abs(angle_diff) < 0.05:  # Within ~3 degrees
            return True
        else:
            cmd_vel = Twist()
            cmd_vel.angular.z = self.angular_velocity * math.copysign(1, angle_diff)
            self.cmd_vel_pub.publish(cmd_vel)
            return False
        
    def odom_callback(self, msg):
        self.robot_position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        _, _, self.robot_orientation = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])

    def get_distance(self, pos1, pos2):
        return self.get_distance_xy(pos1.x, pos1.y, pos2.x, pos2.y)

    def get_distance_xy(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def get_angle_to_goal(self, goal_pose):
        dx = goal_pose.position.x - self.robot_position.x
        dy = goal_pose.position.y - self.robot_position.y
        return math.atan2(dy, dx)

    def follow_path(self):
        if not self.path:
            # Final orientation alignment skipped here (since we're rolling back)
            self.cmd_vel_pub.publish(Twist())
            self.is_following_path = False
            return
        
        # Align before starting
        if self.aligning_to_goal_direction:
            aligned = self.align_with_goal_direction(self.path[-1].pose)
            if aligned:
                self.aligning_to_goal_direction = False
                rospy.loginfo("Aligned with goal direction. Beginning movement.")
            else:
                return  # Keep rotating before moving
            
        goal_pose = self.path[0].pose
        distance = self.get_distance(self.robot_position, goal_pose.position)
        angle_to_goal = self.get_angle_to_goal(goal_pose)
        angle_diff = angle_to_goal - self.robot_orientation
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        cmd_vel = Twist()

        # === Proportional control gains ===
        k_linear = 0.5
        k_angular = 1.0
        max_linear = 0.4
        max_angular = 1.0
        min_linear = 0.2

        if distance > 0.1:
            # Compute linear and angular speeds
            linear = k_linear * distance
            linear = max(min(linear, max_linear), min_linear)

            angular = k_angular * angle_diff
            angular = max(min(angular, max_angular), -max_angular)

            # Reverse logic if goal is behind
            if abs(angle_diff) > 2 * math.pi / 3:
                linear *= -1
                angular *= -1

            cmd_vel.linear.x = linear
            cmd_vel.angular.z = angular
        else:
            rospy.loginfo("Reached waypoint.")
            self.path.pop(0)

        self.cmd_vel_pub.publish(cmd_vel)

    def run(self):
        while not rospy.is_shutdown():
            if self.is_following_path:
                self.follow_path()
            self.rate.sleep()


if __name__ == '__main__':
    try:
        PathFollower().run()
    except rospy.ROSInterruptException:
        pass
