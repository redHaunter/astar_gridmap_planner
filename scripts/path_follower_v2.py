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

        # Control parameters
        self.linear_velocity = 0.2
        self.angular_velocity = 1.0

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
        self.is_following_path = True
        rospy.loginfo("Started following smoothed path with {} points.".format(len(self.path)))

    def smooth_path(self, points, angle_threshold_deg=20):
        if len(points) < 3:
            return points  # Not enough points to smooth

        def angle_between(p1, p2):
            return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

        smoothed = [points[0]]
        prev_angle = angle_between(points[0], points[1])

        for i in range(1, len(points) - 1):
            current_angle = angle_between(points[i], points[i + 1])
            angle_diff = abs(current_angle - prev_angle)
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]

            if abs(angle_diff) * 180 / math.pi > angle_threshold_deg:
                smoothed.append(points[i])  # Keep point only if angle changes too much

            prev_angle = current_angle

        smoothed.append(points[-1])  # Always include last point
        return smoothed

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
            self.is_following_path = False
            return

        goal_pose = self.path[0].pose
        distance = self.get_distance(self.robot_position, goal_pose.position)
        angle_to_goal = self.get_angle_to_goal(goal_pose)  # Desired heading to goal
        angle_diff = angle_to_goal - self.robot_orientation

        # Normalize angle_diff to [-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        cmd_vel = Twist()

        if distance > 0.1:
            # Decide if reversing is better than turning around
            # Threshold 90 degrees (pi/2 radians)
            if abs(angle_diff) > math.pi / 2:
                # Reverse: drive backward and turn opposite direction
                cmd_vel.linear.x = -self.linear_velocity
                # Adjust angular velocity so robot turns toward goal while reversing
                if angle_diff > 0:
                    cmd_vel.angular.z = -self.angular_velocity
                else:
                    cmd_vel.angular.z = self.angular_velocity
            else:
                # Drive forward normally
                cmd_vel.linear.x = self.linear_velocity
                if abs(angle_diff) > 0.1:
                    cmd_vel.angular.z = self.angular_velocity * math.copysign(1, angle_diff)
                else:
                    cmd_vel.angular.z = 0.0
        else:
            # Close enough to goal point
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            rospy.loginfo("Reached intermediate goal.")
            self.path.pop(0)

        self.cmd_vel_pub.publish(cmd_vel)

        if not self.path:
            rospy.loginfo("Finished following path.")
            self.is_following_path = False


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
