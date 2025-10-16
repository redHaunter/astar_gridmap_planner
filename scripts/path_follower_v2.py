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
        self.aligning_to_path_direction = False

        # Control parameters
        self.linear_velocity = 0.2
        self.angular_velocity = 0.5

        # Control parameters
        self.k_linear = 1.0
        self.k_angular = 1.0
        self.max_linear = 0.35
        self.min_linear = 0.2
        self.max_angular = 1.0
        self.distance_threshold = 0.1  # distance to consider waypoint reached
        self.angle_deadband = math.radians(3)  # ignore small angle errors

        # Previous command for smoothing
        self.prev_linear = 0.0
        self.prev_angular = 0.0

        # Maximum rate of change per cycle (Δv / Δt)
        self.max_linear_accel = 0.01   # m/s per 0.05s (20 Hz), i.e. ~1.0 m/s²
        self.max_angular_accel = 0.03   # rad/s per 0.05s (20 Hz), i.e. ~2.0 rad/s²

        self.rate = rospy.Rate(20) # 20hz

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
        # Store goal point
        last_point = smooth_points[-1]
        # Sample path
        smooth_points = self.sample_path(smooth_points, step=0.3)
        # Add goal point if not sampled
        if smooth_points[-1] != last_point:
            smooth_points.append(last_point)

        # Check if robot is already at the last goal
        last_x, last_y = last_point
        dist_to_goal = self.get_distance_xy(self.robot_position.x, self.robot_position.y, last_x, last_y)
        if dist_to_goal < self.distance_threshold:
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
            pose.pose.position.z = 1.0
            smooth_path_msg.poses.append(pose)

        self.smooth_path_pub.publish(smooth_path_msg)  # Publish for RViz

        # Store path and start following
        self.path = smooth_path_msg.poses
        self.aligning_to_path_direction = True
        self.is_following_path = True
        rospy.loginfo("Started following smoothed path with {} points.".format(len(self.path)))


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
        
    def sample_path(self, points, step=0.1):
        """
        Resample the path so that consecutive points are spaced at least `step` meters apart.
        """
        if not points:
            return []

        sampled = [points[0]]
        last_x, last_y = points[0]

        for x, y in points[1:]:
            dist = math.hypot(x - last_x, y - last_y)
            if dist >= step:
                sampled.append((x, y))
                last_x, last_y = x, y

        # Make sure the final point is included
        if sampled[-1] != points[-1]:
            sampled.append(points[-1])

        return sampled
    
    def smooth_cmd(self, target_linear, target_angular):
        """
        Smooth velocity commands by limiting acceleration and angular acceleration.
        """
        # Linear
        linear_diff = target_linear - self.prev_linear
        max_linear_step = self.max_linear_accel
        if abs(linear_diff) > max_linear_step:
            target_linear = self.prev_linear + math.copysign(max_linear_step, linear_diff)

        # Angular
        angular_diff = target_angular - self.prev_angular
        max_angular_step = self.max_angular_accel
        if abs(angular_diff) > max_angular_step:
            target_angular = self.prev_angular + math.copysign(max_angular_step, angular_diff)

        # Update previous for next iteration
        self.prev_linear = target_linear
        self.prev_angular = target_angular

        return target_linear, target_angular

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


    def align_with_path_direction(self, first_pose):
        """
        Align robot to face the direction from current position to final goal point.
        """
        dx = first_pose.position.x - self.robot_position.x
        dy = first_pose.position.y - self.robot_position.y
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
        
    def find_farthest_collinear_index(self, path_points, start_index=0, threshold=0.05):
        """
        Starting from start_index, find the farthest index where all points between
        start and that index are approximately collinear.
        """
        if start_index >= len(path_points) - 1:
            return start_index

        base_point = (path_points[start_index].pose.position.x,
                    path_points[start_index].pose.position.y)
        next_index = start_index + 1
        last_valid_index = next_index

        # Keep extending as long as points stay on the line
        while next_index + 1 < len(path_points):
            p_next = (path_points[next_index + 1].pose.position.x,
                    path_points[next_index + 1].pose.position.y)
            p_last = (path_points[last_valid_index].pose.position.x,
                    path_points[last_valid_index].pose.position.y)
            # Check if the next point lies on the line formed by base_point and last_valid_point
            if self.is_point_on_line(base_point, p_last, p_next, threshold):
                last_valid_index = next_index + 1
                next_index += 1
            else:
                break
        return last_valid_index
        
    def is_point_on_line(self, p1, p2, p3, threshold=0.01):
        """
        Check if p3 lies approximately on the line segment from p1 to p2.
        Uses perpendicular distance from p3 to line p1-p2.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Handle degenerate case
        if math.hypot(x2 - x1, y2 - y1) < 1e-6:
            return False

        # Area of triangle * 2 / base = height
        numerator = abs((y2 - y1)*x3 - (x2 - x1)*y3 + x2*y1 - y2*x1)
        denominator = math.hypot(y2 - y1, x2 - x1)
        distance = numerator / denominator

        return distance <= threshold


        
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

    def normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    
    def follow_path(self):
        if not self.path:
            self.cmd_vel_pub.publish(Twist())  # stop
            self.is_following_path = False
            return

        # Align before starting
        if self.aligning_to_path_direction:
            aligned = self.align_with_path_direction(self.path[1].pose)
            if aligned:
                self.aligning_to_path_direction = False
                self.cmd_vel_pub.publish(Twist())
                rospy.loginfo("Aligned with path direction. Beginning movement.")
            else:
                return

        # Find farthest collinear waypoint to target
        target_index = self.find_farthest_collinear_index(self.path, start_index=0, threshold=0.05)
        target_pose = self.path[target_index].pose

        # Compute control
        dx = target_pose.position.x - self.robot_position.x
        dy = target_pose.position.y - self.robot_position.y

        distance = math.hypot(dx, dy)
        angle_to_goal = self.normalize_angle(math.atan2(dy, dx))
        angle_diff = self.normalize_angle(angle_to_goal - self.robot_orientation)

        # Proportional control
        linear = self.k_linear * distance
        linear = max(min(linear, self.max_linear), self.min_linear) if distance > self.distance_threshold else 0.0

        if abs(angle_diff) < self.angle_deadband:
            angle_diff = 0.0
        angular = max(min(self.k_angular * angle_diff, self.max_angular), -self.max_angular)

        # Smooth command
        linear_smooth, angular_smooth = self.smooth_cmd(linear, angular)

        cmd_vel = Twist()
        cmd_vel.linear.x = linear_smooth
        cmd_vel.angular.z = angular_smooth
        self.cmd_vel_pub.publish(cmd_vel)

        # Waypoint reached?
        if distance < self.distance_threshold:
            # Remove all waypoints up to target_index
            self.path = self.path[target_index+1:]
            if not self.path:
                rospy.loginfo("Reached final waypoint.")
                self.cmd_vel_pub.publish(Twist())
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
