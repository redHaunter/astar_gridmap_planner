#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <queue>
#include <unordered_map>
#include <Eigen/Dense>

struct Node {
  Eigen::Array2i idx;
  double g;
  double f;
  Eigen::Array2i parent;
};

struct CompareF {
  bool operator()(const Node& a, const Node& b) {
    return a.f > b.f;
  }
};

class AStarPlanner {
public:
  AStarPlanner(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : tfListener_(tfBuffer_) {

    pnh.param("grid_map_topic", grid_map_topic_, std::string("/elevation_mapping/elevation_map_raw"));
    pnh.param("traversability_layer", layer_, std::string("traversability"));
    pnh.param("path_topic", path_topic_, std::string("/planned_path"));
    pnh.param("cost_scale", cost_scale_, 100.0);
    pnh.param("traversability_obstacle_threshold", obstacle_threshold_, 0.5);
    pnh.param("allow_diagonal", allow_diagonal_, true);
    pnh.param("map_frame", map_frame_, std::string("odom"));

    gridMapSub_ = nh.subscribe(grid_map_topic_, 1, &AStarPlanner::gridMapCallback, this);
    odomSub_ = nh.subscribe("/spot/odometry", 1, &AStarPlanner::odomCallback, this);
    goalSub_ = nh.subscribe("/move_base_simple/goal", 1, &AStarPlanner::goalCallback, this);
    pathPub_ = nh.advertise<nav_msgs::Path>(path_topic_, 1, true);
    ROS_INFO_STREAM("Travesability_Threshold: " << obstacle_threshold_);

  }

private:
  ros::Subscriber gridMapSub_, odomSub_, goalSub_;
  ros::Publisher pathPub_;
  grid_map::GridMap gridMap_;
  bool hasMap_ = false;
  geometry_msgs::PoseStamped currentPose_;
  bool hasPose_ = false;
  tf2_ros::Buffer tfBuffer_;
  tf2_ros::TransformListener tfListener_;

  std::string grid_map_topic_, layer_, path_topic_, map_frame_;
  double cost_scale_, obstacle_threshold_;
  bool allow_diagonal_;

  void gridMapCallback(const grid_map_msgs::GridMap& msg) {
    grid_map::GridMapRosConverter::fromMessage(msg, gridMap_);
    hasMap_ = true;
  }

  void odomCallback(const nav_msgs::Odometry& msg) {
    geometry_msgs::PoseStamped pose;
    pose.header = msg.header;
    pose.pose = msg.pose.pose;
    currentPose_ = pose;
    hasPose_ = true;
  }

  void goalCallback(const geometry_msgs::PoseStamped& goal) {
    if (!hasMap_ || !hasPose_) {
      ROS_WARN("No map or pose yet");
      return;
    }

    geometry_msgs::PoseStamped start;
    try {
      tfBuffer_.transform(currentPose_, start, map_frame_, ros::Duration(0.1));
    } catch (...) {
      ROS_ERROR("TF transform failed for start");
      return;
    }

    geometry_msgs::PoseStamped goal_tf;
    try {
      tfBuffer_.transform(goal, goal_tf, map_frame_, ros::Duration(0.1));
    } catch (...) {
      ROS_ERROR("TF transform failed for goal");
      return;
    }

    nav_msgs::Path path = planPath(start, goal_tf);
    if (!path.poses.empty()) {
      pathPub_.publish(path);
      ROS_INFO("Path found");
    } else {
      ROS_WARN("No path found");
    }
  }

  nav_msgs::Path planPath(const geometry_msgs::PoseStamped& start,
                          const geometry_msgs::PoseStamped& goal) {
    nav_msgs::Path path;
    path.header.frame_id = map_frame_;
    path.header.stamp = ros::Time::now();

    Eigen::Vector2d start_pos(start.pose.position.x, start.pose.position.y);
    Eigen::Vector2d goal_pos(goal.pose.position.x, goal.pose.position.y);

    Eigen::Array2i start_idx, goal_idx;
    if (!gridMap_.getIndex(start_pos, start_idx) || !gridMap_.getIndex(goal_pos, goal_idx)) {
      ROS_ERROR("Start or goal outside gridmap");
      return path;
    }

    std::priority_queue<Node, std::vector<Node>, CompareF> open;
    std::unordered_map<long long, Node> allNodes;

    auto idxToKey = [](const Eigen::Array2i& idx) {
      return (static_cast<long long>(idx.x()) << 32) | (unsigned int)idx.y();
    };

    Node start_node{start_idx, 0.0, (start_pos - goal_pos).norm(), start_idx};
    open.push(start_node);
    allNodes[idxToKey(start_idx)] = start_node;

    bool found = false;
    Node current;

    std::vector<Eigen::Array2i> moves;
    moves.push_back(Eigen::Array2i(1,0));
    moves.push_back(Eigen::Array2i(-1,0));
    moves.push_back(Eigen::Array2i(0,1));
    moves.push_back(Eigen::Array2i(0,-1));
    if (allow_diagonal_) {
      moves.push_back(Eigen::Array2i(1,1));
      moves.push_back(Eigen::Array2i(1,-1));
      moves.push_back(Eigen::Array2i(-1,1));
      moves.push_back(Eigen::Array2i(-1,-1));
    }

    while (!open.empty()) {
      current = open.top();
      open.pop();

      if ((current.idx == goal_idx).all()) { found = true; break; }

      for (auto& m : moves) {
        Eigen::Array2i neighbor_idx = current.idx + m;
        if ((neighbor_idx < 0).any() || (neighbor_idx >= gridMap_.getSize()).any()) continue;
        if (!gridMap_.isValid(neighbor_idx, layer_)) continue;

        double curr_val = gridMap_.at(layer_, current.idx);
        double neigh_val = gridMap_.at(layer_, neighbor_idx);
        if (curr_val < obstacle_threshold_ || neigh_val < obstacle_threshold_) continue;
        
        
        double avg_trav = 0.5 * (curr_val + neigh_val);
        double distance = m.cast<double>().matrix().norm();
        double move_cost = distance * (1.0 + (1.0 - avg_trav) * cost_scale_);

        double g_new = current.g + move_cost;
        Eigen::Vector2d neigh_pos;
        gridMap_.getPosition(neighbor_idx, neigh_pos);
        double h = (neigh_pos - goal_pos).norm();
        double f = g_new + h;

        long long key = idxToKey(neighbor_idx);
        if (!allNodes.count(key) || g_new < allNodes[key].g) {
          Node n{neighbor_idx, g_new, f, current.idx};
          allNodes[key] = n;
          open.push(n);
          ROS_INFO("avg traversability value: %f", avg_trav);
        }
      }
    }

    if (!found) return path;

    std::vector<Eigen::Array2i> rev;
    Eigen::Array2i cur = current.idx;
    while ((cur != start_idx).any()) {
      rev.push_back(cur);
      long long key = idxToKey(cur);
      cur = allNodes[key].parent;
    }
    rev.push_back(start_idx);

    std::reverse(rev.begin(), rev.end());
    for (auto& c : rev) {
      Eigen::Vector2d pos;
      gridMap_.getPosition(c, pos);
      geometry_msgs::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = pos.x();
      pose.pose.position.y = pos.y();
      pose.pose.position.z = 1.0;
      pose.pose.orientation.w = 1.0;
      path.poses.push_back(pose);
    }
    return path;
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "astar_gridmap_planner");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  AStarPlanner planner(nh, pnh);
  ros::spin();
  return 0;
}
