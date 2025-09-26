// A* 2.5D planner for ROS1 Noetic using grid_map::GridMap
// File: astar_gridmap_2p5d_planner.cpp
// This node plans in a 2.5D graph built from a GridMap with at least the layers:
//   - elevation  (z value for each cell)
//   - traversability (0..1, where 1=best traversable)
// The planner treats navigable *cells* (treads/surfaces) as nodes and allows
// vertical transitions (steps/stairs) when elevation differences are within
// configured step/climb thresholds. It also supports a configurable neighbor
// radius so the planner may reach slightly farther traversable cells (e.g., next
// stair tread) when immediate neighbors are untraversable.

#include <ros/ros.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <queue>
#include <unordered_map>
#include <vector>
#include <limits>
#include <cmath>
#include <mutex>
#include <algorithm>
#include <Eigen/Dense>

using grid_map::GridMap;
using grid_map::Position;

struct Node {
  Eigen::Array2i idx;            // grid index
  double g = 0.0;       // cost from start
  double f = 0.0;       // g + h
  Eigen::Array2i parent;         // parent index
  bool has_parent = false;
};

struct CompareF {
  bool operator()(const Node &a, const Node &b) const { return a.f > b.f; }
};

class AStar2p5DPlanner {
public:
  AStar2p5DPlanner(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    : tfListener_(tfBuffer_) {
    // parameters
    pnh.param<std::string>("grid_map_topic", grid_map_topic_, "/elevation_mapping/elevation_map_raw");
    pnh.param<std::string>("elevation_layer", elevation_layer_, "elevation");
    pnh.param<std::string>("traversability_layer", trav_layer_, "inferenced_traversability");
    pnh.param<std::string>("path_topic", path_topic_, "/planned_path");
    pnh.param<std::string>("map_frame", map_frame_, std::string("odom"));

    pnh.param<double>("traversability_threshold", trav_threshold_, 0.2);
    pnh.param<double>("cost_scale", cost_scale_, 100.0);
    pnh.param<double>("max_step_up", max_step_up_, 0.25);   // meters
    pnh.param<double>("max_drop_down", max_drop_down_, 0.5); // meters
    pnh.param<double>("max_slope_deg", max_slope_deg_, 45.0); // degrees
    pnh.param<double>("vertical_cost_weight", vertical_cost_weight_, 5.0);
    pnh.param<bool>("allow_diagonal", allow_diagonal_, true);

    // neighbor radius (meters) - allows reaching slightly farther traversable cells
    pnh.param<double>("neighbor_radius", neighbor_radius_, 0.5);

    grid_map_sub_ = nh.subscribe(grid_map_topic_, 1, &AStar2p5DPlanner::gridMapCallback, this);
    odom_sub_ = nh.subscribe("/spot/odometry", 5, &AStar2p5DPlanner::odomCallback, this);
    goal_sub_ = nh.subscribe("/move_base_simple/goal", 5, &AStar2p5DPlanner::goalCallback, this);
    path_pub_ = nh.advertise<nav_msgs::Path>(path_topic_, 1, true);

    ROS_INFO("AStar2.5D: listening to '%s' (elev='%s' trav='%s') frame='%s' | neighbor_radius=%.2f m", grid_map_topic_.c_str(), elevation_layer_.c_str(), trav_layer_.c_str(), map_frame_.c_str(), neighbor_radius_);
  }

private:
  // ROS
  ros::Subscriber grid_map_sub_, odom_sub_, goal_sub_;
  ros::Publisher path_pub_;
  tf2_ros::Buffer tfBuffer_;
  tf2_ros::TransformListener tfListener_;

  // params
  std::string grid_map_topic_, elevation_layer_, trav_layer_, path_topic_, map_frame_;
  double trav_threshold_, cost_scale_, max_step_up_, max_drop_down_, max_slope_deg_, vertical_cost_weight_;
  bool allow_diagonal_;
  double neighbor_radius_;

  // map/state
  GridMap gridMap_;
  std::mutex map_mutex_;
  bool have_map_ = false;
  std::vector<Eigen::Array2i> neighbor_offsets_;

  geometry_msgs::PoseStamped last_odom_pose_;
  bool have_odom_ = false;

  void gridMapCallback(const grid_map_msgs::GridMapConstPtr &msg) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    try {
      GridMap tmp;
      grid_map::GridMapRosConverter::fromMessage(*msg, tmp);
      gridMap_ = tmp;
      have_map_ = true;
      // (re)compute neighbor offsets when map arrives or resolution changes
      precomputeNeighborOffsets();
    } catch (std::exception &e) {
      ROS_WARN("GridMap conversion failed: %s", e.what());
      have_map_ = false;
    }
  }

  void odomCallback(const nav_msgs::OdometryConstPtr &msg) {
    geometry_msgs::PoseStamped p;
    p.header = msg->header;
    p.pose = msg->pose.pose;
    // transform to map_frame_ if needed
    if (!have_map_ || gridMap_.getFrameId().empty() || gridMap_.getFrameId() == p.header.frame_id) {
      last_odom_pose_ = p;
      have_odom_ = true;
      return;
    }
    try {
      tfBuffer_.transform(p, last_odom_pose_, gridMap_.getFrameId(), ros::Duration(0.1));
      have_odom_ = true;
    } catch (tf2::TransformException &ex) {
      ROS_DEBUG("odom transform failed: %s", ex.what());
    }
  }

  void goalCallback(const geometry_msgs::PoseStampedConstPtr &msg) {
    if (!have_map_) { ROS_WARN("No grid map available yet"); return; }
    if (!have_odom_) { ROS_WARN("No odom pose yet"); return; }

    geometry_msgs::PoseStamped goal_in_map;
    try {
      tfBuffer_.transform(*msg, goal_in_map, gridMap_.getFrameId(), ros::Duration(0.1));
    } catch (tf2::TransformException &ex) {
      if (msg->header.frame_id == gridMap_.getFrameId()) goal_in_map = *msg;
      else { ROS_WARN("Cannot transform goal into grid map frame: %s", ex.what()); return; }
    }

    geometry_msgs::PoseStamped start_in_map = last_odom_pose_;
    // ensure frame matches
    if (start_in_map.header.frame_id != gridMap_.getFrameId()) {
      try { tfBuffer_.transform(last_odom_pose_, start_in_map, gridMap_.getFrameId(), ros::Duration(0.1)); }
      catch (tf2::TransformException &ex) { ROS_WARN("start transform failed: %s", ex.what()); return; }
    }

    nav_msgs::Path path = planPath(start_in_map, goal_in_map);
    if (!path.poses.empty()) {
      path.header.stamp = ros::Time::now();
      path_pub_.publish(path);
      ROS_INFO("Published 2.5D path with %lu poses", path.poses.size());
    } else {
      ROS_WARN("2.5D planner: no path found");
    }
  }

  // helper: make 64-bit key from index
  inline long long idxToKey(const Eigen::Array2i &i) const {
    unsigned long long ux = static_cast<unsigned long long>(static_cast<int>(i.x()));
    unsigned long long uy = static_cast<unsigned long long>(static_cast<int>(i.y()));
    return (static_cast<long long>(ux) << 32) | static_cast<long long>(uy);
  }

  // precompute neighbor offsets within neighbor_radius_, sorted by distance (closest first)
  void precomputeNeighborOffsets() {
    neighbor_offsets_.clear();
    if (!have_map_) return;
    double res = gridMap_.getResolution();
    int max_steps = std::max(1, static_cast<int>(std::ceil(neighbor_radius_ / res)));
    for (int dx = -max_steps; dx <= max_steps; ++dx) {
      for (int dy = -max_steps; dy <= max_steps; ++dy) {
        if (dx == 0 && dy == 0) continue;
        double dist = std::hypot(dx * res, dy * res);
        if (dist <= neighbor_radius_) {
          neighbor_offsets_.emplace_back(dx, dy);
        }
      }
    }
    std::sort(neighbor_offsets_.begin(), neighbor_offsets_.end(), [&](const Eigen::Array2i &a, const Eigen::Array2i &b){
      double da = std::hypot(a.x() * gridMap_.getResolution(), a.y() * gridMap_.getResolution());
      double db = std::hypot(b.x() * gridMap_.getResolution(), b.y() * gridMap_.getResolution());
      return da < db;
    });
    // ROS_INFO("Precomputed %zu neighbor offsets (radius=%.2f m, resolution=%.3f m)", neighbor_offsets_.size(), neighbor_radius_, gridMap_.getResolution());
  }

  nav_msgs::Path planPath(const geometry_msgs::PoseStamped &start_pose, const geometry_msgs::PoseStamped &goal_pose) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    nav_msgs::Path out;
    out.header.frame_id = gridMap_.getFrameId();
    out.header.stamp = ros::Time::now();

    // positions
    Eigen::Vector2d start_xy(start_pose.pose.position.x, start_pose.pose.position.y);
    Eigen::Vector2d goal_xy(goal_pose.pose.position.x, goal_pose.pose.position.y);

    Eigen::Array2i start_idx, goal_idx;
    if (!gridMap_.getIndex(start_xy, start_idx) || !gridMap_.getIndex(goal_xy, goal_idx)) {
      ROS_WARN("Start or goal outside grid map bounds");
      return out;
    }

    // quick check: start/goal traversability
    if (!gridMap_.exists(trav_layer_) || !gridMap_.exists(elevation_layer_)) {
      ROS_WARN("Required layers missing (%s, %s)", trav_layer_.c_str(), elevation_layer_.c_str());
      return out;
    }

    if (!gridMap_.isValid(start_idx) || !gridMap_.isValid(goal_idx)) { ROS_WARN("start/goal invalid cells"); return out; }

    double start_elev = gridMap_.at(elevation_layer_, start_idx);
    double goal_elev = gridMap_.at(elevation_layer_, goal_idx);
    double start_trav = gridMap_.at(trav_layer_, start_idx);
    double goal_trav = gridMap_.at(trav_layer_, goal_idx);

    if (start_trav < trav_threshold_) { ROS_WARN("Start cell not traversable (trav=%f)", start_trav); return out; }
    if (goal_trav < trav_threshold_) { ROS_WARN("Goal cell not traversable (trav=%f)", goal_trav); return out; }

    // A* containers
    typedef std::priority_queue<Node, std::vector<Node>, CompareF> PriorityQueue;
    PriorityQueue open;
    std::unordered_map<long long, Node> allNodes; // key->Node
    std::unordered_map<long long, bool> closed;

    // heuristic: Euclidean in 3D
    auto heuristic = [&](const Eigen::Array2i &a, const Eigen::Array2i &b)->double {
      Position pa, pb; double za, zb;
      gridMap_.getPosition(a, pa); za = gridMap_.at(elevation_layer_, a);
      gridMap_.getPosition(b, pb); zb = gridMap_.at(elevation_layer_, b);
      double dx = pa.x() - pb.x();
      double dy = pa.y() - pb.y();
      double dz = za - zb;
      return std::sqrt(dx*dx + dy*dy + dz*dz);
    };

    // init
    Node s; s.idx = start_idx; s.g = 0.0; s.f = heuristic(start_idx, goal_idx); s.has_parent = false;
    long long sk = idxToKey(start_idx);
    open.push(s); allNodes[sk] = s;

    bool found = false; Node current;
    const double max_slope_tan = std::tan(max_slope_deg_ * M_PI / 180.0);

    // main loop
    while (!open.empty()) {
      current = open.top(); open.pop();
      long long cur_key = idxToKey(current.idx);
      if (closed[cur_key]) continue;
      closed[cur_key] = true;

      if ((current.idx == goal_idx).all()) { found = true; break; }

      // explore neighbors using distance-sorted offsets (allows skipping untraversable immediate cells)
      Position cur_xy; gridMap_.getPosition(current.idx, cur_xy);
      double cur_z = gridMap_.at(elevation_layer_, current.idx);
      double cur_trav = gridMap_.at(trav_layer_, current.idx);

      // ensure neighbor_offsets_ is computed
      if (neighbor_offsets_.empty()) precomputeNeighborOffsets();

      for (const Eigen::Array2i &off : neighbor_offsets_) {
        Eigen::Array2i nidx = current.idx + off;
        // if (!gridMap_.isInside(nidx)) continue;
        if ((nidx < 0).any() || (nidx >= gridMap_.getSize()).any()) continue;
        if (!gridMap_.isValid(nidx)) continue;

        // neighbor's top surface traversability and elevation
        double neigh_trav = gridMap_.at(trav_layer_, nidx);
        double neigh_z = gridMap_.at(elevation_layer_, nidx);

        // skip cells that are not treadable
        if (neigh_trav < trav_threshold_) continue;

        // vertical constraints (step up / drop down) - allow stepping over small untraversable risers
        double dz = neigh_z - cur_z;
        if (dz > max_step_up_) continue;            // too tall to climb up
        if (dz < -max_drop_down_) continue;         // too large drop down

        // slope constraint (check slope between cell centers)
        Eigen::Vector2d neigh_xy; gridMap_.getPosition(nidx, neigh_xy);
        double horiz = std::hypot(neigh_xy.x() - cur_xy.x(), neigh_xy.y() - cur_xy.y());
        double slope_tan = std::abs(dz) / (horiz + 1e-6);
        if (slope_tan > max_slope_tan) continue;

        // compute move cost: horizontal distance weighted by traversability and climb penalty
        double avg_trav = 0.5 * (cur_trav + neigh_trav);
        double horiz_cost = horiz * (1.0 + (1.0 - avg_trav) * cost_scale_);
        double climb_cost = (dz > 0.0) ? dz * vertical_cost_weight_ : 0.0; // penalize climbs
        // additional penalty for larger jumps (because off may be >1 cell)
        double jump_penalty = (std::hypot(off.x() * gridMap_.getResolution(), off.y() * gridMap_.getResolution()) - gridMap_.getResolution()) * 2.0;
        if (jump_penalty < 0.0) jump_penalty = 0.0;
        double move_cost = horiz_cost + climb_cost + jump_penalty;

        double g_new = current.g + move_cost;
        double h = heuristic(nidx, goal_idx);
        double f_new = g_new + h;

        long long nkey = idxToKey(nidx);
        if (!allNodes.count(nkey) || g_new < allNodes[nkey].g) {
          Node node; node.idx = nidx; node.g = g_new; node.f = f_new; node.parent = current.idx; node.has_parent = true;
          allNodes[nkey] = node;
          open.push(node);
        }
      }
    }

    if (!found) return out;

    // reconstruct path
    std::vector<Eigen::Array2i> rev;
    Eigen::Array2i cur_idx = current.idx;
    while (true) {
      rev.push_back(cur_idx);
      if ((cur_idx == start_idx).all()) break;
      Node &n = allNodes[idxToKey(cur_idx)];
      if (!n.has_parent) break; // safety
      cur_idx = n.parent;
    }

    std::reverse(rev.begin(), rev.end());

    // populate nav_msgs::Path with z from elevation layer
    for (const Eigen::Array2i &i : rev) {
      Eigen::Vector2d p; gridMap_.getPosition(i, p);
      double z = gridMap_.at(elevation_layer_, i);
      geometry_msgs::PoseStamped ps;
      ps.header = out.header;
      ps.header.stamp = ros::Time::now();
      ps.pose.position.x = p.x();
      ps.pose.position.y = p.y();
      ps.pose.position.z = z;
      ps.pose.orientation.w = 1.0;
      out.poses.push_back(ps);
    }

    return out;
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "astar_2p5d_planner");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  AStar2p5DPlanner planner(nh, pnh);
  ros::spin();
  return 0;
}

/*
Parameters (private):
  ~grid_map_topic (string)    default "grid_map"
  ~elevation_layer (string)   default "elevation"
  ~traversability_layer       default "traversability"
  ~path_topic                 default "/planned_path"
  ~map_frame                  default "odom"

  ~traversability_threshold   default 0.2  -- cell must be >= this value to be considered a tread
  ~neighbor_radius            default 0.5  -- meters reachable from a cell (search radius)
  ~max_step_up                default 0.25 -- meters the robot can climb up
  ~max_drop_down              default 0.5  -- max safe drop
  ~max_slope_deg              default 45.0 -- max slope between adjacent cells
  ~cost_scale                 default 100.0 -- how strongly low traversability increases cost
  ~vertical_cost_weight       default 5.0  -- extra cost per meter climbed
  ~allow_diagonal             default true

Notes:
 - This planner treats each grid cell with traversability >= threshold as a node.
 - Vertical transitions are allowed if elevation difference between neighbouring cells
   is within ~max_step_up (for up) and not lower than max_drop_down (for down).
 - Instead of only 4/8 neighbors, the planner considers all cells within ~neighbor_radius (sorted by distance),
   allowing it to "reach" the next stair tread even if intervening risers are untraversable.
 - Heuristic used is Euclidean 3D distance (x,y,z).
 - The generated nav_msgs/Path includes z positions (useful for visualization or 3D-following controllers).
 - This is a 2.5D approach: it keeps a 2D grid of nodes but reasons about elevation for costs and feasibility.
*/
