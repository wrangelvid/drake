#include "pybind11/eigen.h"
#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <common_robotics_utilities/simple_graph.hpp>
#include <common_robotics_utilities/simple_prm_planner.hpp>
#include <common_robotics_utilities/simple_rrt_planner.hpp>

#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"

namespace drake {
namespace pydrake {

PYBIND11_MODULE(common_robotics_utilities, m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace common_robotics_utilities;

  using T = Eigen::VectorXd;

  py::module::import("pydrake.common");

  {
    using Class = simple_rrt_planner::SimpleRRTPlannerState<T>;
    py::class_<Class>(m, "SimpleRRTPlannerState", "")
        .def(py::init<>(), "")
        .def(py::init<const T&>(), py::arg("value"), "")
        .def(py::init<const T&, const int64_t>(), py::arg("value"),
            py::arg("parent_index"), "")
        .def(py::init<const T&, const int64_t, const std::vector<int64_t>&>(),
            py::arg("value"), py::arg("parent_index"), py::arg("child_indices"),
            "")
        .def("GetParentIndex", &Class::GetParentIndex, "");
  }

  {
    using Class = simple_rrt_planner::PropagatedState<T>;
    py::class_<Class>(m, "PropagatedState", "")
        .def(py::init<const T&, const int64_t>(), py::arg("state"),
            py::arg("relative_parent_index"), "")
        .def("State", &Class::State, "");
  }

  {
    using Class = simple_rrt_planner::MultipleSolutionPlanningResults<T>;
    py::class_<Class>(m, "MultipleSolutionPlanningResults", "")
        .def("Paths", &Class::Paths, "")
        .def("Statistics", &Class::Statistics, "");
  }

  {
    using Class = simple_rrt_planner::SingleSolutionPlanningResults<T>;
    py::class_<Class>(m, "SingleSolutionPlanningResults", "")
        .def("Path", &Class::Path, "")
        .def("Statistics", &Class::Statistics, "");
  }

  m.def("MakeKinematicLinearRRTNearestNeighborsFunction",
      &simple_rrt_planner::MakeKinematicLinearRRTNearestNeighborsFunction<T>,
      py::arg("distance_fn"), py::arg("use_parallel") = true, "");

  m.def("MakeRRTTimeoutTerminationFunction",
      &simple_rrt_planner::MakeRRTTimeoutTerminationFunction,
      py::arg("planning_timeout"), "");

  m.def("RRTPlanMultiPath", &simple_rrt_planner::RRTPlanMultiPath<T>,
      py::arg("tree"), py::arg("sampling_fn"), py::arg("nearest_neighbor_fn"),
      py::arg("forward_propagation_fn"), py::arg("state_added_callback_fn"),
      py::arg("check_goal_reached_fn"), py::arg("goal_reached_callback_fn"),
      py::arg("termination_check_fn"), "");

  m.def("RRTPlanSinglePath", &simple_rrt_planner::RRTPlanSinglePath<T>,
      py::arg("tree"), py::arg("sampling_fn"), py::arg("nearest_neighbor_fn"),
      py::arg("forward_propagation_fn"), py::arg("state_added_callback_fn"),
      py::arg("check_goal_reached_fn"), py::arg("goal_reached_callback_fn"),
      py::arg("termination_check_fn"), "");

  // Simple graph for PRM
  {
    using Class = simple_graph::GraphEdge;
    py::class_<Class>(m, "GraphEdge", "")
        .def(py::init<>(), "")
        .def(py::init<const int64_t, int64_t, double>(), py::arg("from_index"),
            py::arg("to_index"), py::arg("weight"), "")
        .def(py::init<const int64_t, int64_t, double, uint64_t>(),
            py::arg("from_index"), py::arg("to_index"), py::arg("weight"),
            py::arg("scratchpad"), "");
  }
  {
    using Class = simple_graph::GraphNode<T>;
    py::class_<Class>(m, "GraphNode", "")
        .def(py::init<>(), "")
        .def(py::init<const T&, std::vector<simple_graph::GraphEdge>&,
                 std::vector<simple_graph::GraphEdge>&>(),
            py::arg("value"), py::arg("new_in_edges"), py::arg("new_out_edges"),
            "");
  }

  {
    using Class = simple_graph::Graph<T>;
    py::class_<Class>(m, "Graph", "")
        .def(py::init<>(), "")
        .def(py::init<const size_t>(), py::arg("expected_size"), "")
        .def(py::init<const std::vector<simple_graph::GraphNode<T>,
                 Eigen::aligned_allocator<simple_graph::GraphNode<T>>>&>(),
            py::arg("nodes"), "")
        .def("MakePrunedCopy", &Class::MakePrunedCopy,
            py::arg("nodes_to_prune"), py::arg("use_parallel"), "")
        .def("CheckGraphLinkage",
            overload_cast_explicit<bool>(&Class::CheckGraphLinkage), "")
        .def("GetNodesImmutable", &Class::GetNodesImmutable, "");
  }

  // PRM
  m.def("AddNodeToRoadmap", &simple_prm_planner::AddNodeToRoadmap<T>,
      py::arg("state"), py::arg("nn_distance_direction"), py::arg("roadmap"),
      py::arg("distance_fn"), py::arg("edge_validity_check_fn"), py::arg("K"),
      py::arg("use_parallel") = true, py::arg("distance_is_symmetric") = true,
      py::arg("add_duplicate_states") = false, "");

  m.def("GrowRoadMap", &simple_prm_planner::GrowRoadMap<T>, py::arg("roadmap"),
      py::arg("sampling_fn"), py::arg("distance_fn"),
      py::arg("state_validity_check_fn"), py::arg("edge_validity_check_fn"),
      py::arg("termination_check_fn"), py::arg("K"),
      py::arg("use_parallel") = true, py::arg("distance_is_symmetric") = true,
      py::arg("add_duplicate_states") = false, "");

  m.def("UpdateRoadMapEdges", &simple_prm_planner::UpdateRoadMapEdges<T>,
      py::arg("roadmap"), py::arg("edge_validity_check_fn"),
      py::arg("distance_fn"), py::arg("use_parallel") = true, "");

  m.def("ExtractSolution", &simple_prm_planner::ExtractSolution<T>,
      py::arg("roadmap"), py::arg("astar_index_solution"), "");

  m.def("LazyQueryPathAndAddNodes",
      &simple_prm_planner::LazyQueryPathAndAddNodes<T>, py::arg("starts"),
      py::arg("goals"), py::arg("roadmap"), py::arg("distance_fn"),
      py::arg("edge_validity_check_fn"), py::arg("K"),
      py::arg("use_parallel") = true, py::arg("distance_is_symmetric") = true,
      py::arg("add_duplicate_states") = false,
      py::arg("limit_astar_pqueue_duplicates") = true, "");

  m.def("QueryPathAndAddNodes", &simple_prm_planner::QueryPathAndAddNodes<T>,
      py::arg("starts"), py::arg("goals"), py::arg("roadmap"),
      py::arg("distance_fn"), py::arg("edge_validity_check_fn"), py::arg("K"),
      py::arg("use_parallel") = true, py::arg("distance_is_symmetric") = true,
      py::arg("add_duplicate_states") = false,
      py::arg("limit_astar_pqueue_duplicates") = true, "");

  m.def("LazyQueryPath", &simple_prm_planner::LazyQueryPath<T>,
      py::arg("starts"), py::arg("goals"), py::arg("roadmap"),
      py::arg("distance_fn"), py::arg("edge_validity_check_fn"), py::arg("K"),
      py::arg("use_parallel") = true, py::arg("distance_is_symmetric") = true,
      py::arg("add_duplicate_states") = false,
      py::arg("limit_astar_pqueue_duplicates") = true, "");

  m.def("QueryPath", &simple_prm_planner::QueryPath<T>, py::arg("starts"),
      py::arg("goals"), py::arg("roadmap"), py::arg("distance_fn"),
      py::arg("edge_validity_check_fn"), py::arg("K"),
      py::arg("use_parallel") = true, py::arg("distance_is_symmetric") = true,
      py::arg("add_duplicate_states") = false,
      py::arg("limit_astar_pqueue_duplicates") = true, "");
}

}  // namespace pydrake
}  // namespace drake
