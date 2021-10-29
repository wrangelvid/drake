#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <common_robotics_utilities/simple_rrt_planner.hpp>
#include <common_robotics_utilities/simple_prm_planner.hpp>
#include <common_robotics_utilities/simple_graph.hpp>

#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"

using Eigen::VectorXd;

namespace drake {
namespace pydrake {

PYBIND11_MODULE(common_robotics_utilities, m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace common_robotics_utilities;

  py::module::import("pydrake.common");

  {
    using Class = simple_rrt_planner::SimpleRRTPlannerState<VectorXd>;
    py::class_<Class>(m, "SimpleRRTPlannerState", "")
        .def(py::init<>(), "")
        .def(py::init<const VectorXd&>(), py::arg("value"), "")
        .def(py::init<const VectorXd&, const int64_t>(), py::arg("value"),
            py::arg("parent_index"), "")
        .def(py::init<const VectorXd&, const int64_t,
                 const std::vector<int64_t>&>(),
            py::arg("value"), py::arg("parent_index"), py::arg("child_indices"),
            "")
        .def("GetParentIndex", &Class::GetParentIndex, "");
  }

  {
    using Class = simple_rrt_planner::PropagatedState<VectorXd>;
    py::class_<Class>(m, "PropagatedState", "")
        .def(py::init<const VectorXd&, const int64_t>(), py::arg("state"),
            py::arg("relative_parent_index"), "")
        .def("State", &Class::State, "");
  }

  {
    using Class = simple_rrt_planner::MultipleSolutionPlanningResults<VectorXd>;
    py::class_<Class>(m, "MultipleSolutionPlanningResults", "")
        .def("Paths", &Class::Paths, "")
        .def("Statistics", &Class::Statistics, "");
  }

  {
    using Class = simple_rrt_planner::SingleSolutionPlanningResults<VectorXd>;
    py::class_<Class>(m, "SingleSolutionPlanningResults", "")
        .def("Path", &Class::Path, "")
        .def("Statistics", &Class::Statistics, "");
  }

  m.def("MakeKinematicLinearRRTNearestNeighborsFunction",
      &simple_rrt_planner::MakeKinematicLinearRRTNearestNeighborsFunction<VectorXd>,
      py::arg("distance_fn"), py::arg("use_parallel") = true, "");

  m.def("MakeRRTTimeoutTerminationFunction", &simple_rrt_planner::MakeRRTTimeoutTerminationFunction,
      py::arg("planning_timeout"), "");

  m.def("RRTPlanMultiPath", &simple_rrt_planner::RRTPlanMultiPath<VectorXd>, py::arg("tree"),
      py::arg("sampling_fn"), py::arg("nearest_neighbor_fn"),
      py::arg("forward_propagation_fn"), py::arg("state_added_callback_fn"),
      py::arg("check_goal_reached_fn"), py::arg("goal_reached_callback_fn"),
      py::arg("termination_check_fn"), "");

  m.def("RRTPlanSinglePath", &simple_rrt_planner::RRTPlanSinglePath<VectorXd>, py::arg("tree"),
      py::arg("sampling_fn"), py::arg("nearest_neighbor_fn"),
      py::arg("forward_propagation_fn"), py::arg("state_added_callback_fn"),
      py::arg("check_goal_reached_fn"), py::arg("goal_reached_callback_fn"),
      py::arg("termination_check_fn"), "");

  //Simple graph for PRM

  {
    using Class = simple_graph::GraphNode<VectorXd>;
    py::class_<Class>(m, "GraphNode", "")
        .def(py::init<>(), "")
        .def(py::init<const VectorXd&, std::vector<VectorXd&>, std::vector<VectorXd&>>(),
            py::arg("value"), py::arg("new_in_edges"), py::arg("new_out_edges"), "");
  }

  {
    using Class = simple_graph::GraphEdge;
    py::class_<Class>(m, "GraphEdge", "")
        .def(py::init<>(), "")
        .def(py::init<const int64_t, const int64_t, const double, const uint64_t>(),
            py::arg("from_index"), py::arg("to_index"), py::arg("weight"), py::arg("scratchpad"), "")
        .def(py::init<const int64_t, const int64_t, const double, const uint64_t>(),
            py::arg("from_index"), py::arg("to_index"), py::arg("weight"),  "");
  }

  {
    using Class = simple_graph::Graph<VectorXd>;
    py::class_<Class>(m, "Graph", "")
        .def(py::init<>(), "");
  }


  //PRM 
  m.def("AddNodeToRoadmap", &simple_prm_planner::AddNodeToRoadmap<VectorXd>, py::arg("state"),
      py::arg("nn_distance_direction"), py::arg("roadmap"), py::arg("distance_fn"),
      py::arg("edge_validity_check_fn"), py::arg("K"), py::arg("use_parallel"),
      py::arg("distance_is_symmetric"), py::arg("add_duplicate_states"), "");

  m.def("GrowRoadMap", &simple_prm_planner::GrowRoadMap<VectorXd>, py::arg("roadmap"),
      py::arg("sampling_fn"), py::arg("distance_fn"), py::arg("state_validity_check_fn"),
      py::arg("edge_validity_check_fn"), py::arg("termination_check_fn"), py::arg("K"),
      py::arg("use_parallel"), py::arg("distance_is_symmetric"), py::arg("add_duplicate_states"), "");

  m.def("UpdateRoadMapEdges", &simple_prm_planner::UpdateRoadMapEdges<VectorXd>, py::arg("roadmap"),
      py::arg("edge_validity_check_fn"), py::arg("distance_fn"), py::arg("use_parallel"), "");

  m.def("ExtractSolution", &simple_prm_planner::ExtractSolution<VectorXd>,
      py::arg("roadmap"), py::arg("astar_index_solution"), "");

  m.def("LazyQueryPathAndAddNodes", &simple_prm_planner::LazyQueryPathAndAddNodes<VectorXd>,
      py::arg("starts"), py::arg("goals"), py::arg("roadmap"), py::arg("distance_fn"),
      py::arg("edge_validity_check_fn"), py::arg("K"), py::arg("use_parallel"),
      py::arg("distance_is_symmetric"), py::arg("add_duplicate_states"),
      py::arg("limit_astar_pqueue_duplicates"), "");

  m.def("QueryPathAndAddNodes", &simple_prm_planner::QueryPathAndAddNodes<VectorXd>,
      py::arg("starts"), py::arg("goals"), py::arg("roadmap"), py::arg("distance_fn"),
      py::arg("edge_validity_check_fn"), py::arg("K"), py::arg("use_parallel"),
      py::arg("distance_is_symmetric"), py::arg("add_duplicate_states"),
      py::arg("limit_astar_pqueue_duplicates"), "");

  m.def("LazyQueryPath", &simple_prm_planner::LazyQueryPath<VectorXd>,
      py::arg("starts"), py::arg("goals"), py::arg("roadmap"), py::arg("distance_fn"),
      py::arg("edge_validity_check_fn"), py::arg("K"), py::arg("use_parallel"),
      py::arg("distance_is_symmetric"), py::arg("add_duplicate_states"),
      py::arg("limit_astar_pqueue_duplicates"), "");

  m.def("QueryPath", &simple_prm_planner::QueryPath<VectorXd>, py::arg("starts"),
      py::arg("goals"), py::arg("roadmap"), py::arg("distance_fn"),
      py::arg("edge_validity_check_fn"), py::arg("K"), py::arg("use_parallel"),
      py::arg("distance_is_symmetric"), py::arg("add_duplicate_states"),
      py::arg("limit_astar_pqueue_duplicates"), "");

}

}  // namespace pydrake
}  // namespace drake
