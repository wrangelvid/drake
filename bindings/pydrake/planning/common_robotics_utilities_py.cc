#include "pybind11/eigen.h"
#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <common_robotics_utilities/simple_rrt_planner.hpp>

#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"

namespace drake {
namespace pydrake {

PYBIND11_MODULE(common_robotics_utilities, m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace common_robotics_utilities::simple_rrt_planner;

  using T = Eigen::VectorXd;

  py::module::import("pydrake.common");

  {
    using Class = SimpleRRTPlannerState<T>;
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
    using Class = PropagatedState<T>;
    py::class_<Class>(m, "PropagatedState", "")
        .def(py::init<const T&, const int64_t>(), py::arg("state"),
            py::arg("relative_parent_index"), "")
        .def("State", &Class::State, "");
  }

  {
    using Class = MultipleSolutionPlanningResults<T>;
    py::class_<Class>(m, "MultipleSolutionPlanningResults", "")
        .def("Paths", &Class::Paths, "")
        .def("Statistics", &Class::Statistics, "");
  }

  {
    using Class = SingleSolutionPlanningResults<T>;
    py::class_<Class>(m, "SingleSolutionPlanningResults", "")
        .def("Path", &Class::Path, "")
        .def("Statistics", &Class::Statistics, "");
  }

  m.def("MakeKinematicLinearRRTNearestNeighborsFunction",
      &MakeKinematicLinearRRTNearestNeighborsFunction<T>,
      py::arg("distance_fn"), py::arg("use_parallel") = true, "");

  m.def("MakeRRTTimeoutTerminationFunction", &MakeRRTTimeoutTerminationFunction,
      py::arg("planning_timeout"), "");

  m.def("RRTPlanMultiPath", &RRTPlanMultiPath<T>, py::arg("tree"),
      py::arg("sampling_fn"), py::arg("nearest_neighbor_fn"),
      py::arg("forward_propagation_fn"), py::arg("state_added_callback_fn"),
      py::arg("check_goal_reached_fn"), py::arg("goal_reached_callback_fn"),
      py::arg("termination_check_fn"), "");

  m.def("RRTPlanSinglePath", &RRTPlanSinglePath<T>, py::arg("tree"),
      py::arg("sampling_fn"), py::arg("nearest_neighbor_fn"),
      py::arg("forward_propagation_fn"), py::arg("state_added_callback_fn"),
      py::arg("check_goal_reached_fn"), py::arg("goal_reached_callback_fn"),
      py::arg("termination_check_fn"), "");
}

}  // namespace pydrake
}  // namespace drake
