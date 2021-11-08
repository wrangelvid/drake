//
// Created by amice on 11/8/21.
//
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#include "drake/bindings/pydrake/common/default_scalars_pybind.h"
#include "drake/bindings/pydrake/common/sorted_pair_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"

#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
//#include "drake/multibody/inverse_kinematics/unit_quaternion_constraint.h"

namespace drake {
    namespace pydrake {

PYBIND11_MODULE(inverse_kinematics, m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::multibody;
//  constexpr auto& doc = pydrake_doc.drake.multibody;
  m.doc() = "RationalForwardKinematics module";

  py::module::import("pydrake.math");
  py::module::import("pydrake.multibody.plant");
  {
    using Class = RationalForwardKinematics;
    // no class docs built
//    constexpr auto& cls_doc = doc.RationalForwardKinematics;
    py::class_<Class>(m, "RationalForwardKinematics")
        .def(py::init<const MultibodyPlant<double>&>(), py::arg("plant"),
//              Keep alive, reference: `self` keeps `plant` alive.
            py::keep_alive<1, 2>()  // BR
                    )
        .def("CalcLinkPoses", &Class::CalcLinkPoses,
             py::arg("q_star"), py::arg("expressed_body_index")
//             cls_doc.CalcLinkPoses
             )
        .def("CalcLinkPosesAsMultilinearPolynomials", &Class::CalcLinkPosesAsMultilinearPolynomials,
                     py::arg("q_star"), py::arg("expressed_body_index")
        //             cls_doc.CalcLinkPoses
                     )
        .def("ConvertMultilinearPolynomialToRationalFunction", &Class::ConvertMultilinearPolynomialToRationalFunction,
                     py::arg("e")
                     //             cls_doc.CalcLinkPoses
                     )
        .def("plant", &Class::plant
                     //             cls_doc.CalcLinkPoses
                     )
        .def("t", &Class::t
                     //             cls_doc.CalcLinkPoses
                     )
        .def("ComputeTValue", &Class::ComputeTValue,
                     py::arg("q_val"),
                     py::arg("q_star_val"),
                     py::arg("clamp_angle") = false
                     //             cls_doc.CalcLinkPoses
                     )
        .def("FindTOnPath", &Class::FindTOnPath,
                     py::arg("start"),
                     py::arg("end")
                     //             cls_doc.CalcLinkPoses
                     )

        ;
  }
}
    }//pydrake
}//drake