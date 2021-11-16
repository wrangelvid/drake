//
// Created by amice on 11/8/21.
//
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#include "drake/bindings/pydrake/common/default_scalars_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"

#include "drake/bindings/pydrake/common/value_pybind.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"



namespace drake {
namespace pydrake {

template <typename T>
void DoPoseDeclaration(py::module m, T)    {
    using namespace drake::multibody;
    py::tuple param = GetPyParam<T>();
    using Class = RationalForwardKinematics::Pose<T>;
    auto cls = DefineTemplateClassWithDefault<Class>(m, "RationalForwardKinematicsPose", param);
//    py::class_<Class>(m, "Pose")
    cls
        .def("translation", [](const Class& self) {return self.p_AB;})
        .def("rotation", [](const Class& self) {return self.R_AB;})
//        .def_readwrite("p_AB", &RationalForwardKinematics::Pose<T>::p_AB)
//        .def_readwrite("R_AB", &RationalForwardKinematics::Pose<T>::R_AB)
        .def_readwrite(
            "frame_A_index", &RationalForwardKinematics::Pose<T>::frame_A_index)
        .def("asRigidTransformExpr", [](const Class& self) {return self.asRigidTransformExpression();});


    DefCopyAndDeepCopy(&cls);
//    AddValueInstantiation<Class>(m);
//    AddValueInstantiation<std::vector<Class>>(m);
//    AddValueInstantiation<RationalForwardKinematics::Pose>(m);
  }

PYBIND11_MODULE(rational_forward_kinematics, m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::multibody;
//  constexpr auto& doc = pydrake_doc.drake.multibody;
  m.doc() = "RationalForwardKinematics module";

  py::module::import("pydrake.math");
  py::module::import("pydrake.multibody.plant");
  //RationalForwardKinematics Class
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
                     );
  }//RationalForwardKinematics Class
  //RationalForwardKinematics Util methods
    {
//      constexpr auto& cls_doc = doc.rational_forward_kinematics.generate_monomial_basis;
        m.def("GenerateMonomialBasisWithOrderUpToOne", &GenerateMonomialBasisWithOrderUpToOne, py::arg("t_angles"));
        m.def("GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo",
              &GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo, py::arg("t_angles"));
    }//RationalForwardKinematics Util methods

//  type_visit([m](auto dummy) { DoPoseDeclaration(m, dummy); },
//      CommonScalarPack{});
  type_pack<symbolic::Polynomial, symbolic::RationalFunction> sym_pack;
  type_visit([m](auto dummy) { DoPoseDeclaration(m, dummy); },
      sym_pack);
}
//Pose

}//pydrake
}//drake