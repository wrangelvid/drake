#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/rational_function_types_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"

namespace drake {
namespace pydrake {

PYBIND11_MODULE(rational_function, m) {
  {
    // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
    using namespace drake::symbolic;
    constexpr auto& cls_doc = pydrake_doc.drake.symbolic.RationalFunction;
    py::module::import("pydrake.common");

    using Class = RationalFunction;

    py::class_<Class>(m, "RationalFunction", cls_doc.doc)
        .def(py::init<>(), cls_doc.ctor.doc_0args);
        // class doc is probably wrong
//        .def(py::init<const symbolic::Polynomial, const symbolic::Polynomial>(), cls_doc.ctor.doc_0args)
//        .def(py::init<const symbolic::Polynomial>, cls_doc.ctor.doc_0args);
//        .def("GetNumberOfCoefficients", &Class::GetNumberOfCoefficients,
//            cls_doc.GetNumberOfCoefficients.doc)
//        .def("GetDegree", &Class::GetDegree, cls_doc.GetDegree.doc)
//        .def("IsAffine", &Class::IsAffine, cls_doc.IsAffine.doc)
//        .def("GetCoefficients", &Class::GetCoefficients,
//            cls_doc.GetCoefficients.doc)
//        .def("Derivative", &Class::Derivative, py::arg("derivative_order") = 1,
//            cls_doc.Derivative.doc)
//        .def("Integral", &Class::Integral,
//            py::arg("integration_constant") = 0.0, cls_doc.Integral.doc)
//        .def("CoefficientsAlmostEqual", &Class::CoefficientsAlmostEqual,
//            py::arg("other"), py::arg("tol") = 0.0,
//            py::arg("tol_type") = ToleranceType::kAbsolute,
//            cls_doc.CoefficientsAlmostEqual.doc)
        // Arithmetic
//        .def(-py::self)
//        .def(py::self + py::self)
//        .def(py::self + double())
//        .def(double() + py::self)
//        .def(py::self - py::self)
//        .def(py::self - double())
//        .def(double() - py::self)
//        .def(py::self * py::self)
//        .def(py::self * double())
//        .def(double() * py::self)
//        .def(py::self / double())
//        // Logical comparison
//        .def(py::self == py::self);

  }
}

}  // namespace pydrake
}  // namespace drake
