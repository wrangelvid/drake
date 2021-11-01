#include "drake/bindings/pydrake/symbolic_py_deconstruct.h"

#include "fmt/format.h"
#include "fmt/ostream.h"
#include "pybind11/stl.h"

namespace drake {
namespace pydrake {
namespace internal {
namespace {

using drake::symbolic::Expression;
using drake::symbolic::ExpressionKind;

py::object DeconstructConstructor(py::module m, const symbolic::Expression& e) {
  switch (e.get_kind()) {
    case ExpressionKind::Abs:
      return m.attr("abs");
    case ExpressionKind::Acos:
      return m.attr("acos");
    case ExpressionKind::Add:
      return m.attr("_reduce_add");
    case ExpressionKind::Asin:
      return m.attr("asin");
    case ExpressionKind::Atan2:
      return m.attr("atan2");
    case ExpressionKind::Atan:
      return m.attr("atan");
    case ExpressionKind::Ceil:
      return m.attr("ceil");
    case ExpressionKind::Constant:
      return m.attr("Expression");
    case ExpressionKind::Cos:
      return m.attr("cos");
    case ExpressionKind::Cosh:
      return m.attr("cosh");
    case ExpressionKind::Div:
      return m.attr("operator").attr("truediv");
    case ExpressionKind::Exp:
      return m.attr("exp");
    case ExpressionKind::Floor:
      return m.attr("floor");
    case ExpressionKind::IfThenElse:
      return m.attr("if_then_else");
    case ExpressionKind::Log:
      return m.attr("log");
    case ExpressionKind::Max:
      return m.attr("max");
    case ExpressionKind::Min:
      return m.attr("min");
    case ExpressionKind::Mul:
      return m.attr("_reduce_mul");
    case ExpressionKind::NaN:
      return m.attr("Expression");
    case ExpressionKind::Pow:
      return m.attr("pow");
    case ExpressionKind::Sin:
      return m.attr("sin");
    case ExpressionKind::Sinh:
      return m.attr("sinh");
    case ExpressionKind::Sqrt:
      return m.attr("sqrt");
    case ExpressionKind::Tan:
      return m.attr("tan");
    case ExpressionKind::Tanh:
      return m.attr("tanh");
    case ExpressionKind::UninterpretedFunction:
      return m.attr("uninterpreted_function");
    case ExpressionKind::Var:
      return m.attr("Expression");
  }
  DRAKE_UNREACHABLE();
}

py::list DeconstructArgs(const symbolic::Expression& e) {
  py::list result;
  switch (e.get_kind()) {
    // The only non-Expression args to extract are constants and variables.
    case ExpressionKind::Constant: {
      result.append(get_constant_value(e));
      break;
    }
    case ExpressionKind::NaN: {
      result.append(NAN);
      break;
    }
    case ExpressionKind::Var: {
      result.append(get_variable(e));
      break;
    }
    // These are all UnaryExpressionCell.
    case ExpressionKind::Abs:
    case ExpressionKind::Acos:
    case ExpressionKind::Asin:
    case ExpressionKind::Atan:
    case ExpressionKind::Ceil:
    case ExpressionKind::Cos:
    case ExpressionKind::Cosh:
    case ExpressionKind::Exp:
    case ExpressionKind::Floor:
    case ExpressionKind::Log:
    case ExpressionKind::Sin:
    case ExpressionKind::Sinh:
    case ExpressionKind::Sqrt:
    case ExpressionKind::Tan:
    case ExpressionKind::Tanh: {
      result.append(get_argument(e));
      break;
    }
    // These are all BinaryExpressionCell.
    case ExpressionKind::Div:
    case ExpressionKind::Atan2:
    case ExpressionKind::Min:
    case ExpressionKind::Max:
    case ExpressionKind::Pow: {
      result.append(get_first_argument(e));
      result.append(get_second_argument(e));
      break;
    }
    // Add and Mul are reductions over lists of expressions.
    case ExpressionKind::Add: {
      result.append(get_constant_in_addition(e));
      for (const auto& [expr, coeff] : get_expr_to_coeff_map_in_addition(e)) {
        result.append(coeff * expr);
      }
      break;
    }
    case ExpressionKind::Mul: {
      result.append(get_constant_in_multiplication(e));
      for (const auto& [base, exp] :
          get_base_to_exponent_map_in_multiplication(e)) {
        result.append(symbolic::pow(base, exp));
      }
      break;
    }
    // Special forms.
    case ExpressionKind::IfThenElse: {
      result.append(get_conditional_formula(e));
      result.append(get_then_expression(e));
      result.append(get_else_expression(e));
      break;
    }
    case ExpressionKind::UninterpretedFunction: {
      py::list function_args;
      for (const auto& expr : get_uninterpreted_function_arguments(e)) {
        function_args.append(expr);
      }
      result.append(get_uninterpreted_function_name(e));
      result.append(function_args);
      break;
    }
  }
  return result;
}

}  // namespace

py::object Deconstruct(py::module m, const symbolic::Expression& e) {
  return py::make_tuple(DeconstructConstructor(m, e), DeconstructArgs(e));
}

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
