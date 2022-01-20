import pydrake.symbolic as sym
from pydrake.symbolic import RationalFunction
import numbers
import numpy as np


class NotRationalFunctionException(Exception):
    pass


def xreplace(expr, rules):
    if isinstance(expr, float) or isinstance(expr, sym.Variable):
        return expr
    assert isinstance(expr, sym.Expression), expr
    for old, new in rules:
        if expr.EqualTo(old):
            return new
    ctor, old_args = expr.Deconstruct()
    new_args = [xreplace(e, rules) for e in old_args]
    return ctor(*new_args)


def generate_rationalize_trig_expr_rules(q_var, t_var):
    rules = []
    for i in range(t_var.shape[0]):
        sin_rule = (sym.sin(q_var[i]), (2 * t_var[i]) / (1 + t_var[i] ** 2))
        cos_rule = (sym.cos(q_var[i]), (1 - t_var[i] ** 2) / (1 + t_var[i] ** 2))
        rules += [sin_rule, cos_rule]
    return rules


def rationalize_trig_expr(expr, rules):
    return make_rational_function_from_expression(xreplace(expr, rules))


def RationalFunctionFromExpression(expr):
    # TODO handle nested fractions
    LegalPolyExpressionKind = [
        sym.ExpressionKind.Var,
        sym.ExpressionKind.Add,
        sym.ExpressionKind.Mul,
        sym.ExpressionKind.Div,
        sym.ExpressionKind.Pow,
        sym.ExpressionKind.Constant
    ]
    if isinstance(expr, (numbers.Number, np.number)):
        return sym.Polynomial(expr)

    expr_kind = expr.get_kind()

    if expr.is_polynomial():
        return sym.RationalFunction(sym.Polynomial(expr))
    elif expr_kind not in LegalPolyExpressionKind:
        raise NotRationalFunctionException(expr.to_string() + " is not rational")
    elif expr_kind == sym.ExpressionKind.Div:
        (ctor, (numerator, denominator)) = expr.Deconstruct()
        numerator = sym.Polynomial(numerator)
        denominator = sym.Polynomial(denominator)
        return RationalFunction(numerator, denominator)
    elif expr_kind in LegalPolyExpressionKind:
        (ctor, args) = expr.Deconstruct()
        if expr_kind == sym.ExpressionKind.Mul or expr_kind == sym.ExpressionKind.Pow:
            res = RationalFunction(1)
        else:
            res = RationalFunction(0)
        for e in args:
            res = ctor(res, RationalFunctionFromExpression(e))
            print("hit this")

        return res

    else:
        raise NotRationalFunctionException(expr.to_string() + " is not rational but of type " + expr.get_kind())
