//
// Created by amice on 11/4/21.
//

#pragma once

#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/common/symbolic.h"

// The macro `PYBIND11_NUMPY_OBJECT_DTYPE` places symbols into the namespace
// `pybind11::detail`, so we should not place these in `drake::pydrake`.

PYBIND11_NUMPY_OBJECT_DTYPE(drake::symbolic::RationalFunction);
