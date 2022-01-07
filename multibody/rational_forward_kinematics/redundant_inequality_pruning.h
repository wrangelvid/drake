//#pragma once
//
//#include <map>
//#include <string>
//#include <unordered_map>
//#include <unordered_set>
//#include <vector>
//
//#include "drake/solvers/mathematical_program.h"
//#include "drake/solvers/mathematical_program_result.h"
//
//namespace drake{
//namespace multibody{
//
///**
// * Given the C-space free region candidate C*t<=d,
// * find all the inequalities which are redundant by solving
// * a series of linear programs.
// * return the indices of the redundant inequalites
// */
//VectorX<int> FindRedundantInequalitiesInHPolyhedronByIndex(
//    const Eigen::Ref<const Eigen::MatrixXd>& C,
//    const Eigen::Ref<const Eigen::VectorXd>& d);
//
//} // namespace multibody
//} // namespace drake
//}