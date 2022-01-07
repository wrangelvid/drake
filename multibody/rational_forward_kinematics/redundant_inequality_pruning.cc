//#include "drake/multibody/rational_forward_kinematics/redundant_inequality_pruning.h"
//
//namespace drake {
//namespace multibody {
////VectorX<int> FindRedundantInequalitiesInHPolyhedronByIndex(
////    const Eigen::Ref<const Eigen::MatrixXd> &C,
////    const Eigen::Ref<const Eigen::VectorXd> &d) {
////
////    const int num_inequalities = C.rows();
////    const int num_vars = C.cols();
////
////    std::list<int> kept_indices(num_vars);
////    std::list<int> excluded_indices(0);
////
////    for (int excluded_index = 0; excluded_index < num_inequalities; excluded_index++){
////      solvers::MathematicalProgram prog;
////      solvers::VectorXDecisionVariable x = prog.NewContinuousVariables(num_vars, "x");
////      std::list<int>cur_kept_indices = kept_indices;
////      cur_kept_indices.remove(excluded_index);
////
//////      c_to_check_redundant = C[excluded_index,:];
//////      d_to_check_redundant = np.atleast_2d(region.b()).T[excluded_index,:][np.newaxis, :]
////
////
////    }
//
//
//
//
////}
//
//
//} // namespace multibody
//} // namespace drake