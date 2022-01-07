import numpy as np
from pydrake.all import MosekSolver, MathematicalProgram, HPolyhedron, SolverResult

def prune_region(region):
    num_faces = region.A().shape[0]
    solver = MosekSolver()
    kept_indices = np.arange(num_faces).tolist()
    excluded_indices = []

    for excluded_index in range(num_faces):
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(region.A().shape[1], 'x')


        cur_kept_indices = kept_indices.copy()
        cur_kept_indices.remove(excluded_index)

        c_to_check_redundant = region.A()[excluded_index, :][np.newaxis, :]
        d_to_check_redundant = np.atleast_2d(region.b()).T[excluded_index, :][np.newaxis, :]

        A_to_check_against = region.A()[cur_kept_indices, :]
        b_to_check_against = region.b()[cur_kept_indices]

        prog.AddLinearCost(-c_to_check_redundant.T, 0., x)
        prog.AddLinearConstraint(A_to_check_against, -np.inf * np.ones_like(b_to_check_against),
                                 b_to_check_against, x)
        prog.AddLinearConstraint(c_to_check_redundant, -np.inf * np.ones_like(d_to_check_redundant),
                                 d_to_check_redundant + 1, x)

        solution = solver.Solve(prog)
        if solution.get_solution_result() != SolutionResult.kSolutionFound:
            raise ValueError(f"Solution not found status is {solution.get_solution_result()}. Polytope might be empty")
        opt_val = -solution.get_optimal_cost()
        if opt_val <= d_to_check_redundant:
            excluded_indices.append(excluded_index)
            kept_indices.remove(excluded_index)
    new_region = HPolyhedron(region.A()[kept_indices, :], region.b()[kept_indices])
    return new_region, kept_indices, excluded_indices
