# # tindar.py
#
# import logging
# from typing import Optional, Tuple
# from simpleor.base import BaseSolver
# from dataclasses import dataclass
# from pulp import *
# import numpy as np
# from pathlib import Path
# import itertools
# import json
# from simpleor.monitoring import get_logger
#
#
# PROJECT_DIR = str(Path(__file__).resolve().parents[1])
# LOGGING_LEVEL = "info"
#
# logger = get_logger(name=f"{__name__}", verbose=LOGGING_LEVEL)
#
#
# @dataclass
# class MatchMaker():  # BaseSolver
#     """Class to solve matchmaker problems
#
#     Args:
#         match_matrix (np.array): square binary matrix indicating which
#             nodes can be matched. Diagonal must be zero.
#     """
#
#     match_matrix: np.array
#
#     def __post_init__(self):
#         self.validate_input(match_matrix=self.match_matrix)
#         self.n = self.match_matrix.shape[0]
#         self._set_variables()
#         self.constraints_all = []
#         self.objective = None
#
#     def _set_variables(self):
#         self.x_names = [f"x_{i}_{j}" for i in range(self.n) for j in range(self.n)]
#         self.x = [LpVariable(name=x_name, cat="Binary") for x_name in self.x_names]
#         self.x_np = np.array(self.x).reshape((self.n, self.n))
#
#     @staticmethod
#     def validate_input(match_matrix: np.array):
#         # type check
#         if not isinstance(match_matrix, np.ndarray):
#             raise ValueError("match_matrix is not a numpy array")
#         # shape check
#         m, n = match_matrix.shape
#         if m != n:
#             raise ValueError(
#                 f"match_matrix is not square: love_matrix.shape"
#                 f"= {match_matrix.shape}"
#             )
#         # diagonal zero check
#         for i in range(n):
#             if match_matrix[i, i] != 0:
#                 raise ValueError("match_matrix diagonal contains nonzeros")
#
#     # Symmetry constraints: if one is paired, the other is paired
#     def _get_symmetry_constraints(self) -> list:
#         tups = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
#
#         # Left-hand side
#         lhs_symmetry = [
#             LpAffineExpression(
#                 [(self.x_np[tup[0], tup[1]], 1), (self.x_np[tup[1], tup[0]], -1)],
#                 name=f"lhs_sym_{tup[0]}_{tup[1]}",
#             )
#             for tup in tups
#         ]
#
#         # Constraints
#         constraints_symmetry = [
#             LpConstraint(
#                 e=lhs_s,
#                 sense=0,
#                 name=f"constraint_sym_{tups[i][0]}_{tups[i][1]}",
#                 rhs=0,
#             )
#             for i, lhs_s in enumerate(lhs_symmetry)
#         ]
#         return constraints_symmetry
#
#     # Feasibility constraints: only pairs if person likes the other
#     def _get_feasibility_constraints(self) -> list:
#         tups = [(i, j) for i in range(self.n) for j in range(self.n)]
#
#         # Left-hand side
#         lhs_like = [
#             LpAffineExpression(
#                 [(self.x_np[tup[0], tup[1]], 1)], name=f"lhs_like_{tup[0]}_{tup[1]}"
#             )
#             for tup in tups
#         ]
#
#         # Constraints
#         constraints_like = [
#             LpConstraint(
#                 e=lhs_l,
#                 sense=-1,
#                 name=f"constraint_like_{tups[i][0]}_{tups[i][1]}",
#                 rhs=self.match_matrix[tups[i][0], tups[i][1]],
#             )
#             for i, lhs_l in enumerate(lhs_like)
#         ]
#         return constraints_like
#
#     # Single assignment: one person can have at most one other person
#     def _get_single_assignment_constraints(self) -> Tuple[list, list]:
#         # Left-hand side: rowsum <= 1
#         lhs_single_rowsum = [
#             LpAffineExpression(
#                 [(self.x_np[i, j], 1) for j in range(self.n)],
#                 name=f"lhs_single_rowsum_{i}",
#             )
#             for i in range(self.n)
#         ]
#
#         # Left-hand side: colsum <= 1
#         lhs_single_colsum = [
#             LpAffineExpression(
#                 [(self.x_np[i, j], 1) for i in range(self.n)],
#                 name=f"lhs_single_colsum_{j}",
#             )
#             for j in range(self.n)
#         ]
#
#         # Constraints
#         constraints_single_rowsum = self.make_single_assignment_constraints(
#             lhs_single=lhs_single_rowsum, kind="rowsum"
#         )
#         constraints_single_colsum = self.make_single_assignment_constraints(
#             lhs_single=lhs_single_colsum, kind="colsum"
#         )
#
#         return constraints_single_rowsum, constraints_single_colsum
#
#     # Auxiliary functions for single assigment constraints
#     @staticmethod
#     def make_single_assignment_constraints(lhs_single: list, kind: str) -> list:
#         constraints_single = [
#             LpConstraint(e=lhs_s, sense=-1, name=f"constraint_single_{kind}_{i}", rhs=1)
#             for i, lhs_s in enumerate(lhs_single)
#         ]
#         return constraints_single
#
#     def _set_constraints(self):
#         constraints_symmetry = self._get_symmetry_constraints()
#         constraints_like = self._get_feasibility_constraints()
#         (
#             constraints_single_rowsum,
#             constraints_single_colsum
#         ) = self._get_single_assignment_constraints()
#
#         self.constraints_all = [
#             *constraints_symmetry,
#             *constraints_like,
#             *constraints_single_rowsum,
#             *constraints_single_colsum,
#         ]
#
#     def _set_objective(self):
#         self.objective = LpAffineExpression([(x_i, 1) for x_i in self.x])
#
#     def set_problem(self):
#         # Initialize constraints and objective
#         if not self.constraints_all:
#             self._set_constraints()
#         if not self.objective:
#             self._set_objective()
#
#         # Create PuLP problem
#         self.prob_pulp = LpProblem("The_Tindar_Problem", LpMaximize)
#         self.prob_pulp += self.objective
#
#         for c in self.constraints_all:
#             self.prob_pulp += c
#
#     def write_problem(self, path=PROJECT_DIR + "/models/Tindar.lp"):
#         self.prob_pulp.writeLP(path)
#
#     def solve(self, kind: str = "pulp"):
#         if kind == "pulp":
#             self.prob_pulp.solve()
#         elif kind == "heuristic":
#             self.solve_heuristic()
#         else:
#             raise ValueError(f"kind {kind} not allowed" "choose from: pulp, heuristic")
#
#     def solve_heuristic(self):
#         self.x_heuristic_np = np.zeros((self.n, self.n))
#
#         for i in range(self.n - 1):
#             if self.x_heuristic_np[i, :].sum() == 0:
#                 done = False
#                 j = i + 1
#
#                 while not done:
#                     mutual_interest = (self.match_matrix[i, j] == 1) and (
#                         self.match_matrix[j, i] == 1
#                     )
#                     available = (self.x_heuristic_np[j, :] == 0).all()
#
#                     if mutual_interest and available:
#                         self.x_heuristic_np[i, j] = 1
#                         self.x_heuristic_np[j, i] = 1
#                         done = True
#
#                     if j == self.n - 1:
#                         done = True
#                     else:
#                         j += 1
#
#     def get_status(self, kind: str = "pulp"):
#         if kind == "pulp":
#             stat = LpStatus[self.prob_pulp.status]
#             return stat
#         elif kind == "heuristic":
#             return "Solved (optimal unsure)"
#         else:
#             raise ValueError(f"kind {kind} not allowed" "choose from: pulp, heuristic")
#
#     def _pulp_solution_to_np(self, pulp_vars=None):
#         if pulp_vars is None:
#             pulp_vars = self.prob_pulp.variables()
#         solution_np = np.array([v.value() for v in pulp_vars]).reshape((self.n, self.n))
#         return solution_np
#
#     def get_solution(self, kind="pulp", verbose=True):
#         if kind == "pulp":
#             vars_pulp = self.prob_pulp.variables()
#             vars_np = self._pulp_solution_to_np(vars_pulp)
#             if verbose:
#                 print(vars_np)
#             return vars_np
#
#         elif kind == "heuristic":
#             if verbose:
#                 print(self.x_heuristic_np)
#             return self.x_heuristic_np
#
#     def get_objective_value(self, kind="pulp", verbose=True):
#         if kind == "pulp":
#             obj = value(self.prob_pulp.objective)
#         elif kind == "heuristic":
#             obj = self.x_heuristic_np.sum()
#
#         if verbose:
#             print(f"Number of lovers connected by {kind} = ", obj)
#         return obj
#
#
# @dataclass
# class MatchMakerProblemGenerator:
#     """Class to generate MatchMaker objects randomly
#
#     Args:
#         n (int): number of people in the model
#         connectedness (1 < int < 10): connectedness of the
#             MatchMaker problem for nodes, implemented as bernouilli
#             probability for edges to be generated
#     """
#
#     n: int
#     connectedness: int = None
#     nan_probability: float = None
#
#     DEFAULT_MIN_CONNECTEDNESS = 1
#     DEFAULT_MAX_CONNECTEDNESS = 10
#     DEFAULT_MIN_EDGE_PROB = 0.05
#     DEFAULT_MAX_EDGE_PROB = 0.75
#     DEFAULT_UNIF_LOW = 0.3
#     DEFAULT_UNIF_HIGH = 0.6
#
#     def __post_init__(self,):
#         self.validate_input(self.n, self.connectedness)
#
#     # Input validation
#     @classmethod
#     def validate_input(self, n, connectedness):
#         # n
#         if not isinstance(n, int):
#             raise ValueError(f"MatchMakerProblemGenerator init error: " f"type(n) = {type(n)}")
#         if n <= 0:
#             raise ValueError(f"MatchMakerProblemGenerator init error: " f"n={n} < 0")
#         # connectedness
#         if not (isinstance(connectedness, (int, float)) or connectedness is None):
#             raise ValueError(
#                 f"MatchMakerProblemGenerator init error: "
#                 f"type(connectedness) = {type(connectedness)}"
#             )
#
#     @classmethod
#     def bernouilli_parameter(self, connectedness):
#         diff_scaled = (connectedness - self.DEFAULT_MIN_CONNECTEDNESS) / self.DEFAULT_MAX_CONNECTEDNESS
#         return (diff_scaled * self.DEFAULT_MAX_EDGE_PROB) + self.DEFAULT_MIN_EDGE_PROB
#
#     # @classmethod
#     # def _create_interesting_love_values(
#     #     self, n, attractiveness_distr=None, unif_low=None, unif_high=None
#     # ):
#     #     # Sample attractiveness levels
#     #     nu = np.random.uniform(low=unif_low, high=unif_high, size=n)
#     #     nu[nu < 0] = 0
#     #     nu[nu > 1] = 1
#     #
#     #     # Calculate corresponding romance levels
#     #     mu = np.array([self.ROMANCE_LEVEL_FN(n) for n in nu])
#     #     mu[mu < 0] = 0
#     #     mu[mu > 1] = 1
#     #
#     #     # Compute love interests
#     #     mu_colvec = mu.reshape((-1, 1))
#     #     nu_rowvec = nu.reshape((1, -1))
#     #
#     #     love_values = np.dot(mu_colvec, nu_rowvec)
#     #
#     #     return love_values
#     #
#     # @staticmethod
#     # def _convert_love_values_to_binary(love_values):
#     #     love_values_scaled = (love_values - love_values.min()) / (
#     #         love_values.max() - love_values.min()
#     #     )
#     #
#     #     love_matrix = love_values_scaled.copy()
#     #     love_matrix[love_matrix > 0.5] = 1
#     #     love_matrix[love_matrix <= 0.5] = 0
#     #
#     #     return love_matrix
#
#     # def generate(
#     #     self,
#     #     nan_probability: Optional[float] = None,
#     #     generation_kind: Optional[str] = None,
#     #     attractiveness_distr: Optional[str] = None,
#     #     unif_low: Optional[float] = None,
#     #     unif_high: Optional[float] = None,
#     # ):
#     #     # Based on bernouilli sampling
#     #     if generation_kind == "simple":
#     #         self.p = self.bernouilli_parameter(connectedness)
#     #         love_matrix = np.random.binomial(1, self.p, size=(n, n)).astype(float)
#     #
#     #     # See notebook 4 for explanation
#     #     elif generation_kind == "interesting":
#     #         if attractiveness_distr == "uniform":
#     #             love_values = self._create_interesting_love_values(
#     #                 n, attractiveness_distr, unif_low, unif_high
#     #             )
#     #         else:
#     #             raise ValueError(
#     #                 f"attractiveness_distr {attractiveness_distr}" " not implemented"
#     #             )
#     #
#     #         # Convert to binary interest
#     #         love_matrix = self._convert_love_values_to_binary(love_values)
#     #
#     #     else:
#     #         raise ValueError(f"kind {generation_kind} not implemented")
#     #
#     #     # Generate missing values
#     #     if nan_probability is not None:
#     #         nan_indicator = np.random.binomial(
#     #             n=1, p=nan_probability, size=love_matrix.shape
#     #         ).astype(bool)
#     #         love_matrix[nan_indicator] = np.nan
#     #
#     #     for i in range(n):
#     #         love_matrix[i, i] = 0
#     #
#     #     if inplace:
#     #         self.love_matrix = love_matrix
#     #     else:
#     #         return love_matrix
#
#
# def tindar_solution(tindar, solver):
#     if solver == "pulp":
#         tindar.set_problem()
#     tindar.solve(kind=solver)
#
#     return (
#         tindar.get_objective_value(kind=solver, verbose=False),
#         tindar.get_solution(kind=solver, verbose=False).tolist(),
#         tindar.get_status(kind=solver),
#     )
#
#
# def tindar_experiment(
#     experiment_id: str = "default_id",
#     n_list: list = [10, 30, 100, 200, 300],
#     connectedness_list: list = [1, 4, 8],
#     solvers: list = ["pulp", "heuristic"],
#     repeat: int = 10,
#     result_directory: str = PROJECT_DIR + "/data",
#     save_problem_and_solution: bool = False,
#     verbose: bool = True,
# ):
#     """Writes results of Tindar experiment to a json file
#
#     Args:
#         experiment_id: str
#         n_list: list of ints
#             how many people in Tindar community
#         connectedness_list: list of ints or floats
#             controlling how many people are interested in each other
#         solvers: list of strings
#             pulp and/or heuristic, which solvers should be used to
#             compute results
#         repeat: int
#             number of times a combination of n - connectedness should be
#             repeated
#         result_directory: str of a path
#         save_problem_and_solution: bool
#             if true, saves the love_matrix and solution matrix
#         verbose: str
#     """
#
#     result_path = f"{result_directory}/results_experiment_{experiment_id}.json"
#
#     parameters = tuple(itertools.product(n_list, connectedness_list))
#
#     tindar_problems_nested = [
#         [MatchMakerProblemGenerator(p[0], p[1]) for _ in range(repeat)] for p in parameters
#     ]
#     tindar_problems = [item for sublist in tindar_problems_nested for item in sublist]
#     tindars = [
#         MatchMaker(tindar_problem=tindar_problem) for tindar_problem in tindar_problems
#     ]
#
#     results = []
#     counter = 1
#     for solver in solvers:
#         for j, tp in enumerate(tindars):
#             if verbose:
#                 print("----------------------------------------------------")
#                 print(
#                     f"Experiment {counter}/{(len(tindars)*len(solvers))}: "
#                     f"n={tp.n} , connectedness={tp.connectedness}, "
#                     f"solver={solver}"
#                 )
#             obj, sol, stat = tindar_solution(tp, solver)
#
#             result = {
#                 "experiment_id": experiment_id,
#                 "tindar_id": j,
#                 "n": tp.n,
#                 "connectedness": tp.connectedness,
#                 "p": tp.p,
#                 "solver": solver,
#                 "status": stat,
#                 "objective_value": obj,
#             }
#
#             if save_problem_and_solution:
#                 result = {**result, "love_matrix": tp.match_matrix, "solution": sol}
#
#             if verbose:
#                 print(f"{solver} objective value: {obj}")
#
#             results.append(result)
#
#             counter += 1
#
#     with open(result_path, "w") as fp:
#         json.dump(results, fp)
#
#
# def ask_input_val(parameter_name, parameter_type):
#     print(f"Which {parameter_name}?")
#
#     rv_str = input()
#     try:
#         rv = parameter_type(rv_str)
#     except:
#         raise ValueError(f"Could not convert rv_i={rv} to {parameter_type}")
#
#     return rv
#
#
# def ask_input_list(parameter_name, parameter_type):
#     more = True
#     rv = []
#     while more:
#         rv_i = ask_input_val(parameter_name, parameter_type)
#         rv.append(rv_i)
#
#         print("More? Y/N")
#         more_str = input()
#         if more_str.lower() in ["y", "yes"]:
#             more = True
#         elif more_str.lower() in ["n", "no"]:
#             more = False
#         else:
#             raise ValueError("Choose from Y or N")
#
#     return rv
#
#
# if __name__ == "__main__":
#     print("Give this experiment an ID:")
#     experiment_id = input()
#
#     ok = False
#     while not ok:
#         print("Do you want to use the default experiment settings? Y/N")
#         default_setting = input()
#
#         if default_setting in ["Y", "N", "y", "n"]:
#             default_setting = default_setting.lower()[0]
#             ok = True
#         else:
#             print("Choose Y or N")
#
#     if default_setting == "y":
#         n_list = [10, 30, 50, 100, 200, 300, 500]
#         connectedness_list = [1, 3, 5, 8]
#         repeat = 10
#     else:
#         n_list = ask_input_list("n", int)
#         connectedness_list = ask_input_list("connectedness", int)
#         repeat = ask_input_val("repeat", int)
#
#     print("Starting Experiment...")
#     tindar_experiment(
#         experiment_id=experiment_id,
#         n_list=n_list,
#         connectedness_list=connectedness_list,
#         solvers=["heuristic", "pulp"],
#         repeat=repeat,
#         result_directory=PROJECT_DIR + "/data",
#         save_problem_and_solution=False,
#         verbose=True,
#     )
