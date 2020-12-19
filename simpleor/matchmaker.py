from typing import Tuple, List
from simpleor.base import BaseSolver, BaseProblemGenerator
from dataclasses import dataclass
from pulp import (
    LpVariable,
    LpAffineExpression,
    LpConstraint,
    LpProblem,
    LpMaximize,
    LpStatus,
    value,
)
import numpy as np
import itertools
import json
from simpleor.utils import PROJECT_DIRECTORY, _check_if_type
import logging

logger = logging.getLogger(f"{__name__}")
INVALID_GENERATE_MATCHMAKER_PROBLEM_INPUT = "generate_matchmaker_problem input invalid."

PROJECT_DIRECTORY_STR = str(PROJECT_DIRECTORY)


@dataclass
class MatchMaker(BaseSolver):
    """Class to solve matchmaker problems

    Args:
        match_matrix (np.array): square binary matrix indicating which
            nodes can be matched. Diagonal must be zero.
    """

    match_matrix: np.array

    def __post_init__(self):
        self.validate_input()
        self.n = self.match_matrix.shape[0]
        self._set_variables()
        self.constraints_all = []
        self.objective = None
        self.x_heuristic_np = np.zeros((self.n, self.n))

    def _set_variables(self):
        self.x_names = [f"x_{i}_{j}" for i in range(self.n) for j in range(self.n)]
        self.x = [LpVariable(name=x_name, cat="Binary") for x_name in self.x_names]
        self.x_np = np.array(self.x).reshape((self.n, self.n))

    def validate_input(self):
        # type check
        if not isinstance(self.match_matrix, np.ndarray):
            raise ValueError("match_matrix is not a numpy array")
        # shape check
        m, n = self.match_matrix.shape
        if m != n:
            raise ValueError(
                f"match_matrix is not square: love_matrix.shape"
                f"= {self.match_matrix.shape}"
            )
        # diagonal zero check
        for i in range(n):
            if self.match_matrix[i, i] != 0:
                raise ValueError("match_matrix diagonal contains nonzeros")

    # Symmetry constraints: if one is paired, the other is paired
    def _get_symmetry_constraints(self) -> list:
        tups = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]

        # Left-hand side
        lhs_symmetry = [
            LpAffineExpression(
                [(self.x_np[tup[0], tup[1]], 1), (self.x_np[tup[1], tup[0]], -1)],
                name=f"lhs_sym_{tup[0]}_{tup[1]}",
            )
            for tup in tups
        ]

        # Constraints
        constraints_symmetry = [
            LpConstraint(
                e=lhs_s,
                sense=0,
                name=f"constraint_sym_{tups[i][0]}_{tups[i][1]}",
                rhs=0,
            )
            for i, lhs_s in enumerate(lhs_symmetry)
        ]
        return constraints_symmetry

    # Feasibility constraints: only pairs if person likes the other
    def _get_feasibility_constraints(self) -> list:
        tups = [(i, j) for i in range(self.n) for j in range(self.n)]

        # Left-hand side
        lhs_like = [
            LpAffineExpression(
                [(self.x_np[tup[0], tup[1]], 1)], name=f"lhs_like_{tup[0]}_{tup[1]}"
            )
            for tup in tups
        ]

        # Constraints
        constraints_like = [
            LpConstraint(
                e=lhs_l,
                sense=-1,
                name=f"constraint_like_{tups[i][0]}_{tups[i][1]}",
                rhs=self.match_matrix[tups[i][0], tups[i][1]],
            )
            for i, lhs_l in enumerate(lhs_like)
        ]
        return constraints_like

    # Single assignment: one person can have at most one other person
    def _get_single_assignment_constraints(self) -> Tuple[list, list]:
        # Left-hand side: rowsum <= 1
        lhs_single_rowsum = [
            LpAffineExpression(
                [(self.x_np[i, j], 1) for j in range(self.n)],
                name=f"lhs_single_rowsum_{i}",
            )
            for i in range(self.n)
        ]

        # Left-hand side: colsum <= 1
        lhs_single_colsum = [
            LpAffineExpression(
                [(self.x_np[i, j], 1) for i in range(self.n)],
                name=f"lhs_single_colsum_{j}",
            )
            for j in range(self.n)
        ]

        # Constraints
        constraints_single_rowsum = self._make_single_assignment_constraints(
            lhs_single=lhs_single_rowsum, kind="rowsum"
        )
        constraints_single_colsum = self._make_single_assignment_constraints(
            lhs_single=lhs_single_colsum, kind="colsum"
        )

        return constraints_single_rowsum, constraints_single_colsum

    # Auxiliary functions for single assigment constraints
    @staticmethod
    def _make_single_assignment_constraints(lhs_single: list, kind: str) -> list:
        constraints_single = [
            LpConstraint(e=lhs_s, sense=-1, name=f"constraint_single_{kind}_{i}", rhs=1)
            for i, lhs_s in enumerate(lhs_single)
        ]
        return constraints_single

    def _set_constraints(self):
        constraints_symmetry = self._get_symmetry_constraints()
        constraints_like = self._get_feasibility_constraints()
        (
            constraints_single_rowsum,
            constraints_single_colsum,
        ) = self._get_single_assignment_constraints()

        self.constraints_all = [
            *constraints_symmetry,
            *constraints_like,
            *constraints_single_rowsum,
            *constraints_single_colsum,
        ]

    def _set_objective(self):
        self.objective = LpAffineExpression([(x_i, 1) for x_i in self.x])

    def set_problem(self):
        # Initialize constraints and objective
        if not self.constraints_all:
            self._set_constraints()
        if not self.objective:
            self._set_objective()

        # Create PuLP problem
        self.prob_pulp = LpProblem("The_Tindar_Problem", LpMaximize)
        self.prob_pulp += self.objective

        for c in self.constraints_all:
            self.prob_pulp += c

    def write_problem(self, path=PROJECT_DIRECTORY_STR + "/models/Tindar.lp"):
        self.prob_pulp.writeLP(path)

    def solve(self, kind: str = "pulp"):
        if kind == "pulp":
            self.prob_pulp.solve()
        elif kind == "heuristic":
            self.solve_heuristic()
        else:
            raise ValueError(f"kind {kind} not allowed" "choose from: pulp, heuristic")

    def solve_heuristic(self):
        for i in range(self.n - 1):
            if self.x_heuristic_np[i, :].sum() == 0:
                done = False
                j = i + 1

                while not done:
                    mutual_interest = (self.match_matrix[i, j] == 1) and (
                        self.match_matrix[j, i] == 1
                    )
                    available = (self.x_heuristic_np[j, :] == 0).all()

                    if mutual_interest and available:
                        self.x_heuristic_np[i, j] = 1
                        self.x_heuristic_np[j, i] = 1
                        done = True

                    if j == self.n - 1:
                        done = True
                    else:
                        j += 1

    def get_status(self, kind: str = "pulp"):
        if kind == "pulp":
            stat = LpStatus[self.prob_pulp.status]
            return stat
        elif kind == "heuristic":
            return "Solved (optimal unsure)"
        else:
            raise ValueError(f"kind {kind} not allowed" "choose from: pulp, heuristic")

    def _pulp_solution_to_np(self, pulp_vars=None):
        if pulp_vars is None:
            pulp_vars = self.prob_pulp.variables()
        solution_np = np.array([v.value() for v in pulp_vars]).reshape((self.n, self.n))
        return solution_np

    def get_solution(self, kind="pulp", verbose=True):
        if kind == "pulp":
            vars_pulp = self.prob_pulp.variables()
            vars_np = self._pulp_solution_to_np(vars_pulp)
            if verbose:
                print(vars_np)
            return vars_np.tolist()

        elif kind == "heuristic":
            if verbose:
                print(self.x_heuristic_np)
            return self.x_heuristic_np.tolist()

    def get_objective_value(self, kind="pulp", verbose=True):
        if kind == "pulp":
            obj = value(self.prob_pulp.objective)
        elif kind == "heuristic":
            obj = self.x_heuristic_np.sum()
        else:
            raise ValueError("kind must be pulp or heuristic")

        if verbose:
            print(f"Number of lovers connected by {kind} = ", obj)
        return obj


@dataclass
class MatchMakerProblemGenerator(BaseProblemGenerator):
    """Class to generate a matchmaker problem

    Args:
        n (int): number of nodes of the matchmaker problem
        edge_probability (0 < float < 1): probability of generating a nonzero
            element of the match_matrix. For small edge_probabilities, there
            are only few potential matches in the matchmaking problem. Vice
            versa for large edge_probability.
        minimum_value (int = 0): minimum value of the matrix' entries
        maximum_value (int = 1): maximum value of the matrix' entries
    """

    n: int
    edge_probability: float
    minimum_value: int
    maximum_value: int

    def __post_init__(self):
        self.validate_input()

    def generate(self):
        """Generates the match_maker_matrix based on sampling.

        First, it samples an (n, n) matrix of integers between minimum_value
        and maximum_value. Then, every sampled element can still be set
        to zero based on a bernoulli draw with parameter edge_probability.


        Sets:
            match_matrix (np.array): matrix of size (n, n) of randomly generated
                integers between minimum_value and maximum_value (zeros on the diagonal).
        """
        match_matrix_unfiltered = np.random.randint(
            low=self.minimum_value, high=self.maximum_value + 1, size=(self.n, self.n)
        )
        keep_edges = np.random.binomial(
            n=1, p=self.edge_probability, size=(self.n, self.n)
        )
        self.match_matrix = match_matrix_unfiltered * keep_edges
        np.fill_diagonal(self.match_matrix, val=0)

    def validate_input(self):
        for x in [self.n, self.minimum_value, self.maximum_value]:
            _check_if_type(
                variable=x,
                kind=int,
                error_message=f"{INVALID_GENERATE_MATCHMAKER_PROBLEM_INPUT}",
            )
        if self.n <= 0:
            raise ValueError(
                f"{INVALID_GENERATE_MATCHMAKER_PROBLEM_INPUT} n={self.n} < 0"
            )
        _check_if_type(
            variable=self.edge_probability,
            kind=float,
            error_message=INVALID_GENERATE_MATCHMAKER_PROBLEM_INPUT,
        )
        if not 0 < self.edge_probability <= 1:
            raise ValueError(
                f"0 < edge_probability <= 1, but edge_probability={self.edge_probability}"
            )


def get_full_match_maker_solution(match_maker: MatchMaker, solver: str) -> Tuple:
    """Get objective value, solution matrix, and solution status

    Args:
        match_maker (MatchMaker): solver MatchMaker instance
        solver (str): solver (pulp/heuristic)

    Returns:
        Tuple(objective value, solution matrix, solution status)
    """
    if solver == "pulp":
        match_maker.set_problem()
    match_maker.solve(kind=solver)

    return (
        match_maker.get_objective_value(kind=solver, verbose=False),
        match_maker.get_solution(kind=solver, verbose=False),
        match_maker.get_status(kind=solver),
    )


def do_match_maker_experiment(
    experiment_id: str,
    n_list: List[int],
    edge_probability_list: List[int],
    repeat: int,
    minimum_value: int = 1,
    maximum_value: int = 10,
    solver_list: list = ["pulp", "heuristic"],
    save_result_directory: str = PROJECT_DIRECTORY_STR + "/data",
    save_problem_and_solution: bool = False,
):
    """Does a match_maker experiment and write results to a json file.

    You can run this function to test the performance of the match_maker solver.
    For each combination of 'n' and 'edge_probability' the function generates
    a match_maker problem 'repeat' times, and solves it with all solvers specified
    in the 'solver_list'.

    Args:
        experiment_id (str): identifier of the experiment
        n_list (list of ints): list of how many nodes in per matchmaker problem
        edge_probability_list: list of probabilities of generating an
            edge between two nodes
        solver_list (list of strings): 'pulp' and/or 'heuristic',
            which solvers should be used to compute results
        repeat (int): number of times a combination of the variables that describe
            a matchmaker problem (n, edge_probability) should be repeated
            in the experiment
        minimum_value (int, default 1): minimum value of element in match_matrix
        maximum_value (int, default 10): maximum value of element in match_matrix
        save_result_directory: str of a path
        save_problem_and_solution: bool
            if true, saves the love_matrix and solution matrix
    """

    result_path = f"{save_result_directory}/results_experiment_{experiment_id}.json"
    parameters = tuple(itertools.product(n_list, edge_probability_list))

    match_maker_problem_objects_nested = [
        [
            MatchMakerProblemGenerator(
                n=n,
                edge_probability=edge_probability,
                minimum_value=minimum_value,
                maximum_value=maximum_value,
            )
            for _ in range(repeat)
        ]
        for n, edge_probability in parameters
    ]
    match_maker_problem_objects_flat = [
        item for sublist in match_maker_problem_objects_nested for item in sublist
    ]
    for problem_generator in match_maker_problem_objects_flat:
        problem_generator.generate()
    match_matrices = [
        problem_generator.match_matrix
        for problem_generator in match_maker_problem_objects_flat
    ]
    number_of_experiments = len(match_matrices) * len(solver_list)

    results = []
    for solver in solver_list:
        for i, match_matrix in enumerate(match_matrices):
            match_maker = MatchMaker(match_matrix=match_matrix)
            logger.info(
                f"Experiment {i + 1}/{number_of_experiments}: "
                f"n={match_maker.n}, solver={solver}"
            )
            objective, solution, status = get_full_match_maker_solution(
                match_maker, solver
            )

            result = {
                "experiment_id": experiment_id,
                "tindar_id": i,
                "n": match_maker.n,
                "solver": solver,
                "status": status,
                "objective_value": objective,
            }

            if save_problem_and_solution:
                result = {
                    **result,
                    "match_matrix": match_maker.match_matrix.tolist(),
                    "solution": solution,
                }

            logger.info(f"{solver} objective value: {objective}")
            results.append(result)

    with open(result_path, "w") as fp:
        json.dump(results, fp)


def ask_input_val(parameter_name, parameter_type):
    print(f"Which {parameter_name}?")
    rv_str = input()
    return parameter_type(rv_str)


def ask_input_list(parameter_name, parameter_type):
    more = True
    rv = []
    while more:
        rv_i = ask_input_val(parameter_name, parameter_type)
        rv.append(rv_i)

        print("More? Y/N")
        more_str = input()
        if more_str.lower() in ["y", "yes"]:
            more = True
        elif more_str.lower() in ["n", "no"]:
            more = False
        else:
            raise ValueError("Choose from Y or N")

    return rv


if __name__ == "__main__":
    print("Give this experiment an ID:")
    experiment_id = input()

    ok = False
    while not ok:
        print("Do you want to use the default experiment settings? Y/N")
        default_setting = input()

        if default_setting in ["Y", "N", "y", "n"]:
            default_setting = default_setting.lower()[0]
            ok = True
        else:
            print("Choose Y or N")

    if default_setting == "y":
        n_list = [10, 30, 50, 100]
        edge_probability_list = np.arange(start=0.3, stop=1, step=0.15)
        repeat = 5
    else:
        n_list = ask_input_list("n", int)
        edge_probability_list = ask_input_list("connectedness", int)
        repeat = ask_input_val("repeat", int)

    logger.info("Starting Experiment...")
    do_match_maker_experiment(
        experiment_id=experiment_id,
        n_list=n_list,
        edge_probability_list=edge_probability_list,
        repeat=repeat,
        solver_list=["heuristic", "pulp"],
        save_result_directory=PROJECT_DIRECTORY_STR + "/data/matchmaker",
        save_problem_and_solution=True,
    )
