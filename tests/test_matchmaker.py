import pytest
from typing import List
import numpy as np
from simpleor.matchmaker import MatchMaker


@pytest.fixture
def fake_match_maker() -> MatchMaker:
    match_matrix = np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ]
    )
    return MatchMaker(match_matrix=match_matrix)


def validate_solution_feasibility(solution: List[List]):
    solution_np = np.array(solution)
    m, n = solution_np.shape

    assert m == n, "solution is not square"

    assert (
        solution_np.sum(axis=1) <= 1
    ).all(), "solution assigns one person to two other people"
    assert (
        solution_np.sum(axis=0) <= 1
    ).all(), "solution assigns one person to two other people"
    for i in range(n):
        for j in range(i + 1, n):
            assert solution_np[i, j] == solution_np[i, j], "solution is not symmetric"


def test__create_symmetry_constraints(fake_match_maker):
    constraints_symmetry = fake_match_maker._get_symmetry_constraints()
    if len(constraints_symmetry) != (fake_match_maker.n ** 2 - fake_match_maker.n) / 2:
        raise Exception(
            "Symmetry constraints not constructed right:"
            f"match_matrix.shape = {fake_match_maker.match_matrix.shape},"
            f"len(constraints_symmetry) should be {(fake_match_maker.n ** 2 - fake_match_maker.n) / 2}"
            f", actually is {len(constraints_symmetry)}"
        )


def test__get_feasibility_constraints(fake_match_maker):
    constraints_feasibility = fake_match_maker._get_feasibility_constraints()
    if len(constraints_feasibility) != fake_match_maker.n ** 2:
        raise Exception(
            "Feasibility constraints not constructed right:"
            f"match_matrix.shape = {fake_match_maker.match_matrix.shape}, len(constraints_feasibility)"
            f"should be {fake_match_maker.n ** 2}, actually is {len(constraints_feasibility)}"
        )


def test__get_single_assignment_constraints(fake_match_maker):
    rowsum_constraints, colsum_constraints = (
        fake_match_maker._get_single_assignment_constraints()
    )
    for i, constraints_single in enumerate([rowsum_constraints, colsum_constraints]):
        if len(constraints_single) != fake_match_maker.n:
            raise Exception(
                f"Constraints single {i} not constructed right:"
                f"A.shape = {fake_match_maker.match_matrix.shape}, "
                f"len(constraints_single_{i}) should be {fake_match_maker.n}, "
                f"actually is {len(constraints_single)}"
            )


def test_solve(fake_match_maker):
    fake_match_maker.set_problem()
    for solve_kind in ["heuristic", "pulp"]:
        fake_match_maker.solve(kind=solve_kind)
        validate_solution_feasibility(
            solution=fake_match_maker.get_solution(kind=solve_kind, verbose=False)
        )
