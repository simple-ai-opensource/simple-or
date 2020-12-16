# import pytest
# import numpy as np
# from simpleor.matchmaker import MatchMaker
#
#
# @pytest.fixture
# def fake_match_maker() -> MatchMaker:
#     match_matrix = np.array([
#         [0, 1, 1, 1, 1],
#         [1, 0, 1, 1, 1],
#         [1, 1, 0, 1, 1],
#         [1, 1, 1, 0, 1],
#         [1, 1, 1, 1, 0],
#     ])
#     return MatchMaker(match_matrix=match_matrix)
#
#
# def test__create_symmetry_constraints(fake_match_maker):
#     constraints_symmetry = fake_match_maker._get_symmetry_constraints()
#     if len(constraints_symmetry) != (fake_match_maker.n ** 2 - fake_match_maker.n) / 2:
#         raise Exception(
#             "Symmetry constraints not constructed right:"
#             f"match_matrix.shape = {fake_match_maker.match_matrix.shape},"
#             f"len(constraints_symmetry) should be {(fake_match_maker.n ** 2 - fake_match_maker.n) / 2}"
#             f", actually is {len(constraints_symmetry)}"
#         )
#
#
# def test__get_feasibility_constraints(fake_match_maker):
#     constraints_feasibility = fake_match_maker._get_feasibility_constraints()
#     if len(constraints_feasibility) != fake_match_maker.n ** 2:
#         raise Exception(
#             "Feasibility constraints not constructed right:"
#             f"match_matrix.shape = {fake_match_maker.match_matrix.shape}, len(constraints_feasibility)"
#             f"should be {fake_match_maker.n ** 2}, actually is {len(constraints_feasibility)}"
#         )
#
#
# def test__get_single_assignment_constraints(fake_match_maker):
#     rowsum_constraints, colsum_constraints = fake_match_maker._get_single_assignment_constraints()
#     for i, constraints_single in enumerate([rowsum_constraints, colsum_constraints]):
#         if len(constraints_single) != fake_match_maker.n:
#             raise Exception(
#                 f"Constraints single {i} not constructed right:"
#                 f"A.shape = {fake_match_maker.match_matrix.shape}, "
#                 f"len(constraints_single_{i}) should be {fake_match_maker.n}, "
#                 f"actually is {len(constraints_single)}"
#             )
