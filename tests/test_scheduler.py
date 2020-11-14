# test_scheduler.py

import pytest
from simpleor.scheduler import ScheduleSolver

test_input = [  # task duration, available schedule
    # test input 0
    [[1], [[0, 0], [0, 0]]],
    # test input 1
    [[2, 3], [[1, 1], [1, 0]]],
    # test input 2
    [[2, 3], [[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]]],
    # test input 3
    [[1, 2], [[1, 0, 0], [1, 1, 0]]],
    # test input 4
    [[2, 1, 3], [[1, 1, 0, 1, 1, 1], [0, 0, 0, 1, 0, 0]]],
    # test input 5
    [[1, 2, 3], [[1, 0, 1, 1], [0, 1, 1, 1]]],
]

test_solution = [  # task duration, available_schedule
    # test input 0
    [],  # nothing gets scheduled
    # test input 1
    [(0, 0, (0, 2))],  # task 0 on machine 0, start at t=0, end at t=2
    # test input 2
    [(0, 0, (0, 2)), (1, 1, (2, 5))],
    # test input 3
    [(0, 0, (0, 1)), (1, 1, (0, 2))],
    # test input 4
    [(0, 0, (0, 2)), (1, 1, (3, 4)), (2, 0, (3, 6))],
    # test input 5
    [(0, 0, (0, 1)), (1, 0, (2, 4)), (2, 1, (1, 4))],
]

pytest_parameters = list(zip(test_input, test_solution))


@pytest.mark.parametrize("test_input, expected", pytest_parameters)
def test_schedule_solver(test_input, expected):
    schedule_solver = ScheduleSolver(
        task_durations=test_input[0], available_timeslots=test_input[1]
    )
    schedule_solver.solve()
    solution = schedule_solver.get_solution()
    assert solution == expected
