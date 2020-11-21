# test_scheduler.py

import pytest
from simpleor.scheduler import ScheduleSolver, read_schedule_problem


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
    # expected output 0
    [],  # nothing gets scheduled
    # expected output 1
    [(0, 0, 0, 2, 2)],  # task 0 on machine 0, start at t=0, end at t=2
    # expected output 2
    [(0, 0, 0, 2, 2), (1, 1, 2, 5, 3)],
    # expected output 3
    [(0, 0, 0, 1, 1), (1, 1, 0, 2, 2)],
    # expected output 4
    [(0, 0, 0, 2, 2), (1, 1, 3, 4, 1), (2, 0, 3, 6, 3)],
    # expected output 5
    [(0, 0, 0, 1, 1), (1, 0, 2, 4, 2), (2, 1, 1, 4, 3)],
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


def test_read_schedule_problem():
    # csv
    task_durations, available_schedule = read_schedule_problem(
        task_durations_file_path="../data/test/scheduler/task_durations.csv",
        available_schedule_file_path="../data/test/scheduler/available_schedule.csv",
        how="csv",
    )

    assert task_durations == [3, 2]
    assert available_schedule == [[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]]
