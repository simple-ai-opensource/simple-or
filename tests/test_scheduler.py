# test_scheduler.py

# https://stackoverflow.com/questions/34406848/mocking-a-class-method-and-changing-some-object-attributes-in-python

import pytest
import pandas as pd
import logging
from pandas.testing import assert_frame_equal
import numpy as np
from unittest.mock import patch
from simpleor.scheduler import (
    ScheduleSolver,
    ScheduleProblemGenerator,
    read_schedule_problem,
)
from simpleor.utils import PROJECT_DIRECTORY

logger = logging.getLogger(f"{__name__}")

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


@patch("simpleor.scheduler.ScheduleSolver._vectorized_get_solution_value")
def test__set_solution(mock_vect_get_solution_value):
    task_durations = [7, 4, 3]
    available_timeslots = [
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    ]
    mock_start_variables_np = np.array(
        [
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
        ]
    )
    mock_active_variables_np = mock_start_variables_np  # irrelevant
    mock_vect_get_solution_value.side_effect = [
        mock_start_variables_np,
        mock_active_variables_np,
    ]

    mock_solver = ScheduleSolver(
        task_durations=task_durations, available_timeslots=available_timeslots
    )
    mock_solver.start_variables_np = None
    mock_solver.active_variables_np = None
    mock_solver._set_solution()

    expected = pd.DataFrame(
        data=[[1, 0, 1, 5, 4], [2, 1, 2, 5, 3]],
        columns=["task", "agent", "start", "stop", "task_duration"],
    )
    assert_frame_equal(mock_solver.solution_df, expected)


@pytest.mark.parametrize("test_input, expected", pytest_parameters)
def test_schedule_solver(test_input, expected):
    logger.info("Testing solver...")
    logger.info("Hardcoded tests...")
    schedule_solver = ScheduleSolver(
        task_durations=test_input[0], available_timeslots=test_input[1]
    )
    schedule_solver.solve()
    solution = schedule_solver.get_solution()
    assert solution == expected

    # Random generation
    logger.info("Random tests...")
    generator_parameters = [*[[3, 5, 7]] * 10]  # operators, timeslots, tasks
    for i, args in enumerate(generator_parameters):
        logger.info(f"Now at test {i + 1}/{len(generator_parameters)}...")
        generator = ScheduleProblemGenerator(*args)
        generator.generate()
        solver = ScheduleSolver(
            task_durations=generator.task_durations,
            available_timeslots=generator.available_timeslots,
        )
        solver.set_problem()
        solver.solve()
        solution_df = solver.get_solution(kind="dataframe")
        agent_busy_np = np.zeros((args[0], args[1]), dtype=int)

        for i, row in solution_df.iterrows():
            agent_busy_np[row["agent"], row["start"] : row["stop"]] += 1
        available_timeslots_np = np.array(generator.available_timeslots)
        schedule_okay = agent_busy_np - available_timeslots_np
        assert (schedule_okay <= 0).all()


def test_schedule_solver_with_reward():
    task_durations = [4, 2, 2, 2]
    available_timeslots = [[1, 1, 1, 1]]
    task_rewards = [100, 1, 2, 1]
    schedule_solver = ScheduleSolver(
        task_durations=task_durations,
        available_timeslots=available_timeslots,
        task_rewards=task_rewards,
    )
    schedule_solver.solve()
    df_solution = schedule_solver.get_solution(kind="dataframe")
    expected = pd.DataFrame(
        {"task": [0], "agent": [0], "start": [0], "stop": [4], "task_duration": [4]}
    )
    assert_frame_equal(df_solution, expected)


def test_read_schedule_problem():
    # csv
    task_durations, available_schedule = read_schedule_problem(
        task_durations_file_path=PROJECT_DIRECTORY.joinpath(
            "data/test/scheduler/task_durations.csv"
        ),
        available_schedule_file_path=PROJECT_DIRECTORY.joinpath(
            "data/test/scheduler/available_schedule.csv"
        ),
        how="csv",
    )

    assert task_durations == [3, 2]
    assert available_schedule == [[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]]


def test__generate_rewards():
    test_inputs = ((1, 1, 1), (10, 1, 1))
    expected = ([1], [1] * 10)
    for i, (n_tasks, min_reward, max_reward) in enumerate(test_inputs):
        generator = ScheduleProblemGenerator(
            n_agents=1, n_timeslots=1, n_tasks=n_tasks  # irrelevant  # irrelevant
        )
        generator._generate_rewards(min_reward=min_reward, max_reward=max_reward)
        assert generator.task_rewards == expected[i]
    for _ in range(20):
        generator._generate_rewards(min_reward=1, max_reward=3)
        assert all([1 <= x <= 3 for x in generator.task_rewards])
