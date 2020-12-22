# scheduler.py

from typing import List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging
from pulp import LpVariable, LpProblem, LpMaximize, lpSum, LpStatus, lpDot
from simpleor.base import BaseSolver, BaseProblemGenerator
from simpleor.utils import PROJECT_DIRECTORY

logger = logging.getLogger(f"{__name__}")

PROBLEM_NAME = "The_Schedule_Problem"
READ_OPTIONS = ["csv", "excel"]
WRITE_OPTIONS = ["csv", "excel"]


@dataclass
class ScheduleSolver(BaseSolver):
    """Class for solving scheduling problems

    Args:
        task_durations (list): integers with task durations
        available_timeslots (list): list of list of integers where 1
            indicates the agent is available and 0 not available.
        task_rewards (list): floats of reward received for completing task

    Typically, you would want to solve the problem after initialization
    and inspect the solution. E.g:
        schedule_solver = ScheduleSolver(task_durations, available_timeslots)
        schedule_solver.solve()
        print(schedule_solver.get_solution_status())
        print(schedule_solver.get_objective_value())
        df_solution = schedule_solver.get_solution(kind="dataframe")
    """

    task_durations: List[int]
    available_timeslots: List[List[int]]
    task_rewards: Optional[List[Union[int, float]]] = None

    def __post_init__(self):
        self.validate_input(
            self.task_durations, self.available_timeslots, self.task_rewards
        )
        self.n_tasks = len(self.task_durations)
        self.n_agents = len(self.available_timeslots)
        self.n_timeslots = len(self.available_timeslots[0])
        self.available_timeslots_np = np.array(self.available_timeslots).reshape(
            (self.n_agents, self.n_timeslots)
        )

        self.pulp_problem = LpProblem(name=PROBLEM_NAME, sense=LpMaximize)
        self.objective = None
        self.constraints_list = None
        self.lp_variables_created = False
        self.problem_is_set = False
        self.big_m = self.n_timeslots * self.n_agents * self.n_timeslots
        self.solution = None
        self.solution_df = None

    @classmethod
    def validate_input(cls, task_durations, available_schedule, task_rewards):
        cls.validate_task_durations(task_durations)
        cls.validate_available_schedule(available_schedule)
        cls.validate_task_rewards(task_rewards)
        cls.validate_task_durations_and_rewards_compatible(
            task_durations=task_durations, task_rewards=task_rewards
        )

    @classmethod
    def validate_task_durations(cls, task_durations):
        if not isinstance(task_durations, list):
            raise ValueError("task_durations must be list")
        if not all([isinstance(x, int) for x in task_durations]):
            raise ValueError("task_durations elements must be int")
        cls._check_if_elements_positive(task_durations)

    @staticmethod
    def _check_if_elements_positive(array):
        if not all([x > 0 for x in array]):
            raise ValueError("element in array is negative")

    @staticmethod
    def validate_available_schedule(available_schedule):
        if not isinstance(available_schedule, list):
            raise ValueError("available_schedule must be list")
        for i, row in enumerate(available_schedule):
            if not isinstance(row, list):
                raise ValueError("available_schedule first order elements must be list")
            if i == 0:
                row_length = len(row)
            else:
                if len(row) != row_length:
                    raise ValueError(
                        "row lengths are not the same in available_schedule"
                    )
            for x in row:
                if (x != 0) and (x != 1):
                    raise ValueError(
                        "available_schedule second order elements must be int 0/1"
                    )

    @classmethod
    def validate_task_rewards(cls, task_rewards):
        if task_rewards is None:
            return
        if not isinstance(task_rewards, list):
            raise ValueError("task_rewards must be list")
        if not all([isinstance(x, int) or isinstance(x, float) for x in task_rewards]):
            raise ValueError("task_rewards elements must be float")
        cls._check_if_elements_positive(task_rewards)

    @staticmethod
    def validate_task_durations_and_rewards_compatible(task_durations, task_rewards):
        if task_rewards is None:
            return
        if len(task_durations) != len(task_rewards):
            raise ValueError("task_durations and task_rewards not the same length")

    def _set_variables(self):
        logger.info("Setting variables...")
        self.start_names = [
            f"start_{i}_{j}_{k}"
            for i in range(self.n_tasks)
            for j in range(self.n_agents)
            for k in range(self.n_timeslots)
        ]
        self.start_variables = [
            LpVariable(name=start_name, cat="Binary") for start_name in self.start_names
        ]
        self.active_names = [
            f"active_{i}_{j}_{k}"
            for i in range(self.n_tasks)
            for j in range(self.n_agents)
            for k in range(self.n_timeslots)
        ]
        self.active_variables = [
            LpVariable(name=active_name, cat="Binary")
            for active_name in self.active_names
        ]
        self.start_variables_np = np.array(self.start_variables).reshape(
            (self.n_tasks, self.n_agents, self.n_timeslots)
        )
        self.active_variables_np = np.array(self.active_variables).reshape(
            (self.n_tasks, self.n_agents, self.n_timeslots)
        )
        self.lp_variables_created = True

    def _set_objective(self):
        if self.task_rewards is None:
            logger.info("Setting objective with equal reward for every task...")
            self.objective = lpSum(self.start_variables_np)
        else:
            logger.info("Setting objective with task rewards...")
            self.objective = lpDot(self.start_variables_np, self.task_rewards)

    def _set_constraints(self):
        logger.info("Setting constraints...")
        self.max_one_start_per_task_constraints_list = (
            self._get_max_one_start_per_task_constraints()
        )
        self.operate_or_start_not_available_constraint_list = (
            self._get_operate_and_start_not_allowed_constraints()
        )
        self.one_task_simultaneous_constraint_list = (
            self._get_one_task_simultaneous_constraints()
        )
        self.finish_if_started_constraint_list = (
            self._get_finish_if_started_constraints()
        )
        self.start_only_if_available_constraint_list = (
            self._get_start_only_if_available()
        )
        self.constraints_list = [
            *self.max_one_start_per_task_constraints_list,
            *self.operate_or_start_not_available_constraint_list,
            *self.one_task_simultaneous_constraint_list,
            *self.finish_if_started_constraint_list,
            *self.start_only_if_available_constraint_list,
        ]

    def _get_max_one_start_per_task_constraints(self) -> List:
        max_one_start_constraints = []
        for i in range(self.n_tasks):
            max_one_start_constraints.append(
                lpSum(self.start_variables_np[i, :, :]) <= 1
            )
        return max_one_start_constraints

    def _get_operate_and_start_not_allowed_constraints(self) -> List:
        not_allowed_constraint_list = []
        for i in range(self.n_tasks):
            for j in range(self.n_agents):
                for t in range(self.n_timeslots):
                    not_allowed_constraint_list.append(
                        self.active_variables_np[i, j, t]
                        <= self.available_timeslots_np[j, t]
                    )
                    not_allowed_constraint_list.append(
                        self.start_variables_np[i, j, t]
                        <= self.available_timeslots_np[j, t]
                    )
        return not_allowed_constraint_list

    # TODO: are constraints above a subset of these below?
    def _get_start_only_if_available(self) -> List:
        constraints = []
        for i, task_duration in enumerate(self.task_durations):
            for j in range(self.n_agents):
                latest_start = self.n_timeslots - task_duration
                for t in range(latest_start + 1):
                    start_not_allowed = (
                        sum(self.available_timeslots_np[j, t : t + task_duration])
                        < task_duration
                    )
                    if start_not_allowed:
                        constraints.append(self.start_variables_np[i, j, t] == 0)
                for t in range(latest_start + 1, self.n_timeslots):
                    constraints.append(self.start_variables_np[i, j, t] == 0)
        return constraints

    def _get_finish_if_started_constraints(self) -> List:
        finish_if_started_constraint_list = []
        for i, current_task_duration in enumerate(self.task_durations):
            current_latest_start = self.n_timeslots - self.task_durations[i]
            for j in range(self.n_agents):
                for t in range(current_latest_start + 1):
                    t_range = t + current_task_duration
                    for t_running in range(t, t_range):
                        finish_if_started_constraint_list.append(
                            self.active_variables_np[i, j, t_running]
                            >= self.start_variables_np[i, j, t]
                        )
        return finish_if_started_constraint_list

    def _get_one_task_simultaneous_constraints(self) -> List:
        one_task_simultaneous_list = []
        for j in range(self.n_agents):
            for t in range(self.n_timeslots):
                one_task_simultaneous_list.append(
                    lpSum([self.active_variables_np[:, j, t]]) <= 1
                )
        return one_task_simultaneous_list

    def set_problem(self):
        """Sets the Linear Programming problem of the object.
        This functions sets the variables, constraints, and objective."""
        logger.info("Setting LP problem...")
        if not self.lp_variables_created:
            self._set_variables()
        if self.objective is None:
            self._set_objective()
        if self.constraints_list is None:
            self._set_constraints()

        self.pulp_problem += self.objective
        for constraint in self.constraints_list:
            self.pulp_problem += constraint
        self.problem_is_set = True
        logger.info("Problem successfully set.")

    def solve(self):
        """Solves the scheduling problem based on the object's attribute"""
        if not self.problem_is_set:
            logger.info("Problem was not set yet, setting now...")
            self.set_problem()
        logger.info("Solving problem...")
        self.pulp_problem.solve()
        self._set_solution()

    def get_status(self) -> str:
        """Returns the solution status after the problem has been tried to solve"""
        return LpStatus[self.pulp_problem.status]

    def get_objective_value(self) -> float:
        """Returns the objective value of the optimal solution (if solution is found)"""
        return self.objective.value()

    @staticmethod
    def _get_one_pulp_variable_value(pulp_variable):
        return pulp_variable.value()

    @property
    def _vectorized_get_solution_value(self):
        return np.vectorize(self._get_one_pulp_variable_value)

    def _set_solution(self):
        self.start_variables_solution = self._vectorized_get_solution_value(
            self.start_variables_np
        ).astype(int)
        self.active_variables_solution = self._vectorized_get_solution_value(
            self.active_variables_np
        ).astype(int)
        task_started, on_agent, at_time = np.where(self.start_variables_solution)

        self.solution = [
            (
                task_started[i],
                on_agent[i],
                at_time[i],
                at_time[i] + self.task_durations[task_started[i]],
                self.task_durations[task_started[i]],
            )
            for i in range(len(task_started))
        ]
        self.solution_df = pd.DataFrame(
            data=self.solution,
            columns=["task", "agent", "start", "stop", "task_duration"],
        )

    def get_solution(self, kind: Optional[str] = "native"):
        """Get the solution of the scheduling problem

        Args:
            kind (str, optional): choose 'native' to return native python objects,
                'dataframe' for a pandas solution.
        Returns:
            the solution as native python object or dataframe
        """
        if (self.solution is None) or (self.solution_df is None):
            logger.info("solution is None. Trying to set the solution...")
            try:
                self._set_solution()
            except AttributeError as e:
                logger.critical(
                    "Could not set solution. Did you set and solve the problem correctly?"
                )
                raise e
        if kind == "native":
            return self.solution
        elif kind == "dataframe":
            return self.solution_df
        else:
            raise ValueError(f"kind {kind} not recognized. Choose native/dataframe")

    def write_solution(self, directory: str, filename: str, how: str):
        """Write solution of scheduling problem to disk

        Args:
            directory (str): directory where the solution should be saved
            filename (str): filename of the solution
            how (str): how the file should be saved
                (choose 'csv' or 'excel')
        """
        full_path = Path(directory).joinpath(Path(filename))
        logger.info(f"Writing solution to {full_path}")
        if how == "csv":
            self.solution_df.to_csv(full_path, index=False)
        elif how == "excel":
            self.solution_df.to_excel(full_path, index=False)
        else:
            raise ValueError(
                f"how = {how} not in WRITE_OPTIONS. " f"Choose from {WRITE_OPTIONS}"
            )


@dataclass
class ScheduleProblemGenerator(BaseProblemGenerator):
    """ Class for generating scheduling problems.

    Args:
        n_agents (int): number of agents
        n_timeslots (int): number of timeslots
        n_tasks (int): number of tasks
        min_task_duration (int, optional): minimum task duration. Defaults to 1.
        max_task_duration (int, optional): maximum task duration. Defaults to n_timeslots.
        min_block_duration (int, optional): minimum number of consecutive timeslots
                            that an agent has to be available. Defaults to 1.
                            (e.g. min_block_duration = 3 implies
                            that the agent is either not available
                            or available for at least 3 timeslots consecutively)
        max_block_duration (int, optional): maximum number of consecutive timeslots
                            that an agent has to be available. Defaults to n_timeslots.
    """

    n_agents: int
    n_timeslots: int
    n_tasks: int
    min_task_duration: Optional[int] = 1
    max_task_duration: Optional[int] = None
    min_block_duration: Optional[int] = 1
    max_block_duration: Optional[int] = None

    def __post_init__(self):
        self.task_durations = []
        self.available_timeslots = [[]]
        self.task_rewards = None

        self.max_n_blocks = (
            int(self.n_timeslots / self.min_block_duration) + 1
        ) * self.n_agents

        if self.max_task_duration is None:
            self.max_task_duration = self.n_timeslots
        if self.max_block_duration is None:
            self.max_block_duration = self.n_timeslots

    def validate_input(self):
        # TODO
        pass

    def generate(
        self,
        generate_rewards: Optional[bool] = False,
        min_reward: Optional[int] = None,
        max_reward: Optional[int] = None,
    ):
        """ Generates a scheduling problem based on class attributes

        The main attributes that are set with this method are
        self.available_timeslots, self.task_durations and optionally
        self.task_rewards if generate_rewards = True.

        Args:
            generate_rewards (bool): if the rewards should be
                generated or not
            min_reward (int, optional): minimum reward value
            max_reward (int, optional): maximum reward value
        """
        self._generate_tasks()
        self._generate_available_timeslots()
        if generate_rewards:
            self._validate_reward_input(min_reward=min_reward, max_reward=max_reward)
            self._generate_rewards(min_reward=min_reward, max_reward=max_reward)

    def _generate_tasks(self):
        allowed_durations = list(
            range(self.min_task_duration, self.max_task_duration + 1)
        )
        self.task_durations = np.random.choice(
            a=allowed_durations, size=self.n_tasks
        ).tolist()

    @staticmethod
    def _validate_reward_input(min_reward, max_reward):
        for x in (min_reward, max_reward):
            if not isinstance(x, int):
                raise ValueError("rewards must be int")
            if x <= 0:
                raise ValueError("rewards must be positive")

    def _generate_rewards(self, min_reward, max_reward):
        allowed_rewards = list(range(min_reward, max_reward + 1))
        self.task_rewards = np.random.choice(
            a=allowed_rewards, size=self.n_tasks
        ).tolist()

    def _generate_available_timeslots(self):
        allowed_block_durations = list(
            range(self.min_block_duration, self.max_block_duration + 1)
        )
        block_durations = np.random.choice(
            a=allowed_block_durations, size=self.max_n_blocks
        ).tolist()

        availability_rows = []
        for i in range(self.n_agents):
            availability_row_np = np.zeros((self.n_timeslots,), dtype=int)
            goto_next_agent = False
            while not goto_next_agent:
                block_duration = block_durations.pop()
                allowed_start_times = self._allowed_start_times(
                    availability_row_np, block_duration
                )
                if allowed_start_times:
                    start_time = np.random.choice(a=allowed_start_times, size=1)[0]
                    availability_row_np[start_time : start_time + block_duration] = 1
                else:
                    availability_rows.append(availability_row_np.tolist())
                    goto_next_agent = True
            self.available_timeslots = availability_rows

    def _allowed_start_times(
        self, availability_row: np.array, block_duration: int
    ) -> List:
        allowed_start_times = []
        index = 0
        latest_start_time = self.n_timeslots - block_duration
        while index <= latest_start_time:
            start = index
            if start > 0:
                start -= 1
            stop = index + block_duration
            if stop < self.n_timeslots:
                stop += 1
            block_inspected = availability_row[start:stop]
            if not any(block_inspected):
                allowed_start_times.append(index)
            index += 1

        return allowed_start_times


def read_schedule_problem(
    task_durations_file_path: str, available_schedule_file_path: str, how: str, **kwargs
) -> Tuple:
    """ Reads scheduling problem data from csv files

    Args:
        task_durations_file_path (str): path to the task durations file
        available_schedule_file_path (str): path to the file with information
            about when agents are available (str)
        how (str): method of how to read (choose csv/excel)

    Returns:
        task_durations (list): list with task durations
        available_schedule (list): list with information about when an agent
            is available
    """
    logger.info(
        f"Reading data from {task_durations_file_path} and {available_schedule_file_path} with how={how}..."
    )
    if how == "csv":
        task_durations_df = pd.read_csv(
            task_durations_file_path, header=None, dtype=int, **kwargs
        )
        available_schedule_df = pd.read_csv(
            available_schedule_file_path, header=None, dtype=bool, **kwargs
        )
    elif how == "excel":
        task_durations_df = pd.read_excel(task_durations_file_path, dtype=int, **kwargs)
        available_schedule_df = pd.read_excel(
            available_schedule_file_path, dtype=bool, **kwargs
        )
    else:
        raise ValueError(
            f"how = {how} not in READ_OPTIONS. " f"Choose from {READ_OPTIONS}"
        )

    validate_task_data(task_durations=task_durations_df)
    validate_schedule_data(available_schedule=available_schedule_df)

    task_durations = task_durations_df.values[:, 0].tolist()
    available_schedule = available_schedule_df.values.tolist()

    return task_durations, available_schedule


def validate_task_data(task_durations: pd.DataFrame):
    assert (
        task_durations.notnull().all().all()
    ), f"Some task durations are missing: {task_durations}"
    assert (
        len(task_durations.columns) == 1
    ), f"Multiple columns for task durations data: {task_durations}"


def validate_schedule_data(available_schedule: pd.DataFrame):
    assert (
        available_schedule.notnull().all().all()
    ), f"Some task schedule data is missing: {available_schedule}"


if __name__ == "__main__":
    schedule_generator = ScheduleProblemGenerator(
        n_agents=5,
        n_timeslots=10,
        n_tasks=7,
        min_task_duration=1,
        max_task_duration=6,
        min_block_duration=2,
        max_block_duration=4,
    )
    schedule_generator.generate(generate_rewards=True, min_reward=1, max_reward=10)

    logger.info("==============================================")
    logger.info("PROBLEM")
    logger.info(f"task_durations:\n{np.array(schedule_generator.task_durations)}\n\n")
    logger.info(
        f"available_timeslots:\n{np.array(schedule_generator.available_timeslots)}\n\n"
    )
    logger.info("----------------------------------------------")
    schedule_solver = ScheduleSolver(
        task_durations=schedule_generator.task_durations,
        available_timeslots=schedule_generator.available_timeslots,
        task_rewards=schedule_generator.task_rewards,
    )
    schedule_solver.set_problem()
    schedule_solver.solve()
    logger.info("----------------------------------------------")
    logger.info(f"solution status: {schedule_solver.get_status()}\n\n")
    logger.info(
        f"Solution objective value: {schedule_solver.get_objective_value()}\n\n"
    )
    logger.info(
        f"Solution (task - on machine - active): {schedule_solver.get_solution()}"
    )
    logger.info("----------------------------------------------")
    logger.info("Writing solution to data/scheduler")
    schedule_solver.write_solution(
        directory=str(PROJECT_DIRECTORY.joinpath("data/scheduler")),
        filename="solution.csv",
        how="csv",
    )
