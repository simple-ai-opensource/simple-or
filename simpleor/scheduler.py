# scheduler.py

from typing import List
from dataclasses import dataclass
import numpy as np
import logging
from pulp import LpVariable, LpProblem, LpMaximize, lpSum, LpStatus
from simpleor.base import Solver, Generator

logger = logging.getLogger(f"{__name__}")

PROBLEM_NAME = "The_Schedule_Problem"


@dataclass
class ScheduleSolver(Solver):
    task_durations: List[int]
    available_timeslots: List[List[int]]

    def __post_init__(self):
        self.validate_input(self.task_durations, self.available_timeslots)
        self.n_tasks = len(self.task_durations)
        self.n_operators = len(self.available_timeslots)
        self.n_timeslots = len(self.available_timeslots[0])
        self.available_timeslots_np = np.array(self.available_timeslots).reshape(
            (self.n_operators, self.n_timeslots)
        )

        self.pulp_problem = LpProblem(name=PROBLEM_NAME, sense=LpMaximize)
        self.objective = None
        self.constraints_list = None
        self.lp_variables_created = False
        self.problem_is_set = False
        self.big_m = self.n_timeslots * self.n_operators * self.n_timeslots

    @classmethod
    def validate_input(cls, task_durations, available_schedule):
        cls.validate_task_durations(task_durations)
        cls.validate_available_schedule(available_schedule)

    @staticmethod
    def validate_task_durations(task_durations):
        if not isinstance(task_durations, list):
            raise ValueError("task_durations must be list")
        if not all([isinstance(x, int) for x in task_durations]):
            raise ValueError("task_durations elements must be int")

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

    def _set_variables(self):
        logger.info("Setting variables...")
        self.start_names = [
            f"start_{i}_{j}_{k}"
            for i in range(self.n_tasks)
            for j in range(self.n_operators)
            for k in range(self.n_timeslots)
        ]
        self.start_variables = [
            LpVariable(name=start_name, cat="Binary") for start_name in self.start_names
        ]
        self.active_names = [
            f"active_{i}_{j}_{k}"
            for i in range(self.n_tasks)
            for j in range(self.n_operators)
            for k in range(self.n_timeslots)
        ]
        self.active_variables = [
            LpVariable(name=active_name, cat="Binary")
            for active_name in self.active_names
        ]
        self.start_variables_np = np.array(self.start_variables).reshape(
            (self.n_tasks, self.n_operators, self.n_timeslots)
        )
        self.active_variables_np = np.array(self.active_variables).reshape(
            (self.n_tasks, self.n_operators, self.n_timeslots)
        )
        self.lp_variables_created = True

    def _set_objective(self):
        logger.info("Setting objective...")
        self.objective = lpSum(self.start_variables_np)

    def _set_constraints(self):
        logger.info("Setting constraints...")
        self.max_one_start_per_task_constraints_list = (
            self._get_max_one_start_per_task_constraints()
        )
        self.operate_or_start_not_allowed_constraint_list = (
            self._get_operate_and_start_not_allowed_constraints()
        )
        self.one_task_simultaneous_constraint_list = (
            self._get_one_task_simultaneous_constraints()
        )
        self.finish_if_started_constraint_list = (
            self._get_finish_if_started_constraints()
        )
        self.no_operation_if_no_finish_constraint_list = (
            self._get_no_start_or_active_if_no_finish_constraints()
        )
        self.constraints_list = [
            *self.max_one_start_per_task_constraints_list,
            *self.operate_or_start_not_allowed_constraint_list,
            *self.one_task_simultaneous_constraint_list,
            *self.finish_if_started_constraint_list,
            *self.no_operation_if_no_finish_constraint_list,
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
            for j in range(self.n_operators):
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

    # TODO: fix range for t!
    def _get_finish_if_started_constraints(self) -> List:
        finish_if_started_constraint_list = []
        for i, current_task_duration in enumerate(self.task_durations):
            current_latest_start = self.n_timeslots - self.task_durations[i]
            for j in range(self.n_operators):
                for t in range(current_latest_start + 1):
                    t_range = t + current_task_duration
                    finish_if_started_constraint_list.append(
                        lpSum(self.active_variables_np[i, j, t:t_range])
                        >= self.start_variables_np[i, j, t] * current_task_duration
                    )
        return finish_if_started_constraint_list

    def _get_no_start_or_active_if_no_finish_constraints(self) -> List:
        no_do_if_no_finish_constraint_list = []
        for i, current_task_duration in enumerate(self.task_durations):
            current_latest_start = self.n_timeslots - self.task_durations[i]
            for j in range(self.n_operators):
                for t in range(current_latest_start + 1, self.n_timeslots):
                    no_do_if_no_finish_constraint_list.append(
                        self.start_variables_np[i, j, t] == 0
                    )
        return no_do_if_no_finish_constraint_list

    # def _get_finish_if_started_constraints(self):
    #     finish_if_started_constraint_list = []
    #     for i, current_task_duration in enumerate(self.task_durations):
    #         current_latest_start = self.n_timeslots - self.task_durations[i]
    #         for j in range(self.n_operators):
    #             for t_start in range(0, current_latest_start):
    #                 for t_running in range(t_start, t_start + current_task_duration + 1):
    #                     finish_if_started_constraint_list.append(
    #                         self.start_variables_np[i, j, t_start] == self.active_variables_np[i, j, t_running]
    #                     )

    def _get_one_task_simultaneous_constraints(self) -> List:
        one_task_simultaneous_list = []
        for j in range(self.n_operators):
            for t in range(self.n_timeslots):
                one_task_simultaneous_list.append(
                    lpSum([self.active_variables_np[:, j, t]]) <= 1
                )
        return one_task_simultaneous_list

    def set_problem(self):
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
        if not self.problem_is_set:
            logger.info("Problem was not set yet, setting now...")
            self.set_problem()
        logger.info("Solving problem...")
        self.pulp_problem.solve()

    def get_status(self) -> str:
        return LpStatus[self.pulp_problem.status]

    def get_objective_value(self) -> float:
        return self.objective.value()

    @staticmethod
    def _get_one_pulp_variable_value(pulp_variable):
        return pulp_variable.value()

    def get_solution(self):
        start_variable_values_np = np.vectorize(self._get_one_pulp_variable_value)(
            self.start_variables_np
        ).astype(int)
        task_started, on_machine, at_time = np.where(start_variable_values_np)

        solution = [
            (
                task_started[i],
                on_machine[i],
                (at_time[i], at_time[i] + self.task_durations[i]),
            )
            for i in range(len(task_started))
        ]
        return solution


@dataclass
class ScheduleGenerator(Generator):
    n_operators: int
    n_timeslots: int
    n_tasks: int
    min_task_duration: int
    max_task_duration: int
    min_block_duration: int
    max_block_duration: int

    def __post_init__(self):
        self.max_n_blocks = (
            int(self.n_timeslots / self.min_block_duration) + 1
        ) * self.n_operators

    def generate(self):
        self._generate_tasks()
        self._generate_available_timeslots()

    def _generate_tasks(self):
        allowed_durations = list(
            range(self.min_task_duration, self.max_task_duration + 1)
        )
        self.task_durations = np.random.choice(
            a=allowed_durations, size=self.n_tasks
        ).tolist()

    def _generate_available_timeslots(self):
        allowed_block_durations = list(
            range(self.min_block_duration, self.max_block_duration + 1)
        )
        block_durations = np.random.choice(
            a=allowed_block_durations, size=self.max_n_blocks
        ).tolist()

        availability_rows = []
        for i in range(self.n_operators):
            availability_row_np = np.zeros((self.n_timeslots,), dtype=int)
            goto_next_operator = False
            while not goto_next_operator:
                block_duration = block_durations.pop()
                allowed_start_times = self._allowed_start_times(
                    availability_row_np, block_duration
                )
                if allowed_start_times:
                    start_time = np.random.choice(a=allowed_start_times, size=1)[0]
                    availability_row_np[start_time : start_time + block_duration] = 1
                else:
                    availability_rows.append(availability_row_np.tolist())
                    goto_next_operator = True
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


if __name__ == "__main__":
    schedule_generator = ScheduleGenerator(
        n_operators=5,
        n_timeslots=10,
        n_tasks=7,
        min_task_duration=1,
        max_task_duration=6,
        min_block_duration=2,
        max_block_duration=4,
    )
    schedule_generator.generate()

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
