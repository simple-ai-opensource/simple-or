# base.py

from abc import ABCMeta, abstractmethod


class Solver(metaclass=ABCMeta):
    @abstractmethod
    def validate_input(self):
        pass

    @abstractmethod
    def set_problem(self):
        pass

    @abstractmethod
    def _set_variables(self):
        pass

    @abstractmethod
    def _set_objective(self):
        pass

    @abstractmethod
    def _set_constraints(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def get_status(self):
        pass

    @abstractmethod
    def get_objective_value(self):
        pass

    @abstractmethod
    def get_solution(self):
        pass


class ProblemGenerator(metaclass=ABCMeta):
    @abstractmethod
    def generate(self):
        pass
