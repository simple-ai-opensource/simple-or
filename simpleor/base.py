# base.py

from abc import ABCMeta, abstractmethod


class Solver(metaclass=ABCMeta):
    @abstractmethod
    def validate_input(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_problem(self, *args, **kwargs):
        pass

    @abstractmethod
    def _set_variables(self, *args, **kwargs):
        pass

    @abstractmethod
    def _set_objective(self, *args, **kwargs):
        pass

    @abstractmethod
    def _set_constraints(self, *args, **kwargs):
        pass

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_status(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_objective_value(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_solution(self, *args, **kwargs):
        pass


class Generator(metaclass=ABCMeta):
    @abstractmethod
    def generate(self):
        pass
