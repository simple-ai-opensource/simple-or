=====
Usage
=====

Scheduler
---------
Currently, simpleor only has a scheduler, but this will be
extended in the future.

With the scheduler you can optimize the assignment of tasks to agents.
All tasks have an integer duration, and all agents have an available schedule.
The scheduler tries to assign as many jobs to agents as possible while respecting
the schedule of the agents.

For example, suppose we have two tasks with a duration of 2 and 3.

.. code-block:: python

    task_durations = [2, 3]

We also have two agents, which operate in the time interval [1, 2, ..., 5].
Agent 1 is available during the first two timeslots, agent 2 during the last three.

.. code-block:: python

    available_timeslots = [
        [True, True, False, False, False],
        [False, False, True, True, True],
    ]

We can immediately see that the optimal assignment is to assign the start of task 1 to
agent 1 in timeslot 1, and task 2 to operator 2 in timeslot 3. This may seem
trivial, but imagine having 1000 tasks and 100 agents!

Luckily, we can use the ScheduleSolver:

.. code-block:: python

    from simpleor.scheduler import ScheduleSolver

    schedule_solver = ScheduleSolver(
        task_durations=task_durations,
        available_timeslots=available_timeslots,
    )


Since we are now in programming language, we start counting from 0, not 1.
Calling .solve() on the ScheduleSolver instance will run an Integer Linear
Programming solver on the problem description.

.. code-block:: python

    schedule_solver.solve()

Now we can inspect the solution. The inspection returns a list.
Each element of the list is an assigned task, with the following fields:
task_id, agent_id, start_time, stop_time, task_duration

.. code-block:: python

    schedule_solver.get_solution()

    [
        (0, 0, 0, 2, 2),
        (1, 1, 2, 5, 3),
    ]

TODO: command line and generator documentation
