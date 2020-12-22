=====
Usage
=====

Scheduler
---------
With the scheduler you can optimize the assignment of tasks to agents.
All tasks have an integer duration, and all agents have an available schedule.
The scheduler tries to assign jobs to agents such that it maximizes the reward
of job execution, while respecting the schedule of the agents.

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

Luckily, we can use the ScheduleSolver. You can:

- import the ``ScheduleSolver`` in your own code
- use the command line to read problem description files and write to a solution file.

Import in sourcecode
~~~~~~~~~~~~~~~~~~~~
You can import the schedule solver as follows:

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

    schedule_solver.get_solution(kind="native")

    [
        (0, 0, 0, 2, 2),
        (1, 1, 2, 5, 3),
    ]

You can also use ``kind="dataframe"``, in which case you get back a pandas
dataframe.

Command line
~~~~~~~~~~~~
In case you do not want to program in Python, you can use the command line.

First, you need to create two files.

- task_durations.csv, which should be a list of the task durations (in one column). The task durations should be integer.
- available_schedule.csv, where every row corresponds to an agent. A row corresponds to an agent, a column to a period.
  A 1 indicates the agent is available in that timeslot, a 0 means not available.

You can store these files anywhere you like. Store the paths to these files
somewhere.

Next, open a terminal and type the following command (replace <TASK_DURATION_PATH>
and <AVAILABLE_SCHEDULE_PATH> with the paths you just stored):

.. code-block:: bash

    $ schedule --durationsfile <TASK_DURATION_PATH> --schedulefile <AVAILABLE_SCHEDULE_PATH>

By default, the solution will be stored in the data directory of the package. If you want
to store it somewhere else, add the following flag: ``--solutiondir <SOLUTION_DIRECTORY_PATH>``

By default, the name of the solution file is ``solution_cli.csv``. In case you want to
change it, add the flag `--solutionfile solution_cli`

Instead of csv, you can also use excel files. In that case, add the following
flag: ``--read excel`` or ``--write excel``


# TODO: matchmaker documentation!
