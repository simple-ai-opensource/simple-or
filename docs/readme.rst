.. include:: ../README.rst

Scheduler
---------
In the ``simpleor`` scheduling problem, we try to assign
tasks to agents such that the reward for completing those tasks
is as high as possible.

Each task is defined by:

- the time it takes to complete it
- the reward you get for completing it.

Each agent is defined by the timeslots that it is available.

MatchMaker
----------
In the ``simpleor`` matchmaker problem, we try to find pairs
of nodes such that the reward of assigning those pairs is as high
as possible. The reward of choosing a pair is defined by the value
corresponding to the edge between two nodes. A node can only be
paired up with one other node at most.
