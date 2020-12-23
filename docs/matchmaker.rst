Matchmaker
==========

Concept
-------
.. dropdown:: Concept

    The ``simpleor`` matchmaker problem solves a well known problem
    in literature generally referred to as 'maximum matching'.

    In a graph :math:`G = (V, E)` of Vertices and Edges, where each edge :math:`e \in E` has
    a weight :math:`w(e)`, the goal is to find pairs of nodes such that the sum of the weights
    of the chosen edges is maximized. A node can only be connected to one other node
    at most.

    If that sounded confusing, think of it this way:

    Suppose you are the owner of a dating platform. After a round of
    speed dating, every member of your community expresses how much
    they would like to see the other person again. For instance, if there
    are three community member Alberto, Britta, and Charly, then Alberto
    rates Britta and Charly, B rates A and C, and C rates A and B. This can
    also be represented with a matrix :math:`A`, where element
    :math:`A_{i,j}` corresponds to the rating person :math:`i`
    gives to person :math:`j`.

    For instance, suppose now

    .. math::
       A = \begin{bmatrix}
            0 & 6 & 1 & 2\\
            8 & 0 & 1 & 1\\
            0 & 2 & 0 & 10\\
            2 & 3 & 7 & 0\\
           \end{bmatrix}

    Our job is to make pairs of people (no triplets, no singletons)
    such that the community as a whole is as happy as possible.
    It is allowed to leave people without a pair.

    From :math:`A` we can see that person 1 really likes person 2 (rating of 6),
    but is not really into person 3 and 4 (rated 1 and 2).
    Also person 2 is into person 1 (rating 8). Similar story for
    the pair 3 and 4: they like each other, but not person 1 and 2.

    Also note that we put zeros on the diagonal to avoid pairing
    people with themselves.

    In this case, we would like to match person 1 to 2 and
    3 to 4. This may seem trivial in this example,
    but imagine having a community of 1000 people!

Usage
-----
.. dropdown:: Usage Solver

    Still on my todo list :)

.. dropdown:: Usage Generator

    Still on my todo list :)

Code
----
.. dropdown:: Code

    .. automodule:: simpleor.matchmaker
        :members:
