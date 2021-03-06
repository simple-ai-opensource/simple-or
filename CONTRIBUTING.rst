.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/simple-ai-opensource/simple-or/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

simpleor could always use more documentation, whether as part of the
official simpleor docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/simple-ai-opensource/simple-or/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `simpleor` for local development.

1. Fork the `simpleor` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/simple-or.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv simpleor
    $ cd simpleor/
    $ pip install -r requirements.txt
    $ pip install -e .[dev]
    $ pre-commit install

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

If you are starting on a new problem, make sure that your Generator or Solver
inherits from the base Generator or Solver. As an example, see the implementation
of ScheduleSolver.

5. When you're done making changes, check that your changes pass the pre-commit hook and tests::

    $ pytest
    $ tox

6. Now update the version with bump2version. In general you want to add the 'patch' tag,
because you should send small PR's. You can choose from 'patch', 'minor', and 'major'::

    $ bump2version patch

7. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.7 and 3.8. Check the Github Actions
   tab of the main simple-or repo and make sure that the tests of the development branch
   pass for all supported Python versions.
4. Did you increase the version number?

Your PR requires a review from one of the authors before it will be merged.
Once your PR is approved, it is merged to master branch and push automatically to
Test PyPI. An author will verify the Test PyPI installation and manually trigger
a workflow that sends the new changes to the official PyPI.

Tips
----

To run a subset of tests::

$ pytest tests.test_scheduler.py


Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
The code has already been push to Test PyPI, so check that the installation does not
raise any errors::

    $ pip install --extra-index-url https://testpypi.python.org/pypi simpleor

If all is well, go to the 'Actions' tab and manually run the worfklow 'master_code_to_pypi.yml'.
