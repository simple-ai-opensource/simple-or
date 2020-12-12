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

Report bugs at https://github.com/lennartdamen/simpleor/issues.

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


4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the pre-commit hook and tests::

    $ pre-commit install
    $ pre-commit
    $ pytest
    $ tox


6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request TO THE DEVELOPMENT BRANCH through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.7 and 3.8. Check the Github Actions
   tab of the main simpleo-or repo and make sure that the tests of the development branch
   pass for all supported Python versions.

Your PR requires a review from one of the authors before it will be merged.
After merge to development, a workflow is triggered that will upload to the testing
branch and eventually to the master branch.

Tips
----

To run a subset of tests::

$ pytest tests.test_simpleor


Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run merge the development branch into testing. Code below assumes
you have set-up your terminal with the authentication for simple-ai-opensource,
and simple-or as the origin::

    $ git checkout master
    $ git fetch origin
    $ git branch -D testing
    $ git checkout -b testing
    $ git pull development
    $ bump2version patch # possible: major / minor / patch
    $ git push origin testing
    $ git push origin testing --tags

Github will then deploy to test PyPI if tests pass.

Then, check if everything works by installing from Test PyPI::

    $ pip install --extra-index-url https://testpypi.python.org/pypi simpleor

If all is well, merge the testing branch into master and deploy to PyPI.
