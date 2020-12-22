#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["Click>=7.0", "numpy", "pulp", "pandas"]
test_requirements = ["pytest>=3", "pytest-cov"]
development_requirements = [
    "black",
    "flake8",
    "pre-commit",
    "mypy",
    "isort",
    "bump2version",
]

setup(
    author="Lennart Damen",
    author_email="lennart.damen.ai@gmail.com",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Package to solve Operations Research problems.",
    entry_points={"console_scripts": ["schedule=simpleor.cli:schedule"]},
    install_requires=requirements,
    tests_require=test_requirements,
    extras_require={
        "test": test_requirements,
        "dev": test_requirements + development_requirements,
    },
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="simpleor",
    name="simpleor",
    packages=find_packages(include=["simpleor", "simpleor.*"]),
    url="https://github.com/simple-ai-opensource/simple-or",
    version="0.0.6",
    zip_safe=False,
)
