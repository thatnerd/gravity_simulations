#!/usr/bin/env python3
"""Setup script for gravity_simulations package."""

from setuptools import setup, find_packages

setup(
    name="gravity_simulations",
    version="0.1.0",
    description="Two-body gravitational simulation with RK4 integration",
    author="Will",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "docopt",
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "two_body_sim=script.two_body_sim:main",
        ],
    },
)
