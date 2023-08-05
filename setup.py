#!/usr/bin/env python
from setuptools import find_packages
from distutils.core import setup

with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name="daisytuner-llvm",
    version="0.1.2",
    description="Daisytuner-llvm is a tool for lifting static-control-parts (Scop) to stateful dataflow multigraphs (SDFG).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Lukas Truemper",
    author_email="lukas.truemper@outlook.de",
    url="https://daisytuner.com",
    python_requires=">=3.8",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        "dace>=0.14.3",
        "daisytuner>=0.2.1",
        "islpy>=2023.1",
        "fire>=0.5.0",
    ],
    extras_require={"dev": ["black==22.10.0", "pytest>=7.2.0", "pytest-cov>=4.1.0"]},
    entry_points={
        "console_scripts": [
            "scop2sdfg = daisytuner_llvm.cli:main",
        ]
    },
)
