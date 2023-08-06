#!/usr/bin/env python
from setuptools import find_packages
from distutils.core import setup

setup(
    name="daisycc",
    version="0.0.1",
    description="Daisy Optimizing C/C++ Compiler Compiler based on LLVM and DaCe",
    long_description="Daisy Optimizing C/C++ Compiler Compiler based on LLVM and DaCe",
    author="Lukas Truemper",
    author_email="lukas.truemper@outlook.de",
    url="https://daisytuner.com",
    python_requires=">=3.8",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=["daisytuner>=0.2.1", "scop2sdfg>=0.0.1", "fire>=0.5.0"],
    extras_require={"dev": ["black==22.10.0", "pytest>=7.2.0", "pytest-cov>=4.1.0"]},
    entry_points={
        "console_scripts": [
            "daisycc = driver:main",
            "daisycc++ = driver:main",
        ]
    },
)
