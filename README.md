<p align="center"><img src="figures/daisy.png" width="300"/></p>

[![Get it from the Snap Store](https://snapcraft.io/static/images/badges/en/snap-store-white.svg)](https://snapcraft.io/daisycc)

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a> 
[![CI](https://github.com/daisytuner/daisycc/actions/workflows/tests.yml/badge.svg)](https://github.com/daisytuner/daisycc/actions/workflows/tests.yml)

daisycc is an optimizing C/C++ compiler collection based on the LLVM infrastructure and the DaCe framework.
daisycc lifts loop nests to *stateful dataflow multigraphs (SDFG)* via polyhedral analysis and optimizes them using similarity-based transfer tuning (https://dl.acm.org/doi/abs/10.1145/3577193.3593714).

**Features:**
- Clang-based C/C++ compilers
- Automatic Optimization for high-performance Intel and AMD CPUs
- Automatic Optimization for NVIDIA GPUs
- Polyhedral Analysis
- Data-Centric Optimization
- Similarity-based Transfer Tuning
