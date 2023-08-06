<p align="center"><img src="figures/daisy.png" width="300"/></p>

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a> 
[![CI](https://github.com/daisytuner/daisycc/actions/workflows/tests.yml/badge.svg)](https://github.com/daisytuner/daisycc/actions/workflows/tests.yml)

daisycc is an optimizing C/C++ compiler collection based on the LLVM infrastructure and the DaCe framework. daisycc lifts loop nests to *stateful dataflow multigraphs (SDFG)* via polyhedral analysis and optimizes them using the [similarity-based transfer tuning algorithm](https://dl.acm.org/doi/abs/10.1145/3577193.3593714).

