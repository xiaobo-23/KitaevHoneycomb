# ![Kitaev.jl]
<!-- [![Docs dev](https://img.shields.io/badge/docs-latest-blue.svg)](https://lukas.weber.science/Carlo.jl/dev/)
[![Docs stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lukas.weber.science/Carlo.jl/stable/)
[![CI](https://github.com/lukas-weber/Carlo.jl/workflows/CI/badge.svg)](https://github.com/lukas-weber/Carlo.jl/actions)
[![codecov](https://codecov.io/gh/lukas-weber/Carlo.jl/branch/main/graph/badge.svg?token=AI8CPOGKXF)](https://codecov.io/gh/lukas-weber/Carlo.jl) -->

Kitaev.jl is a high-performance framework for tensor network simulations of the Kitaev honeycomb model and its variants (based on [ITensors.jl](https://docs.itensor.org/ITensors/stable/)). The package includes the following key implementations: 

* map narrow cylinders to matrix product states (MPS)
* including the anisotropic Kitaev interaction, Heisenberg interaction, \gamma term, nonmagnetic spin vacancy
* implement tailored perturbations, pinning fields to help achieving convergence.


It is an actively evolving project with additional features under development.

![HoneycombLattice](Presentation_Fig1c.png)

## Getting started

To install the package, type

```julia
using Pkg; Pkg.add("Kitaev")
```

<!-- The package itself does not include Monte Carlo algorithms. The quickest way to see how to implement one yourself is to check out the reference implementation for the [Ising](https://github.com/lukas-weber/Ising.jl) model.
For a state-of-the-art Monte Carlo code, take a look at [StochasticSeriesExpansion.jl](https://github.com/lukas-weber/StochasticSeriesExpansion.jl). -->