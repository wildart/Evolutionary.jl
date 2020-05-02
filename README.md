# Evolutionary

A Julia package for [evolutionary](http://www.scholarpedia.org/article/Evolution_strategies) & [genetic](http://en.wikipedia.org/wiki/Genetic_algorithm) algorithms.

| **Documentation**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][travis-img]][travis-url] [![][coverage-img]][coverage-url] |


## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add Evolutionary
```

## Algorithms

- (μ/ρ(+/,)λ)-SA-ES
- (μ/μ<sub>I</sub>,λ)-CMA-ES
- Genetic Algorithms (GA)
- Differential Evolution (DE)

## Operators

- Mutations
  - ES
    - (an)isotropic Gaussian
    - (an)isotropic Cauchy
  - GA
    - binary
      - flip
      - inversion
    - real valued
      - uniform
      - gaussian
      - BGA
    - combinatorial
      - inversion
      - insertion
      - swap2
      - scramble
      - shifting

- Recombinations
  - average
  - marriage

- Crossovers
  - binary
    - N-point
    - uniform
  - real valued
    - discrete
    - weighted average
    - intermediate
    - line
  - permutation
    - PMX
    - OX1
    - OX2
    - CX
    - POS

- Selections
  - rank-based fitness assignment
  - (μ, λ)-uniform ranking
  - roulette (fitness proportionate selection, FSP)
  - stochastic universal sampling (SUS)
  - tournament
  - truncation


[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://wildart.github.io/Evolutionary.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://wildart.github.io/Evolutionary.jl/stable

[travis-img]: https://travis-ci.org/wildart/Evolutionary.jl.svg?branch=master
[travis-url]: https://travis-ci.org/wildart/Evolutionary.jl

[coverage-img]: https://img.shields.io/coveralls/wildart/Evolutionary.jl.svg
[coverage-url]: https://coveralls.io/r/wildart/Evolutionary.jl?branch=master

[issues-url]: https://github.com/wildart/Evolutionary.jl/issues
