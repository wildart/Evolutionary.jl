# Evolutionary

A Julia package for [evolutionary](http://www.scholarpedia.org/article/Evolution_strategies) & [genetic](http://en.wikipedia.org/wiki/Genetic_algorithm) algorithms.

[![Build Status](https://travis-ci.org/wildart/Evolutionary.jl.svg?branch=master)](https://travis-ci.org/wildart/Evolutionary.jl)
[![Coverage Status](https://img.shields.io/coveralls/wildart/Evolutionary.jl.svg)](https://coveralls.io/r/wildart/Evolutionary.jl?branch=master)
[![Documentation Status](https://readthedocs.org/projects/evolutionaryjl/badge/?version=latest)](https://readthedocs.org/projects/evolutionaryjl/?badge=latest)

## Installation

For julia 0.6 and lower, run following command

```julia
Pkg.add("Evolutionary")
```

For julia 0.7 and higher, run in the package manager mode
```
pkg> add https://github.com/wildart/Evolutionary.jl.git#v0.2.0
```

## Functionalities

#### Algorithms

- (μ/ρ(+/,)λ)-SA-ES
- (μ/μ_I,λ)-CMA-ES
- Genetic Algorithms (GA)

#### Operators

- Mutations
    - (an)isotropic mutation (for ES)
    - binary flip
    - real valued
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
	- roulette
	- stochastic universal sampling (SUS)
	- tournament


## TODO
* Documentation
* Concurrent implementation

## Resources
- **Documentation:** <http://evolutionaryjl.readthedocs.org/en/latest/index.html>
