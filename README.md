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


### Creating Chromossomes

There are three types of genes that can be used for optimization, the `BinaryGene`, the `IntegerGene` and the `FloatGene`.


#### `BinaryGene`

A `BinaryGene` contains simply a boolean value. It can be created in two ways:

```julia
# option 1
gene = BinaryGene(true)
# option 2
gene = BinaryGene()
```

The first way sets the initial value to `true`, while the second case sets the initial value to a random boolean value.


#### `IntegerGene`

A `IntegerGene` is a container for a `BitVector` that is used to determine the integer number using base-2 binary arithmetic. To create a `IntegerGene`:

```julia
# option 1
gene = IntegerGene(BitVector(undef, 3), :FM)
# option 2
gene = IntegerGene(BitVector(undef, 3))
# option 3
gene = IntegerGene(3)

int_number = bin(gene)
```

The first one sets a `BitVector` with three elements and sets the Flip Mutation as the mutation algorithm for this gene. The second one sets the Flip Mutation as the default mutation algorithm. The last creates a random `BitVector` with length 3 and the Flip Mutation as the default mutation algorithm. To know all the mutation algorithms implemented and their corresponding symbols, as well as more information about these functions, just type `?IntegerGene` in the command prompt. The function `bin` is a function created to convert the `BitVector` into an integer number using base-2 binary arithmetic.


#### `FloatGene`

A `FloatGene` is a gene that is comprised by real numbers. Can have one or more values, so you can combine all real valued variables in a single `FloatGene`. There are several ways to create a `FloatGene`, so let's name a few:

```julia
# option 1
gene = FloatGene(values, ranges; m ::Int = 20)
# option 2
gene = FloatGene(values, range; m ::Int = 20)
# option 3
gene = FloatGene(value, range; m ::Int = 20)
# option 4
gene = FloatGene(values; m ::Int = 20)
# option 5
gene = FloatGene(value; m ::Int = 20)
# option 6
gene = FloatGene(n)
```

Plural `values` and `ranges` mean vector, while singular `value` and `range` mean scalar. In options 4 and 5, both `range` and `ranges` are created randomly, according to the size of `value` or `values`. Option 6 creates random variables and random ranges of size `n`.


### Running the Genetic Algorithm

Now that we know how to create genes, we need to create a population and an objective function to run our genetic algorithm.

#### Example 1

In this example we want to find the index of a vector associated to the midpoint of said vector. Given:

```julia
x = 0.0:0.01:10
```

We want to find the index that corresponds to the value `5.0`. First let's create our chromossome, which in this case will comprise of one `IntegerGene`:

```julia
gene = IntegerGene(4)
chromossome = [gene]
```

**NOTE:** when dealing with integer numbers, make sure the length of your vector is big enough to embrace the possible value. In this case, for a vector of length 4, the maximum integer value is 16, so our expected result can be represented by this vector.

Our objective function could be something like:

```julia
function objfun(chrom ::Vector{<:AbstractGene})
	ind = bin(chrom[1])
	return abs( x[ind] - (x[end]-x[1]) / 2 )
end
```

Now we have to choose the crossover and selection algorithms:

```julia
Crossover(:SPX) # single point crossover
Selection(:RWS) # roulette wheel selection
```

And now we can run our genetic algorithm:

```julia
N = 100 # population size
ga(objfun, chromossome, N)
```

If you couldn't get the right result, you can increase the population size or increase the number of iterations. The optional arguments are well explained if you go to the command prompt and type `?ga`.


#### Example 2

Using the same vector `x`, now we want to determine the midpoint of said vector, which will be a real number. In that case:

```julia
gene = FloatGene(1.0) # random range value
chromossome = [gene]
```

Now our objective function will be slightly different:

```julia
function objfun(chrom ::Vector{<:AbstractGene})
	return abs(chrom.value[1] - (x[end] - x[1]) / 2)
end
```

After choosing crossover and selection algorithms:

```julia
Crossover(:SPX) # single point crossover
Selection(:RWS) # roulette wheel selection
```

We run the genetic algorithm in the same way:

```julia
N = 100
ga(objfun, chromossome, N)
```


### Algorithms

- (μ/ρ(+/,)λ)-SA-ES
- (μ/μ_I,λ)-CMA-ES
- Genetic Algorithms (GA)

### Operators

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
	- truncation


## TODO
* Documentation
* Concurrent implementation

## Resources
- **Documentation:** <http://evolutionaryjl.readthedocs.org/en/latest/index.html>
