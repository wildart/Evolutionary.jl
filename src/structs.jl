##### structs.jl #####

# In this file you will find all the structures needed to generalize code.
# You will find also some functions to help create the global structures.

####################################################################

export BinaryGene, IntegerGene, FloatGene
export Crossover, Selection
export selection, crossover, bin
export GAExternal, ext_init, distributed_ga

####################################################################

"""
    BinaryGene(value ::Bool, name ::AbstractString)

Creates a `BinaryGene` structure. This gene represents a `Bool` variable (1 or 0). Useful to make decisions (will a reactor work or not, will a SMB pathway be used or not, etc.). `name` is the variable name for presentation purposes.
"""
mutable struct BinaryGene <: AbstractGene
    value ::Bool
    name  ::AbstractString
    
    function BinaryGene(value ::Bool, name ::AbstractString)
        return new(value, name)
    end
end

"""
    BinaryGene()

Creates a `BinaryGene` structure with a random `Bool` value.
"""
BinaryGene(name ::AbstractString) = BinaryGene(rand(Bool), name)

"""
    BinaryGene()

Creates a `BinaryGene` structure with a random `Bool` value and a default variable name.
"""
BinaryGene() = BinaryGene(rand(Bool), "bin")

####################################################################

"""
    IntegerGene(value ::BitVector, name ::AbstractString)

Creates a `IntegerGene` structure. This gene represents an integer variable as `BitVector`. To convert the `BitVector` in an integer, just look at the `bin` function from this package by typing `?bin` on the command prompt. `name` is a string that represents the name of the variable. It's needed for result presentation purposes.
"""
mutable struct IntegerGene <: AbstractGene
    value  ::BitVector
    lbound ::Real
    ubound ::Real
    name   ::AbstractString
end

"""
    IntegerGene(mutation ::Symbol)

Chooses the mutation type. Does NOT create a structure. This dispatch was created because, when creating a population of integer genes, each gene would create this function, which was unnecessary work. You still need to run one of the other dispatches to create the gene.

Below are the types of mutations supported:

| Symbol | Algorithm          |
|--------|--------------------|
| :FM    | Flip Mutation      |
| :InvM  | Inversion Mutation |
| :InsM  | Insertion Mutation |
| :SwM   | Swap Mutation      |
| :ScrM  | Scramble Mutation  |
| :ShM   | Shifting Mutation  |
"""
function IntegerGene(mutation ::Symbol)
    int_func = mutate(mutation)
    @eval mutate(gene ::IntegerGene) = $int_func(gene.value)
    return nothing
end

"""
    IntegerGene(n ::Int64)

Creates a `IntegerGene` structure in which the `BitVector` is of length `n` with random `Bool` values.
"""
function IntegerGene(n ::Int64, name ::AbstractString;
                     lb ::Real = -Inf, ub ::Real = Inf )
    value  = BitVector(undef, n)
    for i in 1:n
        @inbounds value[i] = rand(Bool)
    end
    return IntegerGene(value, lb, ub, name)
end

"""
    bin(gene ::IntegerGene)

Returns the integer number represented by the `BitVector` of `gene`.
"""
function bin(gene ::IntegerGene)
    bin_val = 0
    for (i,j) in enumerate(gene.value)
        bin_val += j ? 2^(i-1) : 0
    end
    return bin_val
end

####################################################################

"""
    FloatGene(value ::Vector{Float64}, range ::Vector{Float64}, m ::Int64)

Creates a `FloatGene` structure. `value` is a vector with the variables to be changed. `range` is a vector with the minimum and maximum values a variable can take. `m` is just a parameter that changes how much in a mutation the variables change, the bigger the value, the bigger the change in each mutation. If the range of a variable is 0.5, then the biggest mutation a variable can suffer in one iteration is 0.5, for instance.
"""
mutable struct FloatGene <: AbstractGene
    value  ::Vector{Float64}
    range  ::Vector{Float64}
    name   ::Vector{AbstractString}
    m      ::Int64
    lbound ::Vector{<:Real}
    ubound ::Vector{<:Real}
    
    function FloatGene( value  ::Vector{Float64}           ,
                        range  ::Vector{Float64}           ,
                        name   ::Vector{String}            ,
                        m      ::Int64                     ,
                        lbound ::Union{Vector{<:Real},Real},
                        ubound ::Union{Vector{<:Real},Real})
        if length(value) != length(range)
            error("value and range have different lengths")
        end
        if typeof(lbound) <: Real
            lb = Float64[lbound for i in value]
        else
            if length(value) != length(lbound)
                error("value and lbound have different lengths")
            else
                lb = lbound
            end
        end
        if typeof(ubound) <: Real
            ub = Float64[ubound for i in value]
        else
            if length(value) != length(ubound)
                error("value and ubound have different lengths")
            else
                ub = ubound
            end
        end
        return new(value, range, name, m, lb, ub)
    end
end

function FloatGene( value ::Vector{Float64}                  ,
                    range ::Vector{Float64}                  ,
                    name  ::Vector{String}                   ;
                    m     ::Int64                      =  20 ,
                    lb    ::Union{Vector{<:Real},Real} = -Inf,
                    ub    ::Union{Vector{<:Real},Real} =  Inf)
    
    return FloatGene(value, range, name, m, lb, ub)
end

"""
    FloatGene(value ::Float64, range ::Float64; m ::Int64 = 20)

Creates a `FloatGene` structure. Handy for creating just one real number variable.
"""
function FloatGene( value ::Float64                          ,
                    range ::Float64                          ,
                    name  ::AbstractString                   ;
                    m     ::Int64                      =  20 ,
                    lb    ::Union{Vector{<:Real},Real} = -Inf,
                    ub    ::Union{Vector{<:Real},Real} =  Inf)
    return FloatGene([value], [range], [name], m, lb, ub)
end

"""
    FloatGene(value ::Vector{Float64}, range ::Float64; m ::Int64 = 20)

Creates a `FloatGene` structure. Handy for creating a vector of real numbers with the same range.
"""
function FloatGene( value ::Vector{Float64}                  ,
                    range ::Float64                          ,
                    name  ::Vector{String}                   ;
                    m     ::Int64                      =  20 ,
                    lb    ::Union{Vector{<:Real},Real} = -Inf,
                    ub    ::Union{Vector{<:Real},Real} =  Inf)
    range_vec = Float64[range for i in value]
    return FloatGene(value, range_vec, name, m, lb, ub)
end

"""
    FloatGene(n ::Int64)

Creates a `FloatGene` structure. Creates a vector of length `n` with random variables and random ranges. Used particularly for testing purposes.
"""
function FloatGene(n ::Int64, name ::AbstractString; m ::Int64 = 20)
    value = rand(Float64, n)
    range = rand(Float64, n)
    lb    = rand(Float64, n)
    ub    = rand(Float64, n)
    vec_name = Vector{String}(undef, n)
    for i in 1:n
        vec_name[i] = string(name, i)
    end
    return FloatGene(value, range, vec_name, m, lb, ub)
end

####################################################################

function isbound(gene ::FloatGene)
    lb = findmin( gene.value  .- gene.lbound )[1]
    ub = findmin( gene.ubound .- gene.value  )[1]
    if lb < 0.0
        return false
    end
    if ub < 0.0
        return false
    end
    return true
end

function isbound(gene ::IntegerGene)
    return bin(gene) >= gene.lbound && bin(gene) <= gene.ubound
end

isbound(gene ::BinaryGene) = true

####################################################################

"""
    Crossover(cross ::Symbol                                    ;
              w     ::Union{Nothing, Vector{Float64}} = nothing ,
              d     ::Union{Nothing, Float64        } = nothing )

Creates a `Crossover` structure. `cross` is a Symbol that represents the type of crossover that would be used. `w` and `d` are not mandatory but need to be set for some types of crossovers. All algorithms will be shown in the table below:

| Symbol | Algorithm                            | Optional Arguments |
|--------|--------------------------------------|--------------------|
| :SPX   | Single Point Crossover               | not needed         |
| :TPX   | Two Point Crossover                  | not needed         |
| :UX    | Uniform Crossover                    | not needed         |
| :DX    | Discrete Crossover                   | not needed         |
| :WMX   | Weighted Mean Crossover              | needs `w`          |
| :IRX   | Intermediate Recombination Crossover | needs `d`          |
| :LRX   | Line Recombination Crossover         | needs `d`          |
| :PMX   | Partially Mapped Crossover           | not needed         |
| :O1X   | Order 1 Crossover                    | not needed         |
| :O2X   | Order 2 Crossover                    | not needed         |
| :CX    | Cycle Crossover                      | not needed         |
| :PX    | Position-based Crossover             | not needed         |
"""
mutable struct Crossover
    cross ::Symbol
    w     ::Union{Nothing, Vector{Float64}}
    d     ::Union{Nothing, Float64        }
    
    function Crossover(cross ::Symbol                                    ;
                       w     ::Union{Nothing, Vector{Float64}} = nothing ,
                       d     ::Union{Nothing, Float64        } = nothing )
            cross_func = nothing
            if cross == :SPX
                cross_func = singlepoint
            elseif cross == :TPX
                cross_func = twopoint
            elseif cross == :UX
                cross_func = uniform
            elseif cross == :DX
                cross_func = discrete
            elseif cross == :WMX
                if isnothing(w)
                    error("value `w` must be given a value")
                end
                cross_func = waverage(w)
            elseif cross == :IRX
                if isnothing(d)
                    error("value `d` must be given a value")
                end
                cross_func = intermediate(d)
            elseif cross == :LRX
                if isnothing(d)
                    error("value `d` must be given a value")
                end
                cross_func = line(d)
            elseif cross == :PMX
                cross_func = pmx
            elseif cross == :O1X
                cross_func = ox1
            elseif cross == :O2X
                cross_func = ox2
            elseif cross == :CX
                cross_func = cx
            elseif cross == :PX
                cross_func = pos
            end
        
            @eval begin
                function crossover(v1 ::T, v2 ::T) where {T <: AbstractVector}
                    return $cross_func(v1, v2)
                end
            end
        return new(cross, w, d)
    end
end

####################################################################

"""
    Selection( select    ::Symbol                            ;
               sp        ::Union{Nothing, Float64} = nothing ,
               μ         ::Union{Nothing,   Int64} = nothing ,
               groupsize ::Union{Nothing,   Int64} = nothing )

Creates a `Selection` structure. `select` is a symbol that represents the type of selection that will be used. `sp`, `μ` and `groupsize` are optional but need to be set for some types of selections. All algorithms will be shown in the table below:

| Symbol | Algorithm                               | Optional Arguments |
|--------|-----------------------------------------|--------------------|
| :RBS   | Rank-based Selection                    | needs `sp`         |
| :URS   | Uniform-Ranking Selection               | needs `μ`          |
| :RWS   | Roulette Wheel Selection                | not needed         |
| :SUSS  | Stochastic Universal Sampling Selection | not needed         |
| :TrS   | Truncation Selection                    | not needed         |
| :ToS   | Tournament Selection                    | needs `groupsize`  |
"""
mutable struct Selection
    select ::Symbol
    sp     ::Union{Nothing, Float64}
    μ      ::Union{Nothing,   Int64}
    gsize  ::Union{Nothing,   Int64}
    
    function Selection( select    ::Symbol                            ;
                        sp        ::Union{Nothing, Float64} = nothing ,
                        μ         ::Union{Nothing,   Int64} = nothing ,
                        groupsize ::Union{Nothing,   Int64} = nothing )
        selec_func = nothing
        if select == :RBS
            if isnothing(sp)
                error("need to specify `sp` value")
            end
            selec_func = ranklinear(sp)
        elseif select == :URS
            if isnothing(μ)
                error("need to specify `μ` value")
            end
            selec_func = uniformranking(μ)
        elseif select == :RWS
            selec_func = roulette
        elseif select == :SUSS
            selec_func = sus
        elseif select == :TrS
            selec_func = truncation
        elseif select == :ToS
            if isnothing(groupsize)
                error("need to specify `groupsize` value")
            end
            selec_func = tournament(groupsize)
        else
            error("Unknown parameter " * string(select))
        end
        
        @eval begin
            function selection(fit ::Vector{<:Real}, N ::Int)
                return $selec_func(fit, N)
            end
        end
        return new(select, sp, μ, groupsize)
    end
end

####################################################################

"""
    function GAExternal( program  ::AbstractString                   ,
                         pipename ::AbstractString                   ;
                         nworkers ::Int64          = Sys.CPU_THREADS ,
                         parallel ::Bool           = false           )

Creates communication pipes for the external program `program`. If `parallel` is `true`, then, considering N workers available, N pipes for reading and N pipes for writing will be created. `pipename` is just a handle for the name of the pipes. If `pipename` is `pipe`, then the pipe names will be `pipe_in` and `pipe_out` if `parallel=false` and `pipe_inn` and `pipe_outn` if `parallel=true`, with `n` being one of the N workers. `nworkers` is the number of cores to be used, including the number of cores of a remote computer. `parallel` sets the the external program to run in several workers.
"""
mutable struct GAExternal
    program       ::AbstractString
    pipes_in      ::Union{Vector{<:AbstractString},DArray{String,1,Vector{String}}}
    pipes_out     ::Union{Vector{<:AbstractString},DArray{String,1,Vector{String}}}
    avail_workers ::Vector{Int64}
    parallel      ::Bool

    function GAExternal( program  ::AbstractString                   ,
                         pipename ::AbstractString                   ;
                         nworkers ::Int64          = Sys.CPU_THREADS ,
                         parallel ::Bool           = false           )
        pipes = Dict{String,Vector{String}}()
        pipes["in" ] = Vector{String}(undef, 0)
        pipes["out"] = Vector{String}(undef, 0)
        avail_workers = workers()[1:nworkers]
        if parallel
            # create one pipe for reading and another for writing
            # for each worker
            for i in ["in","out"]
                for p in avail_workers
                    f = string(pipename, "_", i, p)
                    push!(pipes[i], f)
                    remotecall_fetch(rm, p, f, force=true)
                    remotecall_fetch(run, p, `mkfifo $f`)
                end
            end
            pin  = distribute(pipes["in" ]; procs=avail_workers)
            pout = distribute(pipes["out"]; procs=avail_workers)
        else
            # create one pipe for reading and another for writing
            for i in ["in","out"]
                f = string(pipename, "_", i)
                push!(pipes[i], f)
                rm(f, force=true)
                run(`mkfifo $f`)
            end
            pin = pipes["in"]
            pout = pipes["out"]
        end

        #pin  = distribute(pipes["in" ]; procs=avail_workers)
        #pout = distribute(pipes["out"]; procs=avail_workers)

        # activate writing pipes for a big amount of time
        for (i,p) in enumerate(pin)
            id  = 1*nworkers+1 + i
            id1 = 2*nworkers+1 + i
            @spawnat id  run(pipeline(`sleep 100000000`; stdout=p))
            @spawnat id1 run(pipeline(`$program`; stdin=p))
        end

        # Open reading pipes in separate processes.
        # The pipes have to be open before writing to them,
        # otherwise we get SIGPIPE errors when writing.
        function spawn_readpipes(pipe)
            f = open(pipe, "r")
            return nothing
        end
        for (i,p) in enumerate(pout)
            v = 3*nworkers+1 + i
            @spawnat v spawn_readpipes(p)
        end

        # delete all pipes when exiting julia
        function external_atexit()
            for k in keys(pipes)
                for (i,p) in enumerate(pipes[k])
                    id = i+1
                    remotecall_fetch(rm, id, p)
                end
            end
            return nothing
        end
        atexit(external_atexit)

        return new(program, pin, pout, avail_workers, parallel)
    end
end

####################################################################

"""
    ext_init(gaext ::GAExternal, codeline ::AbstractString)

Function to help initialize the external program.

When using external programs, you usually have an initial script to be run before performing the genetic algorithm. After creating the `GAExternal` structure, just send it to this function along with the line of code to initialize the optimization process.

## Example

In case of AMPL, if we have a script with the initial model called `init.ampl`, then we could do:

```
ext = GAExternal("ampl", "pipe")
ext_init(ext, "include init.ampl;")
```
"""
function ext_init(gaext ::GAExternal, codeline ::AbstractString)
    function ext(gaext ::GAExternal)
        if gaext.parallel
            pin = gaext.pipes_in[:L]
        else
            pin = gaext.pipes_in
        end
        for p in pin
            open(p, "w") do file
                write(file, codeline)
            end
        end
    end
    if gaext.parallel
        spmd(ext, gaext, pids=gaext.avail_workers) 
    else
        ext(gaext)
    end
    return nothing           
end

####################################################################

"""
    distributed_ga( ;
                    localcpu ::Int64           = Sys.CPU_THREADS ,
                    cluster  ::Vector{<:Tuple} = [()]            ,
                    dir      ::AbstractString  = pwd()           )

Function to help set up the local computer or a cluster for parallel run of the Genetic Algorithm. `localcpu` is the number of cores to be used in your local computer. `cluster` is a vector of machine specifications. To know more about this, type `?addprocs` in the command prompt after importing the `Distributed` package. `dir` is the directory where you want julia to run in each remote computer.
"""
function distributed_ga( ;
                         localcpu ::Int64           = Sys.CPU_THREADS ,
                         cluster  ::Vector{<:Tuple} = [()]            ,
                         dir      ::AbstractString  = pwd()           )

    nworkers = localcpu
    if cluster != [()]
        for i in cluster
            nworkers += i[2]
        end
    end
        
    @eval using Distributed
    for i in 1:4
        if localcpu != 0
            addprocs(localcpu)
        end
        if cluster != [()]
            addprocs(cluster, dir=dir)
        end
    end

    @eval @everywhere using Distributed
    @eval @everywhere using DistributedArrays
    @eval @everywhere using DistributedArrays.SPMD
    @eval @everywhere using Evolutionary

    return nworkers
end
