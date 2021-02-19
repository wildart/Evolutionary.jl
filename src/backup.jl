##### backup.jl #####

# In this file you'll find backup and reverse backup strategies for the
# Genetic Algorithm. 

####################################################################

export backup, reverse_backup

####################################################################

"""
    backup(f ::IOStream, gene ::IntegerGene)

Writes the current state of a `IntegerGene` structure to a buffer.
"""
function backup(f ::IOStream, gene ::IntegerGene)
    write(f, 'I')
    write(f, Int8(length(gene.value)))
    for i in gene.value
        write(f, i)
    end
    write(f, Int8(-length(gene.name)), gene.name)
    return nothing
end

"""
    backup(f ::IOStream, gene ::FloatGene)

Writes the current state of a `FloatGene` structure to a buffer.
"""
function backup(f ::IOStream, gene ::FloatGene)
    write(f, 'F')
    write(f, gene.m)
    write(f, Int8(length(gene.value)))
    for i in gene.value
        write(f, i)
    end
    for i in gene.range
        write(f, i)
    end
    for i in gene.name
        l = Int8(-length(i))
        write(f, l, i)
    end
    return nothing
end

"""
    backup(f ::IOStream, gene ::BinaryGene)

Writes the current state of a `BinaryGene` structure to a buffer.
"""
function backup(f ::IOStream, gene ::BinaryGene)
    write(f, 'B')
    write(f, gene.value)
    write(f, Int8(-length(gene.name)), gene.name)
    return nothing
end

"""
    backup( ngens ::Int64              ,
            tgens ::Int64              ,
            chrom ::Vector{Individual} ,
            file  ::AbstractString     )

Writes number of generations `ngens`, total numberof generations `tgens` and the population `chrom` into file `file`. Always writes to folder `backup-files` that is created, if inexistent, inside the `ga` function.
"""
function backup( ngens ::Int64              ,
                 tgens ::Int64              ,
                 chrom ::Vector{Individual} ,
                 file  ::AbstractString     )
    file = "backup-files/$file"
    chromossome = chrom
    psize = length(chromossome   )
    gsize = length(chromossome[1])
    open(file, "w") do f
        write(f, ngens, tgens, psize, gsize)
        for i in chromossome
            for j in i
                backup(f, j)
            end
        end
    end
    return nothing
end

####################################################################

"""
    reverse_backup(file ::AbstractString)

Reads backup file `file` and saves the number of generations run, the total number of generations supposed to run and the population into variables for later continuing the Genetic Algorithm.
"""
function reverse_backup(file ::AbstractString)
    f = open(file, "r")
    
    ngens    = read(f, Int64)
    tgens    = read(f, Int64)
    popsize  = read(f, Int64)
    genesize = read(f, Int64)
    population = Vector{Individual}(undef, popsize)    
    for p in 1:popsize
        population[p] = Individual(undef, genesize)
        for g in 1:genesize
            id = read(f, Char)
            if id == 'I'
                nvals = read(f, Int8)
                bit_vec = BitVector(undef, nvals)
                readbytes!(f, reinterpret(UInt8, bit_vec))
                strsize = -read(f, Int8)
                name = Vector{UInt8}(undef, strsize)
                readbytes!(f, name)
                population[p][g] =
                    IntegerGene(bit_vec, String(name))
            elseif id == 'F'
                m     = read(f, Int64)
                nvals = read(f, Int8 )
                names  = Vector{String }(undef, nvals)
                values = Vector{Float64}(undef, nvals)
                ranges = Vector{Float64}(undef, nvals)
                readbytes!(f, reinterpret(UInt8, values))
                readbytes!(f, reinterpret(UInt8, ranges))
                for i in 1:nvals
                    strsize = -read(f, Int8)
                    name = Vector{UInt8}(undef, strsize)
                    readbytes!(f, name)
                    names[i] = String(name)
                end
                population[p][g] =
                    FloatGene(values, ranges, m, names)
            elseif id == 'B'
                value = read(f, Bool)
                strsize = -read(f, Int8)
                name = Vector{UInt8}(undef, strsize)
                readbytes!(f, name)
                population[p][g] = BinaryGene(value, String(name))
            end
        end    
    end
    
    close(f)
    return ngens, tgens, population
end

"""
    reverse_backup(files ::Vector{<:AbstractString})

This should be used only for backup files of a parallel run, since each worker writes its own backup file.

Reads backup files `files` and returns the number of generations run in the slowest worker, the total number of generations supposed to be run and the entire population.
"""
function reverse_backup(files ::Vector{<:AbstractString})
    tgens = Vector{Int64     }(undef, length(files))
    gens  = Vector{Int64     }(undef, length(files)) 
    pop   = Vector{Individual}(undef, 0            )

    for (i,f) in enumerate(files)
        ngen, tgen, popul = reverse_backup(f)
        gens[i] = ngen
        tgens[i] = tgen
        append!(pop, popul)
    end

    min_gens = findmin( gens)[1]
    tot_gens = findmax(tgens)[1]
 
    return min_gens, tot_gens, pop
end
