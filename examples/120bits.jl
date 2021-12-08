"""
Problem: 120-bit sparse subset

Let a 120-bit string be b = [b_1|b_2|···|b_20], where b_i, i = 1, 2, ..., 20 are
six-bit substrings referred to as blocks. Define the sequence `000001` to be
the target configuration for a block, and denote this configuration by b_t.
The objective function is defined such that the optimal solution is
b* = [b_t|b_t|···|b_t], that is, the string with all 20 blocks having the target
configuration.

This solution can be expressed as the index vector v* = [6, 12, 18,...,120].
"""

using Evolutionary, Test, Random


function fitness(oset, bsize = 6)
    scores = Dict(i => (i != bsize ? 3i : 20) for i in 0:bsize)
    # count 1s in blocks
    ht = Dict{Int,Int}()
    sl = sr = 0
    for i in oset
        if i > 60
            sr += 1
        else
            sl += 1
        end
        blk = div(i, bsize)
        blk -= i % bsize == 0 ? 1 : 0
        if haskey(ht, blk)
            ht[blk] += 1
        else
            ht[blk] = 1
        end
    end
    # calculate total score
    ss = 0
    for blk in sort(collect(keys(ht)))
        cnt = ht[blk]
        if cnt == 7
            println(oset)
            println(ht)
            error("Problem")
        end
        ss += scores[cnt]
        if (cnt == 1) && ((blk+1)*bsize in oset)
            ss += 9
        end
        #println("$blk, $cnt, $(scores[cnt]), $ss")
    end
    ss -= 2(sl > 13 ? sl : 0)
    ss -= 2(sr > 13 ? sr : 0)
    ss
end

# Bit string examples 
bstr = "000100 001010 000010 111111 000100 101000 000010 000000 000001 000000 000010 000001 001000 000000 000000 010000 000000 000010 000000 000000"
idxs = [i for (i,j) in enumerate(filter(c->c != ' ', bstr)) if j == '1']
@test fitness(idxs) == 50

# optimal solution
idxs_opt = [6*i for i in 1:20]
@test fitness(idxs_opt) == 240

# GA solution
mthd = GA(populationSize = 100, selection = tournament(5),
          crossover = SSX, crossoverRate = 0.99,
          mutation  = replace(collect(1:120)), mutationRate = 0.1)

Random.seed!(42)
opts = Evolutionary.Options(show_trace=false, successive_f_tol=30)
init_pop = ()->randperm(120)[1:20]
res = Evolutionary.optimize(idxs->-fitness(idxs), init_pop, mthd, opts)
println(res)
Evolutionary.minimizer(res) |> sort

