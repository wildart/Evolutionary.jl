
using Evolutionary

  int_gene = IntegerGene(10, "integer", lb=-2, ub=5)
float_gene =   FloatGene( 1, "float")

pop = [AbstractGene[float_gene]]

open("backup-test", "w") do f
    backup(1, 1, pop, "backup-test")
end

backup_pop = reverse_backup("backup-files/backup-test")

display(float_gene)
display(backup_pop[3][1][1])

