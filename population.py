import numpy as np
from genetics import Chromosome


class Population: 
    
    def __init__(self,
                 population_size : int, 
                 gene_names : list
                ) -> None:
                
        self.cur_chromosomes = []
        for i in range(population_size):
            chromosome = Chromosome()
            chromosome.create_random_genes(gene_names)
            self.cur_chromosomes.append(chromosome)

    @property
    def chromosomes(self) -> list:
        return self.cur_chromosomes
            
        
    def new_population(self, 
                       n_best_to_save : int, 
                       population_size : int, 
                       crossover_mutations : list = [1, 20],
                       p_crossover_mutations : list = [0.98, 0.02],
                       explore_coeff : float = 0.05) -> list: 
        
        new_chromosomes = []
        
        # Sort the old chromosomes in performance order. 
        self.cur_chromosomes = sorted(self.cur_chromosomes, key=lambda x: x.cost)
        
        # Pick few of the best chromosomes and mutate those
        for i in range(n_best_to_save):
            chromosome = self.cur_chromosomes[i]
            chromosome.mutate(n_mut=1)
            new_chromosomes.append(chromosome)
        
        # Crossover the chromosomes. Fitness is used to prioritize the 
        # chromosome selection. Fitness is calculated from the cost values. 
        costs = np.array([chromosome.cost for chromosome in self.cur_chromosomes])
        costs = costs - np.min(costs)
        costs = costs / (np.max(costs) + 1e-12)
        fitness = 1 / (costs + explore_coeff)
        fitness = fitness / np.sum(fitness) # Scale sum of values to 1
        
        while len(new_chromosomes) < population_size:
            chromosome_a = np.random.choice(self.cur_chromosomes, p=fitness)
            chromosome_b = np.random.choice(self.cur_chromosomes, p=fitness)
            new_chromosome = chromosome_a.crossover(chromosome_b)
            n_mutation = np.random.choice(crossover_mutations, p=p_crossover_mutations)
            new_chromosome.mutate(n_mutation)
            new_chromosomes.append(new_chromosome)
            
        self.cur_chromosomes = new_chromosomes