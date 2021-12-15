import numpy as np
from environment import Environment
from population import Population


class GeneticAlgorithm:
    
    def __init__(self):
        self.history = []
    
    
    def fit(self, 
            iterations : int,
            env : Environment, 
            population : Population,
           ) -> None:

        for iteration in range(iterations): 
            results = []
            costs = []
            
            for chromosome in population.chromosomes: 
                env.deploy_chromosome(chromosome)
                cost_items = env.cost()
                chromosome.cost_items = cost_items
                costs.append(cost_items)

            costs = np.array(costs)
            agg = np.mean(costs, axis=0)
            tot = np.sum(costs, axis=1)
            self.history.append(agg)

            costs = costs - np.min(costs, axis=0)
            max_costs = np.max(costs, axis=0) + 1e-12
            costs = costs / max_costs
            costs = np.sum(costs, axis=1)
            
            for i, chromosome in enumerate(population.chromosomes): 
                cost = costs[i]
                chromosome.cost = cost

            if iteration % 100 == 0:
                print('Iteration {:5} : Cost {:0.6f}'.format(iteration, np.min(tot)))

            if np.min(tot) == 0: 
                print('Done')
                break

            if iteration < iterations - 1:
                population.new_population(
                    n_best_to_save=1, 
                    population_size=20,
                )