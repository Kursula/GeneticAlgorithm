import numpy as np


class Gene:
    
    def __init__(self, name : str, params : dict = None) -> None: 
        """
        Initialize the gene. If params is not provided, will generate random params. 
        """
        self.name = name
        if params is None: 
            params = dict(
                x=np.random.rand() * 5,
                y=np.random.rand() * 5,
                rotation=np.random.choice([True, False])
            )
        self.params = params
    
    
    def copy(self): 
        """
        Makes a deep copy of itself. Copy is often needed e.g. in crossover to make sure that 
        mutations in the child do not backpropagate to the parents. 
        """
        new_gene = type(self)(self.name, self.params.copy())
        return new_gene
    
    
    def mutate(self) -> None: 
        """
        Create random mutation in the gene parameters. 
        All feasibility constraints should be implemented here to make sure that 
        mutations do not result in values that are impossible to evaluate. 
        """
        max_move = 2
        prob = 1 / 3
        if np.random.rand() < prob: 
            # Modify x 
            delta = (np.random.rand() - 0.5) * max_move 
            self.params['x'] += delta
        elif np.random.rand() < prob: 
            # Modify y
            delta = (np.random.rand() - 0.5) * max_move 
            self.params['y'] += delta
        else: 
            # Mofify rotation
            self.params['rotation'] = not self.params['rotation']
    
        
class Chromosome: 
    
    def __init__(self) -> None:
        self.genes = {}
        self.cost = None
        self.cost_items = None
    
    
    def create_random_genes(self, gene_names : list):
        for gname in gene_names: 
            gene = Gene(gname)
            self.add_gene(gene)
    
    
    def add_gene(self, gene : Gene) -> None: 
        self.genes[gene.name] = gene
        
    
    def mutate(self, n_mut : int = 1) -> None:
        """
         Create total n_mut mutations in randomly selected genes. 
        """
        for i in range(n_mut):
            gene = np.random.choice(list(self.genes.values()))
            gene.mutate()
            
                
    def crossover(self, chromosome_b):
        """
        Creates a new chromosome that is random mix of genes from self and chromosome_b.
        """
        new_chromosome = type(self)()
        for key in self.genes.keys(): 
            if np.random.rand() < 0.5:
                gene = self.genes[key].copy()
            else:
                gene = chromosome_b.genes[key].copy()
            new_chromosome.add_gene(gene)
        return new_chromosome