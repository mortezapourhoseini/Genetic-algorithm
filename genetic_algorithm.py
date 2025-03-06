import numpy as np
from typing import List, Callable, Dict, Union

class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        chromosome_length: int,
        gene_type: str = 'binary',  # 'binary' or 'real'
        fitness_func: Callable = None,
        selection_method: str = 'roulette',  # 'roulette', 'tournament', 'rank'
        crossover_method: str = 'single_point',  # 'single_point', 'two_point', 'uniform', 'arithmetic'
        mutation_method: str = 'bit_flip',  # 'bit_flip', 'gaussian', 'swap', 'random'
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism_ratio: float = 0.1,
        bounds: List[tuple] = None,  # For real-valued genes: [(min, max), ...]
        termination_condition: Dict = {'max_generations': 100},
    ):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.gene_type = gene_type
        self.fitness_func = fitness_func
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_ratio = elitism_ratio
        self.bounds = bounds
        self.termination_condition = termination_condition
        
        self._validate_inputs()

    def _validate_inputs(self):
        if self.gene_type not in ['binary', 'real']:
            raise ValueError("gene_type must be 'binary' or 'real'.")
        if self.gene_type == 'real' and not self.bounds:
            raise ValueError("Bounds are required for real-valued genes.")
        if self.elitism_ratio < 0 or self.elitism_ratio > 1:
            raise ValueError("elitism_ratio must be between 0 and 1.")

    def initialize_population(self) -> Union[List[List[int]], List[List[float]]]:
        if self.gene_type == 'binary':
            return np.random.randint(2, size=(self.population_size, self.chromosome_length)).tolist()
        elif self.gene_type == 'real':
            return [
                [np.random.uniform(low, high) for (low, high) in self.bounds]
                for _ in range(self.population_size)
            ]

    def select_parents(self, population: List, fitness_values: List[float]) -> List:
        if self.selection_method == 'roulette':
            return self._roulette_selection(population, fitness_values)
        elif self.selection_method == 'tournament':
            return self._tournament_selection(population, fitness_values)
        elif self.selection_method == 'rank':
            return self._rank_selection(population, fitness_values)
        else:
            raise ValueError("Invalid selection method.")

    def _roulette_selection(self, population, fitness_values):
        probabilities = np.array(fitness_values) / np.sum(fitness_values)
        selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
        return [population[i] for i in selected_indices]

    def _tournament_selection(self, population, fitness_values, tournament_size=3):
        selected = []
        for _ in range(len(population)):
            candidates = np.random.choice(len(population), size=tournament_size, replace=False)
            best_index = candidates[np.argmax([fitness_values[i] for i in candidates])]
            selected.append(population[best_index])
        return selected

    def _rank_selection(self, population, fitness_values):
        ranks = np.argsort(np.argsort(fitness_values)) + 1  # Rank from 1 to N
        probabilities = ranks / np.sum(ranks)
        selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
        return [population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() > self.crossover_rate:
            return parent1, parent2
        
        if self.crossover_method == 'single_point':
            return self._single_point_crossover(parent1, parent2)
        elif self.crossover_method == 'two_point':
            return self._two_point_crossover(parent1, parent2)
        elif self.crossover_method == 'uniform':
            return self._uniform_crossover(parent1, parent2)
        elif self.crossover_method == 'arithmetic':
            return self._arithmetic_crossover(parent1, parent2)
        else:
            raise ValueError("Invalid crossover method.")

    def _single_point_crossover(self, parent1, parent2):
        point = np.random.randint(1, self.chromosome_length)
        return (
            parent1[:point] + parent2[point:],
            parent2[:point] + parent1[point:]
        )

    def _two_point_crossover(self, parent1, parent2):
        point1, point2 = sorted(np.random.choice(self.chromosome_length, 2, replace=False))
        return (
            parent1[:point1] + parent2[point1:point2] + parent1[point2:],
            parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        )

    def _uniform_crossover(self, parent1, parent2):
        mask = np.random.randint(2, size=self.chromosome_length)
        return (
            [p1 if m else p2 for p1, p2, m in zip(parent1, parent2, mask)],
            [p2 if m else p1 for p1, p2, m in zip(parent1, parent2, mask)]
        )

    def _arithmetic_crossover(self, parent1, parent2):
        alpha = np.random.uniform(0, 1)
        return (
            [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)],
            [(1 - alpha) * p1 + alpha * p2 for p1, p2 in zip(parent1, parent2)]
        )

    def mutate(self, individual):
        if self.gene_type == 'binary':
            return self._mutate_binary(individual)
        elif self.gene_type == 'real':
            return self._mutate_real(individual)

    def _mutate_binary(self, individual):
        return [
            gene if np.random.rand() >= self.mutation_rate else 1 - gene
            for gene in individual
        ]

    def _mutate_real(self, individual):
        mutated = []
        for i, gene in enumerate(individual):
            if np.random.rand() < self.mutation_rate:
                if self.mutation_method == 'gaussian':
                    new_gene = gene + np.random.normal(0, 0.1)
                elif self.mutation_method == 'random':
                    low, high = self.bounds[i]
                    new_gene = np.random.uniform(low, high)
                mutated.append(np.clip(new_gene, self.bounds[i][0], self.bounds[i][1]))
            else:
                mutated.append(gene)
        return mutated

    def replace_population(self, old_population, new_population, fitness_values):
        if self.elitism_ratio > 0:
            elite_size = int(self.elitism_ratio * self.population_size)
            elite_indices = np.argsort(fitness_values)[-elite_size:]
            return [old_population[i] for i in elite_indices] + new_population[:self.population_size - elite_size]
        return new_population

    def run(self):
        population = self.initialize_population()
        best_fitness_history = []
        
        for generation in range(self.termination_condition.get('max_generations', 100)):
            fitness_values = [self.fitness_func(ind) for ind in population]
            best_fitness_history.append(np.max(fitness_values))
            
            parents = self.select_parents(population, fitness_values)
            offspring = []
            
            for i in range(0, len(parents), 2):
                parent1, parent2 = parents[i], parents[i+1]
                child1, child2 = self.crossover(parent1, parent2)
                offspring.extend([self.mutate(child1), self.mutate(child2)])
            
            population = self.replace_population(
                population, 
                offspring[:self.population_size], 
                fitness_values
            )
            
            if 'target_fitness' in self.termination_condition:
                if max(fitness_values) >= self.termination_condition['target_fitness']:
                    break
        
        best_index = np.argmax([self.fitness_func(ind) for ind in population])
        return population[best_index], best_fitness_history
