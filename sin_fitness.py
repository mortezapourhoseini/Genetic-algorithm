import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm  

def sin_fitness(individual):
    x = individual[0]
    return np.sin(x)

ga = GeneticAlgorithm(
    population_size=50,
    chromosome_length=1,          # Single variable optimization
    gene_type='real',             
    fitness_func=sin_fitness,
    selection_method='tournament',
    crossover_method='arithmetic',
    mutation_method='gaussian',
    crossover_rate=0.8,
    mutation_rate=0.1,
    elitism_ratio=0.1,
    bounds=[(0, 2 * np.pi)],     # Search space
    termination_condition={
        'max_generations': 50,
        'target_fitness': 0.999  # Stop if near-optimal solution found
    }
)

# Run optimization
best_solution, fitness_history = ga.run()

# Results
print(f"Best solution: x = {best_solution[0]:.4f} (sin(x) = {np.sin(best_solution[0]):.4f})")
print(f"Theoretical maximum at x = π/2 ≈ {np.pi/2:.4f} (sin(π/2) = 1.0)")

# Plot convergence
plt.plot(fitness_history)
plt.xlabel('Generation')
plt.ylabel('Best Fitness (sin(x))')
plt.title('Genetic Algorithm Convergence')
plt.grid(True)
plt.show()
