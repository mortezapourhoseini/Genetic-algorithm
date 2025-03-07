# Genetic Algorithm (GA) Framework

A flexible and modular Python implementation of a Genetic Algorithm for optimization problems, supporting both binary and real-valued encoding.

## Features

- 🧬 **Multi-type genes**: Binary and real-valued representations
- 🔄 **Evolutionary operators**:
  - Selection: Roulette, Tournament, Rank-based
  - Crossover: Single-point, Two-point, Uniform, Arithmetic
  - Mutation: Bit-flip, Gaussian, Random reset
- 🏆 **Elitism**: Preserve best solutions between generations
- 📊 **Visualization**: Integrated with Graphviz for circuit visualization
- ⚙️ **Customizable**: Easily extendable for specific problem domains

## Installation

```bash
pip install numpy 
```

## Usage

### Basic Optimization
```python
from genetic_algorithm import GeneticAlgorithm

# Minimize sphere function: f(x) = Σx²
def sphere_fitness(x):
    return -sum(xi**2 for xi in x)

ga = GeneticAlgorithm(
    population_size=50,
    chromosome_length=3,
    gene_type='real',
    fitness_func=sphere_fitness,
    bounds=[(-5, 5)]*3,
    termination_condition={'max_generations': 100}
)

best_solution, history = ga.run()
```


## Key Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `population_size` | Number of solutions per generation | 50-500 |
| `chromosome_length` | Solution representation length | 10-100 |
| `gene_type` | Encoding type | 'binary'/'real' |
| `selection_method` | Parent selection strategy | 'roulette', 'tournament', 'rank' |
| `crossover_rate` | Probability of crossover | 0.6-0.9 |
| `mutation_rate` | Probability per gene mutation | 0.01-0.1 |
| `elitism_ratio` | Top solutions to preserve | 0.05-0.2 |
| `bounds` | (Real-only) Value ranges | [(min,max), ...] |


**Optimize smarter, evolve faster!** 
