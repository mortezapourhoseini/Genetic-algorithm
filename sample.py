import numpy as np

def fitness(x):
    return x**2  # هدف: بیشینه کردن x^2

# پارامترها
population_size = 50
num_generations = 100
mutation_rate = 0.01

# جمعیت اولیه (اعداد تصادفی بین ۰ تا ۳۱)
population = np.random.randint(0, 32, population_size)

for generation in range(num_generations):
    # محاسبه Fitness
    fitness_values = [fitness(x) for x in population]
    
    # انتخاب والدین (Tournament Selection)
    parents = []
    for _ in range(population_size):
        candidates = np.random.choice(population, size=3, replace=False)
        parent = candidates[np.argmax([fitness(x) for x in candidates])]
        parents.append(parent)
    
    # تقاطع (Single-Point Crossover)
    offspring = []
    for i in range(0, population_size, 2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        crossover_point = np.random.randint(1, 5)  # نقاط تقسیم باینری ۵ بیتی
        mask = (1 << crossover_point) - 1
        child1 = (parent1 & ~mask) | (parent2 & mask)
        child2 = (parent2 & ~mask) | (parent1 & mask)
        offspring.extend([child1, child2])
    
    # جهش (Bit-Flip Mutation)
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            bit_to_flip = np.random.randint(0, 5)
            offspring[i] ^= (1 << bit_to_flip)
    
    # جایگزینی نسل
    population = np.array(offspring)

# بهترین فرد در نسل پایانی
best_solution = population[np.argmax([fitness(x) for x in population])]
print("بهترین راهحل:", best_solution)
