import time
import numpy as np


# Update the position of each Northern Goshawk
def update_positions(population, best_solution, lb, ub, alpha, beta):
    pop_size, dim = population.shape
    new_population = np.copy(population)

    for i in range(pop_size):
        for j in range(dim):
            r1 = np.random.rand()
            r2 = np.random.rand()

            A = 2 * alpha * r1 - alpha
            C = 2 * r2

            D = abs(C * best_solution[j] - population[i, j])
            new_population[i, j] = best_solution[j] - A * D

            # Ensure the new position is within bounds
            new_population[i, j] = np.clip(new_population[i, j], lb, ub)

    return new_population


# Northern Goshawk Optimization (NGO)
def NGO(population, objective_function, VRmin, VRmax, max_iter):
    pop_size, dim = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]
    alpha = 2
    beta = 1

    # Evaluate the fitness of the initial population
    fitness = objective_function(population[:])

    # Find the best initial solution
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx, :]
    best_fitness = fitness[best_idx]

    Convergence_curve = np.zeros((max_iter, 1))

    t = 0
    ct = time.time()
    for t in range(max_iter):
        # Update the positions of Northern Goshawks
        population = update_positions(population, best_solution, lb, ub, alpha, beta)

        # Evaluate the fitness of the new population
        fitness = objective_function(population[:])

        # Update the best solution
        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[current_best_idx, :]
        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[max_iter - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, ct
