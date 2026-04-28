import time

import numpy as np


# Red Kite Optimization Algorithm (RKOA)
def RKOA(red_kites, fobj, lb, ub, max_iterations):
    num_variables, num_red_kites = red_kites.shape[0], red_kites.shape[1]
    c1 = 1  # Exploration parameter
    c2 = 1  # Exploitation parameter
    c3 = 1  # Memory parameter
    best_fitness = np.zeros((num_red_kites, 1))
    best_solution = float('inf')
    Convergence_curve = np.zeros((max_iterations, 1))
    fitness = fobj(red_kites[:])

    t = 0
    ct = time.time()
    for iteration in range(1, max_iterations + 1):
        # Evaluate fitness
        for i in range(num_red_kites):
            fitness[i] = fobj(red_kites[i])

        # Sort red kites by fitness
        sorted_indices = np.argsort(fitness)
        red_kites = red_kites[sorted_indices]
        fitness = fitness[sorted_indices]

        # Exploration phase
        for i in range(num_red_kites):
            step_size = c1 / iteration
            direction = np.random.randn(num_variables)
            red_kites[i] += step_size * direction

        # Exploitation phase
        for i in range(num_red_kites):
            if np.random.rand() < c2:
                reference_index = np.random.randint(num_red_kites)
                while reference_index == i:
                    reference_index = np.random.randint(num_red_kites)
                reference = red_kites[reference_index]
                diff = reference - red_kites[i]
                norm_diff = np.linalg.norm(diff)
                if norm_diff > 0:
                    direction = diff / norm_diff
                    step_size = c1 / iteration
                    red_kites[i] += step_size * direction

        # Memory phase
        best_red_kite = red_kites[0]
        for i in range(1, num_red_kites):
            if fobj(red_kites[i]) < fobj(best_red_kite):
                best_red_kite = red_kites[i]
        red_kites = red_kites + c3 * (best_red_kite - red_kites)

        # Ensure red kites stay within bounds
        red_kites = np.clip(red_kites, lb, ub)

        # Display best solution
        best_solution = red_kites[0]
        best_fitness = fitness[0]
        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[max_iterations - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, ct
