import time
import numpy as np


def update_radius(iteration, max_iter):
    return 0.5 * (1 - abs((2 * iteration / max_iter) - 1))


def create_valleys(center, radius, dim):
    points = []
    num_valleys = 8  # Example: 8 valleys
    for m in range(1, (dim // 2) + 1):
        for k in range(num_valleys):
            angle = 2 * np.pi * k / num_valleys
            new_point = center.copy()
            new_point[2 * m - 2] += radius * np.cos(angle)
            new_point[2 * m - 1] += radius * np.sin(angle)
            points.append(new_point)
    return np.array(points)


def create_quarter_points(center, radius, dim):
    points = []
    for a in np.linspace(0, 1, 4):
        new_point = center + radius * np.random.uniform(-1, 1, dim) * (1 - a)
        points.append(new_point)
    return np.array(points)


def flow_step(point, center):
    flow_speed = 0.033 * np.random.uniform(0, 1)
    direction = np.sign(center - point)
    return point + direction * flow_speed


def MOA(population, func, lb, ub, max_iter):
    population_size, dim = population.shape[0], population.shape[1]
    fitness = func(population[:])
    best_idx = np.argmin(fitness)
    center = population[best_idx].copy()
    best_value = np.zeros((dim, 1))
    best_position = float('inf')
    Convergence_curve = np.zeros((max_iter, 1))

    t = 0
    ct = time.time()
    for iteration in range(1, max_iter + 1):
        radius = update_radius(iteration, max_iter)
        points = create_valleys(center, radius, dim)
        points = np.vstack((points, create_quarter_points(center, radius, dim)))

        fitness = func(population[:])
        sorted_indices = np.argsort(fitness)
        best_candidates = points[sorted_indices[:4]]

        for i in range(len(best_candidates)):
            new_point = flow_step(best_candidates[i], center)
            new_value = func(func, new_point, iteration)
            if new_value < func(best_candidates[i]):
                best_candidates[i] = new_point

        center = best_candidates[func(best_candidates[:])]

        best_position = center
        Convergence_curve[t] = best_value
        t = t + 1
    best_value = Convergence_curve[max_iter - 1][0]
    ct = time.time() - ct
    return best_value, Convergence_curve, best_position, ct
