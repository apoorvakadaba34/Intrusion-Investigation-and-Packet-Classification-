import time

import numpy as np


# Coati Optimization Algorithm (COA)
def COA(SearchAgents, fobj, VRmin, VRmax, Max_iterations):
    N, dimension = SearchAgents.shape[0], SearchAgents.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]
    X = SearchAgents

    # Fitness evaluation
    fit = fobj(SearchAgents[:])

    best_so_far = []
    average = []

    ct = time.time()
    for t in range(Max_iterations):
        # update the best candidate solution
        best = np.min(fit)
        location = np.argmin(fit)

        if t == 0:
            Xbest = X[location, :]  # Optimal location
            fbest = best  # The optimization objective function
        elif best < fbest:
            fbest = best
            Xbest = X[location, :]

        # Phase 1: Hunting and attacking strategy on iguana (Exploration Phase)
        for i in range(SearchAgents.shape[0] // 2):
            iguana = Xbest
            I = round(1 + np.random.rand())
            X_P1 = X[i, :] + np.random.rand() * (iguana - I * X[i, :])  # Eq. (4)
            X_P1 = np.clip(X_P1, lb, ub)

            # Update position based on Eq (7)
            F_P1 = fit
            if F_P1[i] < fit[i]:
                X[i, :] = X_P1
                fit = F_P1

        for i in range(SearchAgents.shape[0] // 2, SearchAgents.shape[0]):
            iguana = lb + np.random.rand() * (ub - lb)  # Eq. (5)
            F_HL = fit
            I = round(1 + np.random.rand())

            if fit[i] > F_HL[i]:
                X_P1 = X[i, :] + np.random.rand() * (iguana - I * X[i, :])  # Eq. (6)
            else:
                X_P1 = X[i, :] + np.random.rand() * (X[i, :] - iguana)  # Eq. (6)

            X_P1 = np.clip(X_P1, lb, ub)

        # Phase 2: The process of escaping from predators (Exploitation Phase)
        for i in range(SearchAgents.shape[0]):
            LO_LOCAL = lb / (t + 1)  # Eq. (9)
            HI_LOCAL = ub / (t + 1)  # Eq. (10)

            X_P2 = X[i, :] + (1 - 2 * np.random.rand()) * (
                    LO_LOCAL + np.random.rand() * (HI_LOCAL - LO_LOCAL))  # Eq. (8)
            X_P2 = np.clip(X_P2, LO_LOCAL, HI_LOCAL)

        best_so_far.append(fbest)
        average.append(np.mean(fit))

    Best_score = fbest
    Best_pos = Xbest
    COA_curve = best_so_far
    ct = time.time() - ct
    return Best_score, COA_curve, Best_pos, ct
