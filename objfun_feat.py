import numpy as np
from Global_Vars import Global_Vars
from Relief_score import reliefF


def objfun(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i, :]
            feat = data[:, sol]
            chi_squared_stat = (((feat - Tar) ** 2) / feat).sum().sum()
            rscore = reliefF(np.array(feat), np.array(Tar.reshape(-1)))
            Fitn[i] = 1 / (rscore + chi_squared_stat)
    else:
        sol = Soln
        feat = data[:, sol]
        chi_squared_stat = (((feat - Tar) ** 2) / feat).sum().sum()
        rscore = reliefF(np.array(feat), np.array(Tar.reshape(-1)))
        Fitn = 1 / (rscore + chi_squared_stat)
        return Fitn
