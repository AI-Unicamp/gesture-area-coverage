import numpy as np


def DiceScore(grid1, grid2):
    s1, s2, tp, fp, fn = SetStats(grid1, grid2)
    # Dice Score, total sum of grid1, total sum of grid2, true positives, false positives, false negatives
    return 2*tp/(s1+s2), s1, s2, tp, fp, fn

def SetStats(grid1, grid2):
    grid1, grid2 = np.asarray(grid1), np.asarray(grid2)
    s1 = (grid1==1).sum()
    s2 = (grid2==1).sum()
    tp = np.logical_and((grid1==1), (grid2==1)).sum()
    fp = np.logical_and((grid1==0), (grid2==1)).sum()
    fn = np.logical_and((grid1==1), (grid2==0)).sum()
    #tn = np.logical_and((grid1==0), (grid2==0)).sum()
    return s1, s2, tp, fp, fn