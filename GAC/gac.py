import numpy as np


def DiceScore(tp, s1, s2):
    return 2*tp/(s1+s2)

def FPFNRatio(fp, fn):
    ratio = fp / fn
    log_ratio = np.log(ratio)
    return ratio, log_ratio

def SetStats(grid1, grid2):
    grid1, grid2 = np.asarray(grid1), np.asarray(grid2)
    s1 = (grid1==1).sum()
    s2 = (grid2==1).sum()
    tp = np.logical_and((grid1==1), (grid2==1)).sum()
    fp = np.logical_and((grid1==0), (grid2==1)).sum()
    fn = np.logical_and((grid1==1), (grid2==0)).sum()
    #tn = np.logical_and((grid1==0), (grid2==0)).sum()
    dice = DiceScore(tp, s1, s2)
    ratio, log_ratio = FPFNRatio(fp, fn)
    return dice, ratio, log_ratio, s1, s2, tp, fp, fn