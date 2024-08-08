import numpy as np
import matplotlib.pyplot as plt

def plot_HL_versus_dice_fgd(hl_ratings,
                            aligned_fgds,
                            aligned_dice,
                            aligned_entries,
                            ):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,8))

    # FGD
    color = 'tab:blue'
    ax1.scatter(aligned_fgds, hl_ratings[1:], label='FGD', color=color, marker='o', s=80)

    # Hardcoded positions for the annotations to avoid overlapping
    for i, entry in enumerate(aligned_entries[1:]):
        x,y = [
            [0,0],      #SG
            [12,0],     #SF
            [12,2.5],   #SJ
            [12,0],     #SL
            [12,0],     #SE
            [12,0],     #SH
            [1,1],      #BD
            [11,-1],    #SD
            [1,0],      #BM
            [3,0],      #SI
            [1.5,-0.6], #SK
            [0,0],      #SA
            [0,0],      #SB
            [1,0.2],    #SC
        ][i]
        ax1.annotate(entry, (aligned_fgds[i]+x-10, hl_ratings[i+1]+y-1.5), fontsize=18)
    #ax.tick_params(axis='y', labelcolor=color)
    # Linear regression for the FGD
    coef = np.polyfit(hl_ratings[1:],aligned_fgds,1)
    poly1d_fn = np.poly1d(coef) 
    ax1.plot(poly1d_fn(hl_ratings[1:]), hl_ratings[1:], '--k', color=color)
    # Hardcoded limits to improve visualization
    ax1.set_xlim(0, 90)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('FGD')
    ax1.set_ylabel('Human-likeness')

    # Dice Score
    color = 'tab:orange'
    ax2.scatter(aligned_dice[1:], hl_ratings[1:], label='Dice', color=color, marker='o', s=80)

    # Hardcoded positions for the annotations to avoid overlapping
    for i, entry in enumerate(aligned_entries[1:]):
        x,y = [
            [0,0],          #SG
            [0,0],          #SF
            [-0.001,1],     #SJ
            [0,0],          #SL
            [-0.003,-1.8],  #SE
            [-0.018,-3],    #SH
            [-0.008,-3],    #BD
            [-0.025,0],     #SD
            [0,0],          #BM
            [0,0],          #SI
            [0,0],          #SK
            [0,0],          #SA
            [0,0],          #SB
            [0,0],          #SC
        ][i]
        ax2.annotate(entry, (aligned_dice[i+1]+x+0.005, hl_ratings[i+1]+y-1.5), fontsize=20)

    # Linear regression for the Dice Score
    coef = np.polyfit(hl_ratings[1:],aligned_dice[1:],1)
    poly1d_fn = np.poly1d(coef) 
    ax2.plot(poly1d_fn(hl_ratings[1:]), hl_ratings[1:], '--k', color=color)
    # Hardcoded limits to improve visualization
    ax2.set_xlim(0.6, 0.8)
    ax2.set_xlabel('Dice Score')
    #ax2.set_yticks(np.arange(0.6,0.8,0.05))
    
    return fig

    #ymin, ymax = ax.get_ylim()
    #limits = [0, 9, 18, 27, 40, 53, 66, 75, 84, 93, 106, 119, 131]
    #xticks = np.arange(len(scores[1:]))
    #ax.set_ylim(0, 1)
    #ax.set_xticks(xticks)
    #_=ax.set_xticklabels(scores[1:], fontdict={'fontsize': 20})