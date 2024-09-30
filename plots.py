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
    ax1.scatter(aligned_fgds, hl_ratings, label='FGD', color=color, marker='o', s=80)

    # Hardcoded positions for the annotations to avoid overlapping
    for i, entry in enumerate(aligned_entries):
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
        ax1.annotate(entry, (aligned_fgds[i]+x-10, hl_ratings[i]+y-1.5), fontsize=18)
    #ax.tick_params(axis='y', labelcolor=color)
    # Linear regression for the FGD
    coef = np.polyfit(aligned_fgds, hl_ratings,1)
    poly1d_fn = np.poly1d(coef) 
    ax1.plot(np.sort(aligned_fgds), poly1d_fn(np.sort(aligned_fgds)), '--k', color=color)
    # Hardcoded limits to improve visualization
    ax1.set_xlim(0, 90)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('FGD')
    ax1.set_ylabel('Human-likeness')

    # Dice Score
    color = 'tab:orange'
    ax2.scatter(aligned_dice, hl_ratings, label='Dice', color=color, marker='o', s=80)

    # Hardcoded positions for the annotations to avoid overlapping
    for i, entry in enumerate(aligned_entries):
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
        ax2.annotate(entry, (aligned_dice[i]+x+0.005, hl_ratings[i]+y-1.5), fontsize=20)

    # Linear regression for the Dice Score
    coef = np.polyfit(aligned_dice,hl_ratings,1)
    poly1d_fn = np.poly1d(coef) 
    ax2.plot(np.sort(aligned_dice), poly1d_fn(np.sort(aligned_dice)), '--k', color=color)
    # Hardcoded limits to improve visualization
    ax2.set_xlim(0.6, 0.8)
    ax2.set_xlabel('Dice Score')
    #ax2.set_yticks(np.arange(0.6,0.8,0.05))
    
    return fig

def plot_zeggs_gac_comparison(zeggs_dataset,
                              rasterizer,
                              neutral_idx = 2,
                              comp1_idx = 5, 
                              comp2_idx = 14,
                              frame_skip = 1,
                              grid_step = 0.5,
                              weigth = 1,
                              ):

    neutral = rasterizer.rasterize(zeggs_dataset.posLckHips()[neutral_idx]-[0,-100,0],  #Subtracting 100 to keep it in center
                                   zeggs_dataset.parents, 
                                   frame_skip=frame_skip, 
                                   grid_step=grid_step, 
                                   weigth=weigth,
                                   skipfirstjoints=1)
    neutral.clipped = np.clip(neutral.grid, 0, 1)

    comp1 = rasterizer.rasterize(zeggs_dataset.posLckHips()[comp1_idx]-[0,-100,0], #Subtracting 100 to keep it in center
                                 zeggs_dataset.parents, 
                                 frame_skip=frame_skip, 
                                 grid_step=grid_step, 
                                 weigth=weigth,
                                 skipfirstjoints=1)
    comp1.clipped = np.clip(comp1.grid, 0, 1).astype(float)

    comp2 = rasterizer.rasterize(zeggs_dataset.posLckHips()[comp2_idx]-[0,-100,0], #Subtracting 100 to keep it in center
                                 zeggs_dataset.parents, 
                                 frame_skip=frame_skip, 
                                 grid_step=grid_step, 
                                 weigth=weigth,
                                 skipfirstjoints=1)
    comp2.clipped = np.clip(comp2.grid, 0, 1).astype(float)

    # Setting 0 values to nan to not take part on cmap
    #z_grid_teste.clipped[z_grid_teste.clipped == 0] = np.nan

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12,12))

    axs[1,0].remove()

    neg = axs[0,0].imshow(neutral.clipped, cmap='binary', interpolation='none')

    neg = axs[0,1].imshow(comp1.clipped, cmap='binary', interpolation='none')#, vmin=0,vmax=4)
    #ax.imshow(z_grid_teste.clipped, cmap=white_at_min, interpolation='nearest')
    #fig.colorbar(neg, ax=ax, anchor=(0, 0.3), shrink=0.7)

    neg = axs[0,2].imshow(comp2.clipped, cmap='binary', interpolation='none')

    comp1.clipped[np.logical_and(neutral.clipped==0,
                                 comp1.clipped==0
                                 )]=np.nan

    neg = axs[1,1].imshow(neutral.clipped-comp1.clipped+1, cmap='Blues', interpolation='none', vmin=-1)

    comp2.clipped[np.logical_and(neutral.clipped==0,
                                 comp2.clipped==0
                                 )]=np.nan

    neg = axs[1,2].imshow(neutral.clipped-comp2.clipped+1, cmap='Blues', interpolation='none', vmin=-1)

    for i in range(2):
        for j in range(3):
            axs[i,j].tick_params(
                    axis='both',       # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    left=False,
                    top=False,         # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False)   # labels along the bottom edge are off
            
            
    axs[0,0].set_xlabel('Neutral')
    axs[0,1].set_xlabel('Sad')
    axs[0,2].set_xlabel('Happy')
    axs[1,1].set_xlabel('Neutral - Sad')
    axs[1,2].set_xlabel('Neutral - Happy')

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.tight_layout()
    return fig

def plot_HL_versus_dice_fgd_g22(hl_ratings,
                               aligned_fgds,
                               aligned_dice,
                               aligned_entries,
                               ):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,8))

    # FGD
    color = 'tab:blue'
    ax1.scatter(aligned_fgds, hl_ratings, label='FGD', color=color, marker='o', s=80)

    # Hardcoded positions for the annotations to avoid overlapping
    for i, entry in enumerate(aligned_entries):
        x,y = [
            [0,0],          #FBT
            [0,0],          #FSA
            [-21,0],          #FSB
            [0,0],          #FSC
            [-22,0],          #FSD
            [0,0],          #FSF
            [0,+1],          #FSG
            [0,-1],          #FSH
            [-20,0],          #FSI
        ][i]
        ax1.annotate(entry, (aligned_fgds[i]+x+3, hl_ratings[i]+y-1.5), fontsize=18)
    #ax.tick_params(axis='y', labelcolor=color)
    # Linear regression for the FGD
    coef = np.polyfit(aligned_fgds, hl_ratings,1)
    poly1d_fn = np.poly1d(coef) 
    ax1.plot(np.sort(aligned_fgds), poly1d_fn(np.sort(aligned_fgds)), '--k', color=color)
    # Hardcoded limits to improve visualization
    ax1.set_xlim(0, 140)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('FGD')
    ax1.set_ylabel('Human-likeness')

    # Dice Score
    color = 'tab:orange'
    ax2.scatter(aligned_dice, hl_ratings, label='Dice', color=color, marker='o', s=80)

    # Hardcoded positions for the annotations to avoid overlapping
    for i, entry in enumerate(aligned_entries):
        x,y = [
            [0.03,4],          #FBT
            [0,0],          #FSA
            [0,0],          #FSB
            [0,0],          #FSC
            [0.03,4],          #FSD
            [0.07,2],          #FSF
            [0.07,-2],          #FSG
            [0,0],          #FSH
            [0.07,1],          #FSI
        ][i]
        ax2.annotate(entry, (aligned_dice[i]+x-0.06, hl_ratings[i]+y-1.5), fontsize=20)

    # Linear regression for the Dice Score
    coef = np.polyfit(aligned_dice,hl_ratings,1)
    poly1d_fn = np.poly1d(coef) 
    ax2.plot(np.sort(aligned_dice), poly1d_fn(np.sort(aligned_dice)), '--k', color=color)
    # Hardcoded limits to improve visualization
    ax2.set_xlim(0.45, 0.85)
    ax2.set_xlabel('Dice Score')
    
    return fig

def plot_APP_versus_dice_fgd_g22(app_ratings,
                            aligned_fgds,
                            aligned_dice,
                            aligned_entries,
                            ):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,8))

    # FGD
    color = 'tab:blue'
    ax1.scatter(aligned_fgds, app_ratings, label='FGD', color=color, marker='o', s=80)

    # Hardcoded positions for the annotations to avoid overlapping
    for i, entry in enumerate(aligned_entries):
        x,y = [
            [-8,-1],          #FBT
            [-8,0.5],          #FSA
            [-8,0.5],          #FSB
            [2,-0.5],          #FSC
            [-8,0.5],          #FSD
            [-8,-1],          #FSF
            [-12,-1],          #FSG
            [-8,0.5],          #FSH
            [1.8,-0.8],          #FSI
        ][i]
        ax1.annotate(entry, (aligned_fgds[i]+x, app_ratings[i]+y), fontsize=18)
    #ax.tick_params(axis='y', labelcolor=color)
    # Linear regression for the FGD
    coef = np.polyfit(aligned_fgds, app_ratings,1)
    poly1d_fn = np.poly1d(coef) 
    ax1.plot(np.sort(aligned_fgds), poly1d_fn(np.sort(aligned_fgds)), '--k', color=color)
    # Hardcoded limits to improve visualization
    ax1.set_xlim(0, 140)
    ax1.set_ylim(45, 65)
    ax1.set_xlabel('FGD')
    ax1.set_ylabel('Appropriateness')

    # Dice Score
    color = 'tab:orange'
    ax2.scatter(aligned_dice, app_ratings, label='Dice', color=color, marker='o', s=80)

    # Hardcoded positions for the annotations to avoid overlapping
    for i, entry in enumerate(aligned_entries):
        x,y = [
            [0,0.5],          #FBT
            [0,0.5],          #FSA
            [-0.03,0.5],          #FSB
            [-0.02,-0.9],          #FSC
            [0,-1],          #FSD
            [0,-1],          #FSF
            [-0.02,-1],          #FSG
            [0,0.5],          #FSH
            [0.03,-0.8],          #FSI
        ][i]
        ax2.annotate(entry, (aligned_dice[i]+x-0.03, app_ratings[i]+y), fontsize=20)

    # Linear regression for the Dice Score
    coef = np.polyfit(aligned_dice,app_ratings,1)
    poly1d_fn = np.poly1d(coef) 
    ax2.plot(np.sort(aligned_dice), poly1d_fn(np.sort(aligned_dice)), '--k', color=color)
    # Hardcoded limits to improve visualization
    ax2.set_xlim(0.45, 0.85)
    ax2.set_xlabel('Dice Score')
    
    return fig

def plot_APP_versus_dice_fgd(app_ratings,
                            aligned_fgds,
                            aligned_dice,
                            aligned_entries,
                            ):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,8))

    # FGD
    color = 'tab:blue'
    ax1.scatter(aligned_fgds, app_ratings, label='FGD', color=color, marker='o', s=80)

    # Hardcoded positions for the annotations to avoid overlapping
    for i, entry in enumerate(aligned_entries):
        x,y = [
            [0,0],          #SG
            [-2,-.7],          #SF
            [0,0],     #SJ
            [0,0],          #SL
            [0,0],  #SE
            [0,0],    #SH
            [-2,-.7],    #BD
            [0,0],     #SD
            [0,0],          #BM
            [-2,-.7],          #SI
            [-3,0],          #SK
            [-2,-.7],          #SA
            [0,0],          #SB
            [0,0],          #SC
        ][i]
        ax1.annotate(entry, (aligned_fgds[i]+x, app_ratings[i]+y), fontsize=18)
    #ax.tick_params(axis='y', labelcolor=color)
    # Linear regression for the FGD
    coef = np.polyfit(aligned_fgds, app_ratings,1)
    poly1d_fn = np.poly1d(coef) 
    ax1.plot(np.sort(aligned_fgds), poly1d_fn(np.sort(aligned_fgds)), '--k', color=color)
    # Hardcoded limits to improve visualization
    ax1.set_xlim(0, 90)
    ax1.set_ylim(44, 64)
    ax1.set_xlabel('FGD')
    ax1.set_ylabel('Appropriateness')

    # Dice Score
    color = 'tab:orange'
    ax2.scatter(aligned_dice, app_ratings, label='Dice', color=color, marker='o', s=80)

    # Hardcoded positions for the annotations to avoid overlapping
    for i, entry in enumerate(aligned_entries):
        x,y = [
            [0,0],          #SG
            [0,0],          #SF
            [0,0],     #SJ
            [0,0],          #SL
            [0,0],  #SE
            [0,0],    #SH
            [0,0],    #BD
            [0,0],     #SD
            [0,0],          #BM
            [0,0],          #SI
            [0,0],          #SK
            [0,0],          #SA
            [0,0],          #SB
            [0,0],          #SC
        ][i]
        ax2.annotate(entry, (aligned_dice[i]+x, app_ratings[i]+y), fontsize=20)

    # Linear regression for the Dice Score
    coef = np.polyfit(aligned_dice,app_ratings,1)
    poly1d_fn = np.poly1d(coef) 
    ax2.plot(np.sort(aligned_dice), poly1d_fn(np.sort(aligned_dice)), '--k', color=color)
    # Hardcoded limits to improve visualization
    ax2.set_xlim(0.6, 0.8)
    ax2.set_xlabel('Dice Score')
    
    return fig

def plot_APP_MAS_versus_dice_fgd(app_ratings,
                            aligned_fgds,
                            aligned_dice,
                            aligned_entries,
                            ):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,8))

    # FGD
    color = 'tab:blue'
    ax1.scatter(aligned_fgds, app_ratings, label='FGD', color=color, marker='o', s=80)

    # Hardcoded positions for the annotations to avoid overlapping
    for i, entry in enumerate(aligned_entries):
        x,y = [
            [0,-.01],          #SG
            [-2,.01],          #SF
            [0,0],     #SJ
            [0,0],          #SL
            [0,0],  #SE
            [0,0],    #SH
            [0,0],    #BD
            [-5,-.02],     #SD
            [0,0],          #BM
            [-5,0],          #SI
            [0,0],          #SK
            [0,0],          #SA
            [0,0],          #SB
            [0,0],          #SC
        ][i]
        ax1.annotate(entry, (aligned_fgds[i]+x, app_ratings[i]+y), fontsize=18)
    #ax.tick_params(axis='y', labelcolor=color)
    # Linear regression for the FGD
    coef = np.polyfit(aligned_fgds, app_ratings,1)
    poly1d_fn = np.poly1d(coef) 
    ax1.plot(np.sort(aligned_fgds), poly1d_fn(np.sort(aligned_fgds)), '--k', color=color)
    # Hardcoded limits to improve visualization
    ax1.set_xlim(0, 90)
    ax1.set_ylim(-0.05, 0.4)
    ax1.set_xlabel('FGD')
    ax1.set_ylabel('Appropriateness')

    # Dice Score
    color = 'tab:orange'
    ax2.scatter(aligned_dice, app_ratings, label='Dice', color=color, marker='o', s=80)

    # Hardcoded positions for the annotations to avoid overlapping
    for i, entry in enumerate(aligned_entries):
        x,y = [
            [0,-.01],          #SG
            [0,0],          #SF
            [0,0],     #SJ
            [0,0],          #SL
            [0,0],  #SE
            [0,0],    #SH
            [0,0],    #BD
            [0,0],     #SD
            [0,0],          #BM
            [0,0],          #SI
            [0,0],          #SK
            [0,0],          #SA
            [0,0],          #SB
            [0,0],          #SC
        ][i]
        ax2.annotate(entry, (aligned_dice[i]+x, app_ratings[i]+y), fontsize=20)

    # Linear regression for the Dice Score
    coef = np.polyfit(aligned_dice,app_ratings,1)
    poly1d_fn = np.poly1d(coef) 
    ax2.plot(np.sort(aligned_dice), poly1d_fn(np.sort(aligned_dice)), '--k', color=color)
    # Hardcoded limits to improve visualization
    ax2.set_xlim(0.6, 0.8)
    ax2.set_xlabel('Dice Score')
    
    return fig

def random_dices_boxplot(aligned_entries_names,
                         entries_loaders,
                         dice,
                         sort_array=np.empty(0),
                         size=(12,8),
                        ):
    fig, ax1 = plt.subplots(figsize=size)
    if sort_array.shape[0] > 0:
        # Get descending order align indeces
        sorted_inds = np.asarray(sort_array).argsort()[::-1]
    else:
        sorted_inds = np.arange(len(dice))
    # Get ordered names
    aligned_entries = aligned_entries_names[sorted_inds]
    
    boxplot_dice = []
    for i, entry in enumerate(aligned_entries):
        boxplot_dice.append(entries_loaders[entry].dataset.dice)
    x=ax1.boxplot(boxplot_dice)
    x=ax1.scatter(np.arange(1,len(dice)+1), dice[sorted_inds], c='black', marker='x')
    x=ax1.set_xticklabels(labels=aligned_entries_names[sorted_inds])
    
    return fig