from genea import load_data_genea, compute_fgd, load_fgd, load_genea_gac, report_gac_fgd
import torch
from tabulate import tabulate
import argparse
import plots
from tqdm import tqdm
from GAC import rasterizer
from GAC.gac import SetStats
import numpy as np
import os

def mismatch_gac_genea(genea_entries_datasets,
                       frame_skip = 1,
                       grid_step = 0.5,
                       weigth = 1,
                       gac_save_path='./GAC/output',
                       runs=100):
    
    
    NA_grids = get_gac_grids(genea_entries_datasets, frame_skip, grid_step, weigth, gac_save_path)
    print('Running mismatch GAC...')
    for i in tqdm(range(runs)):
        run_mismatch_gac(NA_grids, genea_entries_datasets)

    # Save as csv
    csv = f'entry,dice\n'
    for i, entry in enumerate(genea_entries_datasets[1:]):
        csv+=f'{entry.name}'
        for d in entry.dice:
            csv+=f',{d}'
        csv += '\n'
        
    # If the file already exists, append the new data
    if os.path.exists(os.path.join(gac_save_path, f'genea_mismatch_dice_1.csv')):
        with open(os.path.join(gac_save_path, f'genea_mismatch_dice_1.csv'), 'r') as f:
            csv = f.read() + '\n' + csv
    with open(os.path.join(gac_save_path, f'genea_mismatch_dice_1.csv'), 'w') as f:
        f.write(csv)

    for entry in genea_entries_datasets[:1]:
        print(f'{entry.name} dice: {entry.dice}')


def get_gac_grids(genea_entries_datasets,
                  frame_skip = 1,
                  grid_step = 0.5,
                  weigth = 1,
                  gac_save_path='./GAC/output',):
    """
    Same as compute_gac_genea2023, but it computes the GAC metrics for the mismatched takes of the GENEA dataset.
    """
    NA = genea_entries_datasets[2]

    load_failed = False
    # Creating the dice list for each dataset
    for entry_dataset in genea_entries_datasets:
        entry_dataset.mismatch_setstats = []
        entry_dataset.grids = []
        entry_dataset.running_dice = []
        entry_dataset.dice = []

        grid_path = os.path.join(gac_save_path, f'{entry_dataset.name}_grid.npy')
        if os.path.exists(grid_path):
            entry_dataset.grids = np.load(grid_path)
            print(f'Loaded {entry_dataset.name} grid from {grid_path}')
        else:
            load_failed = True
    if not load_failed:
        return NA.grids

    NA_grids = []
    print('Computing NA GAC...')
    for take in tqdm(range(len(NA.pos))):
        # Rasterize NA
        grid = rasterizer.rasterize(NA.posLckHips()[take],
                                    NA.parents,
                                    frame_skip=frame_skip,
                                    grid_step=grid_step,
                                    weigth=weigth)
        # Clip the grid to 0-1
        NA_grids.append(np.clip(grid.grid, 0, 1))
    NA.grids = NA_grids

    # For each entry, rasterize the poses and compute the dice score
    print('Computing GAC for each take and entry...')
    for take in tqdm(range(len(NA.pos))):
        for entry_dataset in genea_entries_datasets:
            if entry_dataset.name != 'NA':
                grid = rasterizer.rasterize(entry_dataset.posLckHips()[take],
                                            entry_dataset.parents,
                                            frame_skip=frame_skip,
                                            grid_step=grid_step,
                                            weigth=weigth)
                entry_dataset.grids.append(np.clip(grid.grid, 0, 1))

    for entry_dataset in genea_entries_datasets:
        np.save(os.path.join(gac_save_path, f'{entry_dataset.name}_grid'), entry_dataset.grids)

    return NA_grids


def run_mismatch_gac(NA_grids, genea_entries_datasets):
    for take in range(len(NA_grids)):
        for entry_dataset in genea_entries_datasets:
            if entry_dataset.name != 'NA':
                # Select a random take from NA except the current take
                i = take
                while i == take:
                    i = np.random.randint(0,len(NA_grids))
                stats = SetStats(NA_grids[i], entry_dataset.grids[take])
                entry_dataset.mismatch_setstats.append(stats)
                #print(f'{entry_dataset.name} dice: {stats[0]:.2f}')
                entry_dataset.running_dice.append(stats[0])
    for entry_dataset in genea_entries_datasets[1:]:
        entry_dataset.dice.append(np.mean(entry_dataset.running_dice))
        entry_dataset.running_dice = []

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--runs', type=int, default=100, help='Number of times to compute random GAC.')
    args = parser.parse_args()


    # GENEA 2023 entries aligned with human-likeness median ratings
    genea23_aligned_entries = ['NA', 'SG', 'SF', 'SJ', 'SL', 'SE', 'SH', 'BD', 'SD', 'BM', 'SI', 'SK', 'SA', 'SB', 'SC']
    hl_median               = [  71,   69,   65,   51,   51,   50,   46,   46,   45,   43,   40,   37,   30,   24,    9]
    hl_mean                 = [68.4, 65.6, 63.6, 51.8, 50.6, 50.9, 45.1, 45.3, 44.7, 42.9, 41.4, 40.2, 32.0, 27.4, 11.6]
    app_mas                 = [0.83, 0.39, 0.20, 0.27, 0.05, 0.16, 0.09, 0.14, 0.14, 0.20, 0.16, 0.18, 0.11, 0.13,-0.02]
    app_pref_match          = [76.6, 61.5, 56.4, 61.3, 55.2, 58.6, 52.9, 56.1, 57.6, 60.0, 55.4, 56.7, 54.8, 56.2, 44.9]
    dice                    = [0.7565510086644295,0.7728359292029654,0.7179384373290701,0.6794268446896473,0.714871738244022,0.6480408418998522,0.6560084953813459,0.6954398556697892,0.7058366159755292,0.7384622668653708,0.7246973956351498,0.6614790981651663,0.6805378340395966,0.6371091619039576]

    genea_trn_loader, genea_entries_loaders, genea_entries_datasets = load_data_genea(genea23_aligned_entries)

    aligned_setstats = mismatch_gac_genea(genea_entries_datasets, runs=args.runs)

    fig = plots.random_dices_boxplot(np.asarray(genea23_aligned_entries[1:]),
                                     genea_entries_loaders,
                                     np.asarray(dice),
                                     #np.asarray(app_pref_match[1:]),
                                     )
    fig.savefig(f'./figures/random_exp_{args.runs}runs_humlike_aligned.png')

    fig = plots.random_dices_boxplot(np.asarray(genea23_aligned_entries[1:]),
                                     genea_entries_loaders,
                                     np.asarray(dice),
                                     np.asarray(app_mas[1:]),
                                     )
    fig.savefig(f'./figures/random_exp_{args.runs}runs_appmas_aligned.png')

    fig = plots.random_dices_boxplot(np.asarray(genea23_aligned_entries[1:]),
                                     genea_entries_loaders,
                                     np.asarray(dice),
                                     np.asarray(app_pref_match[1:]),
                                     )
    fig.savefig(f'./figures/random_exp_{args.runs}runs_apppref_aligned.png')

    print(f'Dice means: {[np.mean(genea_entries_loaders[entry].dataset.dice) for entry in genea23_aligned_entries[1:]]}')