import os
import numpy as np
from loader import DatasetBVHLoader
from FGD.embedding_space_evaluator import EmbeddingSpaceEvaluator
from GAC import rasterizer
from GAC.gac import SetStats
import plots
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import argparse
from tabulate import tabulate
from tqdm import tqdm


def load_data_genea2023(genea_entries,
                        step=10,
                        window=120,
                        batch_size=64,
                        genea_trn_path='./dataset/Genea2023/trn/main-agent/bvh',
                        genea_entries_path='./dataset/SubmittedGenea2023/BVH',
                        ):
    # Load GENEA train dataset
    print('Loading GENEA train dataset')
    genea_trn_dataset = DatasetBVHLoader(name='trn', 
                                   path=genea_trn_path, 
                                   load=True,
                                   step=step, 
                                   window=window)
    
    genea_trn_loader = DataLoader(dataset=genea_trn_dataset, batch_size=batch_size, shuffle=True)

    # Load GENEA entries
    print('Loading GENEA entries')
    genea_entries_path = './dataset/SubmittedGenea2023/BVH'
    genea_entries_datasets = []
    genea_entries_loaders = {}
    for entry in tqdm(genea_entries):
        path = os.path.join(genea_entries_path, entry)
        genea_entries_datasets.append(DatasetBVHLoader(name=entry,
                                                        path=path,
                                                        load=True,
                                                        step=step,
                                                        window=window))
        genea_entries_loaders.update({entry: DataLoader(dataset=genea_entries_datasets[-1], 
                                                        batch_size=batch_size, 
                                                        shuffle=False)})
    return genea_trn_loader, genea_entries_loaders, genea_entries_datasets

def load_data_zeggs(fps,
                    njoints,
                    step = 10,
                    window = 120,
                    zeggs_path='./dataset/ZEGGS/'):
    zeggs_dataset = DatasetBVHLoader(name='zeggs', 
                                     path=zeggs_path, 
                                     load=True,  #change to False to compute a save processed data
                                     pos_mean = os.path.join(zeggs_path, 'processed', 'zeggs_bvh_positions_mean.npy'),
                                     pos_std = os.path.join(zeggs_path, 'processed', 'zeggs_bvh_positions_std.npy'),
                                     rot3d_mean = os.path.join(zeggs_path, 'processed', 'zeggs_bvh_3drotations_mean.npy'),
                                     rot3d_std = os.path.join(zeggs_path, 'processed', 'zeggs_bvh_3drotations_std.npy'),
                                     step=step,
                                     window=window,
                                     fps=fps,
                                     njoints=njoints,
                                     )
    styles = ['Neutral', 'Sad', 'Happy', 'Relaxed', 'Old', 'Angry', 
          'Agreement', 'Disagreement', 'Flirty', 'Pensive', 'Scared', 
          'Distracted', 'Sarcastic', 'Threatening', 'Still', 'Laughing',
          'Sneaky', 'Tired', 'Speech']
    return zeggs_dataset, styles

def compute_fgd(genea_trn_loader,
                genea_entries,
                genea_entries_loaders,
                device,
                fgd_path='./FGD/output/genea_model_checkpoint_120_246.bin',
                window=120,
                ):
    # Load FGD model
    evaluator = EmbeddingSpaceEvaluator(fgd_path, 
                                        n_frames=window, 
                                        device=device)
    
    # Compute features for GENEA train dataset and NA entry
    print('Computing features for GENEA train dataset and NA entry (FGD)')
    genea_trn_feat, _ = evaluator.run_samples(evaluator.net, genea_trn_loader           , device)
    genea_NA_feat , _ = evaluator.run_samples(evaluator.net, genea_entries_loaders['NA'], device)

    # Compute FGDs of every entry againts the GENEA train dataset and the NA entry
    fgds = {}
    print('Computing entries FGD')
    for loader in tqdm(genea_entries_loaders):
        if loader != 'NA':
            feat, labels = evaluator.run_samples(evaluator.net, genea_entries_loaders[loader], device)
            trn_FGD = evaluator.frechet_distance(genea_trn_feat, feat)
            NA_FGD = evaluator.frechet_distance(genea_NA_feat , feat)
            fgds.update( {loader: [trn_FGD, NA_FGD]} )

    # Compute FGD of the NA entry against the GENEA train dataset
    NA_trn_FGD = evaluator.frechet_distance(genea_trn_feat, genea_NA_feat)

    # Genea entries aligned with human-likeness ratings
    aligned_fgds_trn = [fgds[entry][0] if entry != 'NA' else NA_trn_FGD for entry in genea_entries ]
    aligned_fgds_NA  = [fgds[entry][1] if entry != 'NA' else 0 for entry in genea_entries]
    np.save(os.path.join(os.path.dirname(fgd_path), 'aligned_fgds_trn.npy'), aligned_fgds_trn, allow_pickle=True)
    np.save(os.path.join(os.path.dirname(fgd_path), 'aligned_fgds_NA.npy' ), aligned_fgds_NA , allow_pickle=True)

    return aligned_fgds_trn, aligned_fgds_NA

def compute_gac_genea2023(genea_entries_datasets,
                frame_skip = 1,
                grid_step = 0.5,
                weigth = 1,
                gac_save_path='./GAC/output',):
    """
    Compute the GAC metrics for all the entries in the GENEA dataset.

    Parameters:
    genea_entries_datasets: list
        List of the entries in the GENEA dataset.
    frame_skip: int
        Number of frames to skip when rasterizing the poses. Increase to improve performance. If 1, every frame will be used; if 2, every other frame will be used, and so on. Default is 1.
    grid_step: float
        Resolution of the grid, indicates the pixel. The default is 0.5, i.e., 2 pixels per unit. (*)
    weigth: float
        Weigth of each pixel occurance. Default is 1.
    """
    # Selecting Natural Motion entry and creating the GAC-based stats list for each dataset
    for entry in genea_entries_datasets:
        if entry.name == 'NA' : NA = entry
        entry.setstats = []

    print('Computing each take\'s GAC for every entry')
    # For each take of the Natural Motion entry/set (ground truth)
    for take in range(len(NA.pos)):
        print(f'Take {take}:')
        # Some takes have different lengths, so we take the minimum length
        cut = np.min([entry_dataset.pos[take].shape[0] for entry_dataset in genea_entries_datasets]+[NA.pos[take].shape[0]])

        # Rasterize NA
        nagrid = rasterizer.rasterize(NA.posLckHips()[take][:cut],
                                      NA.parents,
                                      frame_skip=frame_skip,
                                      grid_step=grid_step,
                                      weigth=weigth)
        # Clip the grid to 0-1
        nagrid.clipped = np.clip(nagrid.grid, 0, 1)
        
        # For each entry, rasterize the poses and compute the dice score
        for entry_dataset in tqdm(genea_entries_datasets):
            if entry_dataset.name != 'NA':
                grid = rasterizer.rasterize(entry_dataset.posLckHips()[take][:cut], entry_dataset.parents, frame_skip=frame_skip, grid_step=grid_step, weigth=weigth)
                grid.clipped = np.clip(grid.grid, 0, 1)
                stats = SetStats(nagrid.clipped, grid.clipped)
                entry_dataset.setstats.append(stats)

    # Genea entries aligned with human-likeness ratings
    aligned_setstats = [entry.setstats for entry in genea_entries_datasets]
    if not os.path.exists(os.path.dirname(gac_save_path)):
        os.makedirs(os.path.dirname(gac_save_path))
    np.save(os.path.join(gac_save_path, 'genea_setstats.npy'), aligned_setstats, allow_pickle=True)

    # Save as csv
    csv = f'entry,take,dice,fpfn_ratio,log_ratio,s1,s2,tp,fp,fn\n'
    for i, entry in enumerate(genea_entries_datasets):
        for j, stats in enumerate(aligned_setstats[i]):
            csv += f'{entry.name},{j},{stats[0]},{stats[1]},{stats[2]},{stats[3]},{stats[4]},{stats[5]},{stats[6]},{stats[7]}\n'

    csv += '\nEntry Average\n'
    csv += f'entry,dice,dice_std,fpfn_ratio,fpfn_ratio_std,log_ratio,log_ratio_std,s1,s1_std,s2,s2_std,tp,tp_std,fp,fp_std,fn,fn_std\n'
    for i, entry in enumerate(genea_entries_datasets):
        setstats = entry.setstats
        csv += f'{entry.name},{np.mean([entry[0] for entry in setstats])},{np.std([entry[0] for entry in setstats])},{np.mean([entry[1] for entry in setstats])},{np.std([entry[1] for entry in setstats])},{np.mean([entry[2] for entry in setstats])},{np.std([entry[2] for entry in setstats])},{np.mean([entry[3] for entry in setstats])},{np.std([entry[3] for entry in setstats])},{np.mean([entry[4] for entry in setstats])},{np.std([entry[4] for entry in setstats])},{np.mean([entry[5] for entry in setstats])},{np.std([entry[5] for entry in setstats])},{np.mean([entry[6] for entry in setstats])},{np.std([entry[6] for entry in setstats])},{np.mean([entry[7] for entry in setstats])},{np.std([entry[7] for entry in setstats])}\n'
    
    with open(os.path.join(gac_save_path, 'genea_setstats.csv'), 'w') as f:
        f.write(csv)

    return aligned_setstats

def compute_gac_zeggs(zeggs_dataset,
                      styles,
                      frame_skip = 1,
                      grid_step = 0.5,
                      weigth = 1,
                      gac_save_path='./GAC/output',
                      ):
    
    # Computing neutral grids
    neutral_styles_grids = []
    for i in range(5):
        neutral_styles_grids.append(rasterizer.rasterize(zeggs_dataset.posLckHips()[i]-[0,-100,0], 
                                                         zeggs_dataset.parents, 
                                                         frame_skip=frame_skip, 
                                                         grid_step=grid_step, 
                                                         weigth=weigth))
        neutral_styles_grids[-1].clipped = np.clip(neutral_styles_grids[-1].grid, 0, 1)

    # For each style (1), get the gac of each take in the given style (2), and compute the statistics for each neutral take (3)

    zeggs_setstats = {style: [] for style in styles[1:]}
    csv = f'neutral_take,style_take,dice,fpfn_ratio,log_ratio,s1,s2,tp,fp,fn\n'
    # For each style (1)
    for style_num, style in enumerate(styles[1:], start=1):
        print(f'Computing style {style_num}: {style}')
        mean_style_setstats = []
        sum_style_setstats = []
        
        # For each take in the given style (2)
        # Search for the take (I'm doing like this because files might not be ordered)
        for take_idx, take in enumerate(zeggs_dataset.files[5:], start=5):
            if take.split('_')[1] == style:
                print(take)
                # Get the gac
                style_grid = rasterizer.rasterize(zeggs_dataset.posLckHips()[take_idx]-[0,-100,0], 
                                                  zeggs_dataset.parents, 
                                                  frame_skip=frame_skip, 
                                                  grid_step=grid_step, 
                                                  weigth=weigth)
                style_grid.clipped = np.clip(style_grid.grid, 0, 1)
                
                # Compute statistics for each neutral take (3)
                aux_setstats = []
                for neutral_grid, neutral_styles_grid in enumerate(neutral_styles_grids):
                    aux_setstats.append(SetStats(neutral_styles_grid.clipped, style_grid.clipped))
                    csv += f'{neutral_grid},{style+str(take_idx)},{aux_setstats[-1][0]},{aux_setstats[-1][1]},{aux_setstats[-1][2]},{aux_setstats[-1][3]},{aux_setstats[-1][4]},{aux_setstats[-1][5]},{aux_setstats[-1][6]},{aux_setstats[-1][7]}\n'

                # Get the mean and sum of the statistics of this take for each neutral take
                mean_style_setstats.append(np.mean(aux_setstats, axis = 0))
                sum_style_setstats.append(np.sum(aux_setstats, axis = 0)/zeggs_dataset.pos[take_idx].shape[0])
                
        # For every take of the given style, get the mean and std of the statistics
        zeggs_setstats[style] = [np.mean(mean_style_setstats, axis=0), 
                                 np.std(mean_style_setstats, axis=0) ,
                                 np.mean(sum_style_setstats, axis=0) , 
                                 np.std(sum_style_setstats, axis=0)  ]
        
    np.save(os.path.join(gac_save_path, 'zeggs_setstats.npy'), zeggs_setstats, allow_pickle=True)

    csv += '\nstyle,dice,dice_std,fpfn_ratio,fpfn_ratio_std,log_ratio,log_ratio_std,s1,s1_std,s2,s2_std,tp,tp_std,fp,fp_std,fn,fn_std\n'
    for style in zeggs_setstats:
        mean, std, sum_mean, sum_std = zeggs_setstats[style]
        csv += f'{style},{mean[0]},{std[0]},{mean[1]},{std[1]},{mean[2]},{std[2]},{mean[3]},{std[3]},{mean[4]},{std[4]},{mean[5]},{std[5]},{mean[6]},{std[6]},{mean[7]},{std[7]}\n'
    
    csv += '\nNormalized\n'
    for style in zeggs_setstats:
        mean, std, sum_mean, sum_std = zeggs_setstats[style]
        csv += f'{style},{sum_mean[0]},{sum_std[0]},{sum_mean[1]},{sum_std[1]},{sum_mean[2]},{sum_std[2]},{sum_mean[3]},{sum_std[3]},{sum_mean[4]},{sum_std[4]},{sum_mean[5]},{sum_std[5]},{sum_mean[6]},{sum_std[6]},{sum_mean[7]},{sum_std[7]}\n'

    # Save as csv
    with open(os.path.join(gac_save_path, 'zeggs_setstats.csv'), 'w') as f:
        f.write(csv)

    return zeggs_setstats

def compute_correlation(func, arr1, arr2):
    rho, p_value = func(arr1, arr2)
    return rho, p_value

def report_gac_fgd(entries,
                   setstats,
                   fgds,
                   ratings,
                   plot_func=None,
                   plot_name='./figures/output.png'
                   ):
    dice = [np.mean([entry[0] for entry in stats]) for stats in setstats]

    # Ratings vs FGD and Dice
    if plot_func:
        #fig = plots.plot_HL_versus_dice_fgd(ratings, fgds, dice, entries)
        fig = plot_func(ratings, fgds, dice, entries)
        os.makedirs(os.path.dirname(plot_name), exist_ok=True)
        fig.savefig(plot_name)

    # Spearman correlation between HL Median vs FGD and Dice
    r1, p1 = compute_correlation(spearmanr, ratings, fgds)
    r2, p2 = compute_correlation(spearmanr, ratings, dice)
    table = [["Spearman (p-value)", f'{r1:.2f} ({p1:.2f})', f'{r2:.2f} ({p2:.2f})']]

    # Kendall correlation between HL Median vs FGD and Dice
    r1, p1 = compute_correlation(kendalltau, ratings, fgds)
    r2, p2 = compute_correlation(kendalltau, ratings, dice)
    table.append(["Kendall\'s tau (p-value)", f'{r1:.2f} ({p1:.2f})', f'{r2:.2f} ({p2:.2f})'])
    
    return table

def load_fgd(fgd_output_path):
    aligned_fgds_trn = np.load(os.path.join(fgd_output_path, 'aligned_fgds_trn.npy'), allow_pickle=True)
    aligned_fgds_NA  = np.load(os.path.join(fgd_output_path, 'aligned_fgds_NA.npy' ), allow_pickle=True)
    return aligned_fgds_trn, aligned_fgds_NA

def load_genea_gac(gac_output_path,
                   genea_entries_datasets = None):
    aligned_setstats = np.load(gac_output_path, allow_pickle=True)
    if genea_entries_datasets is not None:
        for i, dataset in enumerate(genea_entries_datasets):
            dataset.setstats = aligned_setstats[i]
    return aligned_setstats

def load_zeggs_gac(gac_output_path):
    zeggs_setstats = np.load(gac_output_path, allow_pickle=True)
    return zeggs_setstats.item()

def report_zeggs_gac(zeggs_dataset):
    # Plot
    fig = plots.plot_zeggs_gac_comparison(zeggs_dataset,
                                          rasterizer)
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    fig.savefig('./figures/zeggs_gac_comparison.png')

if __name__ == '__main__':
    # This script will compute the FGD and GAC metrics for the GENEA2023 and ZEGGS datasets.
    # Should take about 3 hours to run.
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', action='store_true', help='Include this argument to load the precomputed FGD and GAC metrics.')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # GENEA 2023 entries aligned with human-likeness median ratings
    genea23_aligned_entries = ['NA', 'SG', 'SF', 'SJ', 'SL', 'SE', 'SH', 'BD', 'SD', 'BM', 'SI', 'SK', 'SA', 'SB', 'SC']
    hl_median               = [  71,   69,   65,   51,   51,   50,   46,   46,   45,   43,   40,   37,   30,   24,    9]
    hl_mean                 = [68.4, 65.6, 63.6, 51.8, 50.6, 50.9, 45.1, 45.3, 44.7, 42.9, 41.4, 40.2, 32.0, 27.4, 11.6]

    genea_trn_loader, genea_entries_loaders, genea_entries_datasets = load_data_genea2023(genea23_aligned_entries)
    if args.load:
        aligned_fgds_trn, aligned_fgds_NA = load_fgd('./FGD/output')
        aligned_setstats = load_genea_gac('./GAC/output/genea_setstats.npy', genea_entries_datasets)
    else: # Set to True to recompute the FGD and GAC metrics
        aligned_fgds_trn, aligned_fgds_NA = compute_fgd(genea_trn_loader, genea23_aligned_entries, genea_entries_loaders, device)
        aligned_setstats = compute_gac_genea2023(genea_entries_datasets)
        
    # Compute correlation between HL Median and FGD and Dice. Print table with results.
    table = report_gac_fgd(genea23_aligned_entries[1:], aligned_setstats[1:], aligned_fgds_NA[1:], hl_median[1:], plots.plot_HL_versus_dice_fgd, './figures/fgd_vs_dice.png')
    header = ["Correlation", "FGD vs Hum. Median", "Dice vs Hum. Median"]
    print(tabulate(table, header, tablefmt="github"))
    # Compute correlation between HL Mean and FGD and Dice. Print table with results.
    table = report_gac_fgd(genea23_aligned_entries[1:], aligned_setstats[1:], aligned_fgds_NA[1:], hl_mean[1:]  , plots.plot_HL_versus_dice_fgd, './figures/fgd_vs_dice_mean.png')
    header = ["Correlation", "FGD vs Hum. Mean", "Dice vs Hum. Mean"]
    print(tabulate(table, header, tablefmt="github"))

    zeggs_dataset, styles = load_data_zeggs(fps=60, njoints=75)
    if args.load:
        zeggs_setstats = load_zeggs_gac('./GAC/output/zeggs_setstats.npy')
    else:
        zeggs_setstats = compute_gac_zeggs(zeggs_dataset, styles)        
    report_zeggs_gac(zeggs_dataset)