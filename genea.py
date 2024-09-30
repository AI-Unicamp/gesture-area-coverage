
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

def load_data_genea(genea_entries,
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
                                         window=window) if genea_trn_path is not None else None
    
    
    genea_trn_loader = DataLoader(dataset=genea_trn_dataset, batch_size=batch_size, shuffle=True) if genea_trn_path is not None else None

    # Load GENEA entries
    print('Loading GENEA entries')
    genea_entries_datasets = []
    genea_entries_loaders  = {}
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

def compute_fgd(genea_trn_loader,
                genea_entries,
                genea_entries_loaders,
                device,
                fgd_path='./FGD/output/genea_model_checkpoint_120_246.bin',
                window=120,
                NA_entry='NA',
                identifier='',
                ):
    # Load FGD model
    evaluator = EmbeddingSpaceEvaluator(fgd_path, 
                                        n_frames=window, 
                                        device=device)
    
    # Compute features for GENEA train dataset and NA entry
    print('Computing features for GENEA train dataset and NA entry (FGD)')
    genea_trn_feat, _ = evaluator.run_samples(evaluator.net, genea_trn_loader           , device) if genea_trn_loader is not None else (None, None)
    genea_NA_feat , _ = evaluator.run_samples(evaluator.net, genea_entries_loaders[NA_entry], device)

    # Compute FGDs of every entry againts the GENEA train dataset and the NA entry
    fgds = {}
    print('Computing entries FGD')
    for loader in tqdm(genea_entries_loaders):
        if loader != NA_entry:
            feat, labels = evaluator.run_samples(evaluator.net, genea_entries_loaders[loader], device)
            trn_FGD = evaluator.frechet_distance(genea_trn_feat, feat) if genea_trn_loader is not None else None
            NA_FGD = evaluator.frechet_distance(genea_NA_feat , feat)
            fgds.update( {loader: [trn_FGD, NA_FGD]} )

    # Compute FGD of the NA entry against the GENEA train dataset
    NA_trn_FGD = evaluator.frechet_distance(genea_trn_feat, genea_NA_feat) if genea_trn_loader is not None else None

    # Genea entries aligned with human-likeness ratings
    aligned_fgds_trn = [fgds[entry][0] if entry != NA_entry else NA_trn_FGD for entry in genea_entries ] if genea_trn_loader is not None else None
    aligned_fgds_NA  = [fgds[entry][1] if entry != NA_entry else 0 for entry in genea_entries]
    if genea_trn_loader is not None:
        np.save(os.path.join(os.path.dirname(fgd_path), identifier + 'aligned_fgds_trn.npy'), aligned_fgds_trn, allow_pickle=True)
    np.save(os.path.join(os.path.dirname(fgd_path), identifier + 'aligned_fgds_NA.npy' ), aligned_fgds_NA , allow_pickle=True)

    return aligned_fgds_trn, aligned_fgds_NA

def compute_gac_genea(genea_entries_datasets,
                      frame_skip = 1,
                      grid_step = 0.5,
                      weigth = 1,
                      gac_save_path='./GAC/output',
                      NA_entry='NA',
                      identifier='',):
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
        if entry.name == NA_entry : NA = entry
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
    np.save(os.path.join(gac_save_path, identifier+'genea_setstats.npy'), aligned_setstats, allow_pickle=True)

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
    
    with open(os.path.join(gac_save_path, identifier+'genea_setstats.csv'), 'w') as f:
        f.write(csv)

    return aligned_setstats

def load_fgd(fgd_output_path,
             identifier='',):
    aligned_fgds_trn = None
    if os.path.exists(os.path.join(fgd_output_path, identifier + 'aligned_fgds_trn.npy')):
        aligned_fgds_trn = np.load(os.path.join(fgd_output_path, identifier + 'aligned_fgds_trn.npy'), allow_pickle=True)
    aligned_fgds_NA  = np.load(os.path.join(fgd_output_path, identifier + 'aligned_fgds_NA.npy' ), allow_pickle=True)
    return aligned_fgds_trn, aligned_fgds_NA

def load_genea_gac(gac_output_path,
                   genea_entries_datasets = None):
    aligned_setstats = np.load(gac_output_path, allow_pickle=True)
    if genea_entries_datasets is not None:
        for i, dataset in enumerate(genea_entries_datasets):
            dataset.setstats = aligned_setstats[i]
    return aligned_setstats

def compute_correlation(func, arr1, arr2):
    rho, p_value = func(arr1, arr2)
    return rho, p_value

def report_gac_fgd(entries,
                   setstats,
                   fgds,
                   ratings,
                   plot_func=None,
                   plot_name='./figures/output.png',
                   decimal=2,
                   ):
    dice = [np.mean([entry[0] for entry in stats]) for stats in setstats]

    # Ratings vs FGD and Dice
    if plot_func:
        fig = plot_func(ratings, fgds, dice, entries)
        os.makedirs(os.path.dirname(plot_name), exist_ok=True)
        fig.savefig(plot_name)
        
    # Spearman correlation between Ratings vs FGD and Dice
    r1, p1 = compute_correlation(spearmanr, ratings, fgds)
    r2, p2 = compute_correlation(spearmanr, ratings, dice)
    table = [["Spearman (p-value)", f'{r1:.{decimal}f} ({p1:.{decimal}f})', f'{r2:.{decimal}f} ({p2:.{decimal}f})']]

    # Kendall correlation between Ratings vs FGD and Dice
    r1, p1 = compute_correlation(kendalltau, ratings, fgds)
    r2, p2 = compute_correlation(kendalltau, ratings, dice)
    table.append(["Kendall\'s tau (p-value)", f'{r1:.{decimal}f} ({p1:.{decimal}f})', f'{r2:.{decimal}f} ({p2:.{decimal}f})'])
    
    return table