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

GENEA_ENTRIES   = ['BD', 'BM', 'NA', 'SA', 'SB', 'SC', 'SD', 'SE', 'SF', 'SG', 'SH', 'SI', 'SJ', 'SK', 'SL']
ALIGNED_ENTRIES = ['NA', 'SG', 'SF', 'SJ', 'SL', 'SE', 'SH', 'BD', 'SD', 'BM', 'SI', 'SK', 'SA', 'SB', 'SC']
HL_MEDIAN       = [  71,   69,   65,   51,   51,   50,   46,   46,   45,   43,   40,   37,   30,   24,    9]
HL_MEAN         = [68.4, 65.6, 63.6, 51.8, 50.6, 50.9, 45.1, 45.3, 44.7, 42.9, 41.4, 40.2, 32.0, 27.4, 11.6]

def load_data_genea2023(step,
                        window,
                        batch_size,
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
    for entry in GENEA_ENTRIES:
        print(f'Loading {entry}...')
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

def load_data_zeggs(step,
                    window,
                    fps,
                    njoints,
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
                genea_entries_loaders,
                fgd_path='./FGD/output/genea_model_checkpoint_120_246.bin',
                ):
    # Load FGD model
    evaluator = EmbeddingSpaceEvaluator(fgd_path, 
                                        n_frames=window, 
                                        device=device)
    
    # Compute features for GENEA train dataset and NA entry
    print('Computing features for GENEA train dataset and NA entry')
    genea_trn_feat, _ = evaluator.run_samples(evaluator.net, genea_trn_loader           , device)
    genea_NA_feat , _ = evaluator.run_samples(evaluator.net, genea_entries_loaders['NA'], device)

    # Compute FGDs of every entry againts the GENEA train dataset and the NA entry
    fgds = {}
    for loader in genea_entries_loaders:
        print(f'Computing FGD for {loader}...')
        if loader != 'NA':
            feat, labels = evaluator.run_samples(evaluator.net, genea_entries_loaders[loader], device)
            trn_FGD = evaluator.frechet_distance(genea_trn_feat, feat)
            NA_FGD = evaluator.frechet_distance(genea_NA_feat , feat)
            fgds.update( {loader: [trn_FGD, NA_FGD]} )

    # Compute FGD of the NA entry against the GENEA train dataset
    NA_trn_FGD = evaluator.frechet_distance(genea_trn_feat, genea_NA_feat)

    # Genea entries aligned with human-likeness ratings
    aligned_fgds_trn = [fgds[loader][0] for loader in ALIGNED_ENTRIES[1:]]
    aligned_fgds_NA  = [fgds[loader][1] for loader in ALIGNED_ENTRIES[1:]]
    np.save(os.path.join(os.path.dirname(fgd_path), 'aligned_fgds_trn.npy'), aligned_fgds_trn, allow_pickle=True)
    np.save(os.path.join(os.path.dirname(fgd_path), 'aligned_fgds_NA.npy' ), aligned_fgds_NA , allow_pickle=True)

    # Print
    print(f'FGD (NA, TRN): {NA_trn_FGD:.2f}. HL: {HL_MEDIAN[0]}')
    for i, entry in enumerate(ALIGNED_ENTRIES[1:]):
        print(f'FGD (NA, {entry}): {aligned_fgds_NA[i]:.2f}. FGD (TRN, {entry}): {aligned_fgds_trn[i]:.2f}. HL: {HL_MEDIAN[i+1]}')

    return aligned_fgds_trn, aligned_fgds_NA

def compute_gac_genea2023(genea_entries_datasets,
                frame_skip = 1,
                grid_step = 0.5,
                weigth = 1,
                gac_save_path='./GAC/output/genea_setstats.npy',):
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
    NA = genea_entries_datasets[2]

    # Creating the dice list for each dataset
    for entry_dataset in genea_entries_datasets:
        entry_dataset.setstats = []

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
        for entry_dataset in genea_entries_datasets:
            if entry_dataset.name != 'NA':
                grid = rasterizer.rasterize(entry_dataset.posLckHips()[take][:cut], entry_dataset.parents, frame_skip=frame_skip, grid_step=grid_step, weigth=weigth)
                grid.clipped = np.clip(grid.grid, 0, 1)
                stats = SetStats(nagrid.clipped, grid.clipped)
                entry_dataset.setstats.append(stats)
                print(f'{entry_dataset.name} dice: {stats[0]:.2f}')

    # Genea entries aligned with human-likeness ratings
    aligned_setstats = [entry.setstats for entry in genea_entries_datasets]
    aligned_setstats = [aligned_setstats[GENEA_ENTRIES.index(entry)] for entry in ALIGNED_ENTRIES]
    if not os.path.exists(os.path.dirname(gac_save_path)):
        os.makedirs(os.path.dirname(gac_save_path))
    np.save(gac_save_path, aligned_setstats, allow_pickle=True)

    # Save as csv
    csv = f'entry,take,dice,fpfn_ratio,log_ratio,s1,s2,tp,fp,fn\n'
    for i, entry in enumerate(ALIGNED_ENTRIES):
        for j, stats in enumerate(aligned_setstats[i]):
            csv += f'{entry},{j},{stats[0]},{stats[1]},{stats[2]},{stats[3]},{stats[4]},{stats[5]},{stats[6]},{stats[7]}\n'
    with open(f'./GAC/output/genea_setstats.csv', 'w') as f:
        f.write(csv)

    return aligned_setstats

def compute_gac_zeggs(zeggs_dataset,
                      styles,
                      frame_skip = 1,
                      grid_step = 0.5,
                      weigth = 1,
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
        
    csv += '\nstyle,dice,dice_std,fpfn_ratio,fpfn_ratio_std,log_ratio,log_ratio_std,s1,s1_std,s2,s2_std,tp,tp_std,fp,fp_std,fn,fn_std\n'
    for style in zeggs_setstats:
        mean, std, sum_mean, sum_std = zeggs_setstats[style]
        csv += f'{style},{mean[0]},{std[0]},{mean[1]},{std[1]},{mean[2]},{std[2]},{mean[3]},{std[3]},{mean[4]},{std[4]},{mean[5]},{std[5]},{mean[6]},{std[6]},{mean[7]},{std[7]}\n'
    
    csv += '\nNormalized\n'
    for style in zeggs_setstats:
        mean, std, sum_mean, sum_std = zeggs_setstats[style]
        csv += f'{style},{sum_mean[0]},{sum_std[0]},{sum_mean[1]},{sum_std[1]},{sum_mean[2]},{sum_std[2]},{sum_mean[3]},{sum_std[3]},{sum_mean[4]},{sum_std[4]},{sum_mean[5]},{sum_std[5]},{sum_mean[6]},{sum_std[6]},{sum_mean[7]},{sum_std[7]}\n'

    # Save as csv
    with open(f'./GAC/output/zeggs_setstats.csv', 'w') as f:
        f.write(csv)

    return zeggs_setstats

def compute_correlation(func, arr1, arr2):
    rho, p_value = func(arr1, arr2)
    print(f'Correlation: {rho:.2f}. p-value: {p_value:.2f}')

def report_gac_fgd(aligned_setstats,
                   aligned_fgds_trn,
                   aligned_fgds_NA,
                   ):
    aligned_dice = [np.mean([entry[0] for entry in setstats]) for setstats in aligned_setstats]

    # HL Median vs FGD and Dice
    fig = plots.plot_HL_versus_dice_fgd(HL_MEDIAN, aligned_fgds_NA, aligned_dice, ALIGNED_ENTRIES)
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    fig.savefig('./figures/fgd_vs_dice.png')

    # HL Mean vs FGD and Dice
    fig = plots.plot_HL_versus_dice_fgd(HL_MEAN, aligned_fgds_NA, aligned_dice, ALIGNED_ENTRIES)
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    fig.savefig('./figures/fgd_vs_dice_mean.png')

    print(aligned_fgds_NA)
    print(aligned_dice)

    # Spearman correlation between HL Median vs FGD and Dice
    print('Spearman correlation between FGD and HL Median:')
    compute_correlation(spearmanr, HL_MEDIAN[1:], aligned_fgds_NA)
    print('Spearman correlation between Dice and HL Median:')
    compute_correlation(spearmanr, HL_MEDIAN[1:], aligned_dice[1:])

    # Kendall correlation between HL Median vs FGD and Dice
    print('Kendall\'s tau  correlation between FGD and HL Median:')
    compute_correlation(kendalltau, HL_MEDIAN[1:], aligned_fgds_NA)
    print('Kendall\'s tau correlation between Dice and HL Median:')
    compute_correlation(kendalltau, HL_MEDIAN[1:], aligned_dice[1:])
    
    # Spearman correlation between HL Mean vs FGD and Dice
    print('Spearman correlation between FGD and HL Mean:')
    compute_correlation(spearmanr, HL_MEAN[1:], aligned_fgds_NA)
    print('Spearman correlation between Dice and HL Mean:')
    compute_correlation(spearmanr, HL_MEAN[1:], aligned_dice[1:])

    # Kendall correlation between HL Mean vs FGD and Dice
    print('Kendall\'s tau correlation between FGD and HL Mean:')
    compute_correlation(kendalltau, HL_MEAN[1:], aligned_fgds_NA)
    print('Kendall\'s tau correlation between Dice and HL Mean:')
    compute_correlation(kendalltau, HL_MEAN[1:], aligned_dice[1:])

def load_fgd(fgd_output_path):
    aligned_fgds_trn = np.load(os.path.join(fgd_output_path, 'aligned_fgds_trn.npy'), allow_pickle=True)
    aligned_fgds_NA  = np.load(os.path.join(fgd_output_path, 'aligned_fgds_NA.npy' ), allow_pickle=True)
    return aligned_fgds_trn, aligned_fgds_NA

def load_gac(gac_output_path,
             genea_entries_datasets = None):
    aligned_setstats = np.load(gac_output_path, allow_pickle=True)
    if genea_entries_datasets is not None:
        for i, entry in enumerate(ALIGNED_ENTRIES):
            genea_entries_datasets[GENEA_ENTRIES.index(entry)].setstats = aligned_setstats[i]
    return aligned_setstats

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    step = 10
    window = 120
    batch_size = 64
    genea_trn_loader, genea_entries_loaders, genea_entries_datasets = load_data_genea2023(step, window, batch_size)
    if False: # Set to True to recompute the FGD and GAC metrics
        aligned_fgds_trn, aligned_fgds_NA = compute_fgd(genea_trn_loader, genea_entries_loaders)
        aligned_setstats = compute_gac_genea2023(genea_entries_datasets)
    else:
        aligned_fgds_trn, aligned_fgds_NA = load_fgd('./FGD/output')
        aligned_setstats = load_gac('./GAC/output/genea_setstats.npy', genea_entries_datasets)
    report_gac_fgd(aligned_setstats, aligned_fgds_trn, aligned_fgds_NA)

    fps=60
    njoints = 75
    zeggs_dataset, styles = load_data_zeggs(step, window, fps, njoints)
    zeggs_setstats = compute_gac_zeggs(zeggs_dataset, styles)