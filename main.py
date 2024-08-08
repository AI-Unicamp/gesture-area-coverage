import os
import numpy as np
from loader import DatasetBVHLoader
from FGD.embedding_space_evaluator import EmbeddingSpaceEvaluator
from GAC import rasterizer
from GAC.gac import DiceScore
import torch
from torch.utils.data import DataLoader

GENEA_ENTRIES   = ['BD', 'BM', 'NA', 'SA', 'SB', 'SC', 'SD', 'SE', 'SF', 'SG', 'SH', 'SI', 'SJ', 'SK', 'SL']
ALIGNED_ENTRIES = ['NA', 'SG', 'SF', 'SJ', 'SL', 'SE', 'SH', 'BD', 'SD', 'BM', 'SI', 'SK', 'SA', 'SB', 'SC']
HL_MEDIAN       = [  71,   69,   65,   51,   51,   50,   46,   46,   45,   43,   40,   37,   30,   24,    9]

def load_data(step,
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

def compute_gac(genea_entries_datasets,
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
                dice = DiceScore(nagrid.clipped, grid.clipped)
                entry_dataset.setstats.append(dice)
                print(f'{entry_dataset.name} dice: {dice[0]:.2f}')

    # Genea entries aligned with human-likeness ratings
    aligned_setstats = [entry.setstats for entry in genea_entries_datasets]
    aligned_setstats = [aligned_setstats[GENEA_ENTRIES.index(entry)] for entry in ALIGNED_ENTRIES]
    if not os.path.exists(os.path.dirname(gac_save_path)):
        os.makedirs(os.path.dirname(gac_save_path))
    np.save(gac_save_path, aligned_setstats, allow_pickle=True)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    step = 10
    window = 120
    batch_size = 64
    genea_trn_loader, genea_entries_loaders, genea_entries_datasets = load_data(step, window, batch_size)
    compute_fgd(genea_trn_loader, genea_entries_loaders)
    compute_gac(genea_entries_datasets)