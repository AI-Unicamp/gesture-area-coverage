import os
from loader import DatasetBVHLoader
from FGD.embedding_space_evaluator import EmbeddingSpaceEvaluator
import torch
from torch.utils.data import DataLoader

GENEA_ENTRIES   = ['BD', 'BM', 'NA', 'SA', 'SB', 'SC', 'SD', 'SE', 'SF', 'SG', 'SH', 'SI', 'SJ','SK','SL']
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

    # Genea entries aligner with human-likeness ratings
    aligned_fgds_trn = [fgds[loader][0] for loader in ALIGNED_ENTRIES[1:]]
    aligned_fgds_NA  = [fgds[loader][1] for loader in ALIGNED_ENTRIES[1:]]

    # Print
    print(f'FGD (NA, TRN): {NA_trn_FGD:.2f}. HL: {HL_MEDIAN[0]}')
    for i, entry in enumerate(ALIGNED_ENTRIES[1:]):
        print(f'FGD (NA, {entry}): {aligned_fgds_NA[i]:.2f}. FGD (TRN, {entry}): {aligned_fgds_trn[i]:.2f}. HL: {HL_MEDIAN[i+1]}')    

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    step = 10
    window = 120
    batch_size = 64
    genea_trn_loader, genea_entries_loaders, genea_entries_datasets = load_data(step, window, batch_size)
    compute_fgd(genea_trn_loader, genea_entries_loaders)