import os
import numpy as np
from loader import DatasetBVHLoader
from GAC import rasterizer
from GAC.gac import SetStats
import plots
import torch
import argparse
from tabulate import tabulate
from tqdm import tqdm

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

    zeggs_dataset, styles = load_data_zeggs(fps=60, njoints=75)
    if args.load:
        zeggs_setstats = load_zeggs_gac('./GAC/output/zeggs_setstats.npy')
    else:
        zeggs_setstats = compute_gac_zeggs(zeggs_dataset, styles)        
    report_zeggs_gac(zeggs_dataset)