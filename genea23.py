from genea import load_data_genea, compute_fgd, compute_gac_genea, load_fgd, load_genea_gac, report_gac_fgd
import torch
from tabulate import tabulate
import argparse
import plots


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
    app_mas                 = [0.83, 0.39, 0.20, 0.27, 0.05, 0.16, 0.09, 0.14, 0.14, 0.20, 0.16, 0.18, 0.11, 0.13,-0.02]
    app_pref_match          = [76.6, 61.5, 56.4, 61.3, 55.2, 58.6, 52.9, 56.1, 57.6, 60.0, 55.4, 56.7, 54.8, 56.2, 44.9]

    genea_trn_loader, genea_entries_loaders, genea_entries_datasets = load_data_genea(genea23_aligned_entries)
    if args.load:
        aligned_fgds_trn, aligned_fgds_NA = load_fgd('./FGD/output')
        aligned_setstats = load_genea_gac('./GAC/output/genea_setstats.npy', genea_entries_datasets)
    else: # Set to True to recompute the FGD and GAC metrics
        aligned_fgds_trn, aligned_fgds_NA = compute_fgd(genea_trn_loader, genea23_aligned_entries, genea_entries_loaders, device)
        aligned_setstats = compute_gac_genea(genea_entries_datasets)
        
    # Compute correlation between HL Median and FGD and Dice. Print table with results.
    table = report_gac_fgd(genea23_aligned_entries[1:], aligned_setstats[1:], aligned_fgds_NA[1:], hl_median[1:], plots.plot_HL_versus_dice_fgd, './figures/fgd_vs_dice.png')
    header = ["Correlation", "FGD vs Hum. Median", "Dice vs Hum. Median"]
    print(tabulate(table, header, tablefmt="github"))
    
    # Compute correlation between HL Mean and FGD and Dice. Print table with results.
    table = report_gac_fgd(genea23_aligned_entries[1:], aligned_setstats[1:], aligned_fgds_NA[1:], hl_mean[1:]  , plots.plot_HL_versus_dice_fgd, './figures/fgd_vs_dice_mean.png')
    header = ["Correlation", "FGD vs Hum. Mean", "Dice vs Hum. Mean"]
    print(tabulate(table, header, tablefmt="github"))

    # Compute correlation between Appropriateness and FGD and Dice. Print table with results.
    table = report_gac_fgd(genea23_aligned_entries[1:], aligned_setstats[1:], aligned_fgds_NA[1:], app_mas[1:], plots.plot_APP_MAS_versus_dice_fgd, './figures/fgd_vs_dice_appropriateness_mas.png')
    header = ["Correlation", "FGD vs App. MAS", "Dice vs App. MAS"]
    print(tabulate(table, header, tablefmt="github"))

    # Compute correlation between Appropriateness and FGD and Dice. Print table with results.
    table = report_gac_fgd(genea23_aligned_entries[1:], aligned_setstats[1:], aligned_fgds_NA[1:], app_pref_match[1:], plots.plot_APP_versus_dice_fgd, './figures/fgd_vs_dice_appropriateness.png')
    header = ["Correlation", "FGD vs App. Pref.", "Dice vs App. Pref."]
    print(tabulate(table, header, tablefmt="github"))