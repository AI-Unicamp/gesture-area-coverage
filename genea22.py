from genea import load_data_genea, compute_fgd, compute_gac_genea, load_fgd, load_genea_gac, report_gac_fgd
import torch
from tabulate import tabulate
import argparse
import plots


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', action='store_true', help='Include this argument to load the precomputed FGD and GAC metrics.')
    args = parser.parse_args()

    aligned_entries_g22 = ['FNA', 'FBT', 'FSA', 'FSB', 'FSC', 'FSD', 'FSF', 'FSG', 'FSH', 'FSI']
    hl_median_g22       = [70   , 27.5 , 71   , 30   , 53   , 34   , 38   , 38   , 36   , 46   ]
    app_match_g22       = [74   , 51.6 , 57.1 , 53.8 , 53   , 51.5 , 51.7 , 54.8 , 60.5 , 55.1 ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, genea_entries_loaders_g22, genea_entries_datasets_g22 = load_data_genea(genea_entries_path = './dataset/SubmittedGenea2022/full-body_bvh',
                                                                                   genea_entries      = aligned_entries_g22,
                                                                                   genea_trn_path     = None,)

    if args.load:
        aligned_fgds_trn, aligned_fgds_NA = load_fgd('./FGD/output', identifier='genea22_')
        aligned_setstats = load_genea_gac('./GAC/output/genea22_genea_setstats.npy', genea_entries_datasets_g22)

    else: # Set to True to recompute the FGD and GAC metrics
        _, aligned_fgds_NA = compute_fgd(genea_trn_loader      = None,
                                         genea_entries         = aligned_entries_g22,
                                         genea_entries_loaders = genea_entries_loaders_g22,
                                         device                = device,
                                         NA_entry              = 'FNA',
                                         identifier            = 'genea22_',)
        aligned_setstats = compute_gac_genea(genea_entries_datasets = genea_entries_datasets_g22,
                                                 NA_entry               = 'FNA',
                                                 identifier             = 'genea22_',)


    # Compute correlation between HL Median and FGD and Dice. Print table with results.
    table = report_gac_fgd(aligned_entries_g22[1:], aligned_setstats[1:], aligned_fgds_NA[1:], hl_median_g22[1:], plots.plot_HL_versus_dice_fgd_g22, './figures/genea22_fgd_vs_dice.png', decimal=3)
    header = ["Correlation", "FGD vs Hum. Median", "Dice vs Hum. Median"]
    print(tabulate(table, header, tablefmt="github"))

    # Compute correlation between HL Median and FGD and Dice. Print table with results.
    table = report_gac_fgd(aligned_entries_g22[1:], aligned_setstats[1:], aligned_fgds_NA[1:], app_match_g22[1:], plots.plot_APP_versus_dice_fgd_g22, './figures/genea22_fgd_vs_dice_appropriateness.png', decimal=3)
    header = ["Correlation", "FGD vs App Percent", "Dice vs App Percent"]
    print(tabulate(table, header, tablefmt="github"))
