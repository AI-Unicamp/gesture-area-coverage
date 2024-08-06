import os
import loader

if __name__ == '__main__':
    # Define the path to the dataset
    dataset_path = 'dataset'

    genea_trn_path = './dataset/Genea2023/trn/main-agent/bvh'
    loader.DatasetBVHLoader(name='trn', 
                            path=genea_trn_path, 
                            load=False,
                            pos_mean=None,
                            window=120)
    
    genea_val_path = './dataset/Genea2023/val/main-agent/bvh'
    loader.DatasetBVHLoader(name='val', 
                            path=genea_val_path, 
                            load=False,  #change to False to compute a save processed data
                            pos_mean=None,
                            window=120)

    entries = ['BD', 'BM', 'NA', 'SA', 'SB', 'SC', 'SD', 'SE', 'SF', 'SG', 'SH', 'SI', 'SJ','SK','SL']
    bvhsubmissionspath = './dataset/SubmittedGenea2023/BVH'
    for entry in entries:
        print(f'Loading {entry}...')
        path = os.path.join(bvhsubmissionspath, entry)
        loader.DatasetBVHLoader(name=entry, 
                                path=path, 
                                load=False,
                                pos_mean=None,
                                window=120)
        
    zeggs_path = './dataset/ZEGGS/'
    loader.DatasetBVHLoader(name='zeggs', 
                            path=zeggs_path, 
                            load=False,
                            pos_mean=None,
                            window=120,
                            fps=60,
                            njoints=75)