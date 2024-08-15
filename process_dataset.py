import os
import loader

if __name__ == '__main__':
    # Define the path to the dataset
    dataset_path = 'dataset'

    # Genea Challenge 2023 train set
    genea_trn_path = './dataset/Genea2023/trn/main-agent/bvh'
    loader.DatasetBVHLoader(name='trn', 
                            path=genea_trn_path, 
                            load=False,
                            pos_mean=None,
                            window=120)
    
    # Genea Challenge 2023 validation set
    genea_val_path = './dataset/Genea2023/val/main-agent/bvh'
    loader.DatasetBVHLoader(name='val', 
                            path=genea_val_path, 
                            load=False,  #change to False to compute a save processed data
                            pos_mean=None,
                            window=120)

    # Submitted Entries to the Genea Challenge 2023
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
        
    # ZEGGS dataset
    zeggs_path = './dataset/ZEGGS/'
    loader.DatasetBVHLoader(name='zeggs', 
                            path=zeggs_path, 
                            load=False,
                            pos_mean=None,
                            window=120,
                            fps=60,
                            njoints=75)
    
    # Submitted Entries to the Genea Challenge 2022 
    entries = ['FBT', 'FNA', 'FSA', 'FSB', 'FSC', 'FSD', 'FSF', 'FSG', 'FSH', 'FSI']
    bvhsubmissionspath = './dataset/SubmittedGenea2022/full-body_bvh'
    # Put identation in FSG so that bvhsdk is able to read it
    files = [f for f in os.listdir(os.path.join(bvhsubmissionspath, 'FSG')) if f.endswith('.bvh')]
    files.sort()
    # Just to check if it has already been indented
    with open(os.path.join(bvhsubmissionspath, 'FSG', files[0]), 'r') as f:
        lines = f.readlines()
    if lines[3].count('  ') == 0: 
        reference = os.path.join(bvhsubmissionspath, 'FNA', files[0])
        with open(reference, 'r') as f:
            lines = f.readlines()
            tabs = []
            for line in lines:
                if line.startswith('MOTION'):
                    break
                else:
                    tabs.append(line.count('  '))
        for file in files:
            with open(os.path.join(bvhsubmissionspath, 'FSG', file), 'r') as f:
                lines = f.readlines()
                for i in range(len(tabs)):
                    lines[i] = f'{"  "*(tabs[i])}' + lines[i]
                with open(os.path.join(bvhsubmissionspath, 'FSG', file), 'w') as f:
                    f.writelines(lines)

    for entry in entries:
        print(f'Loading {entry}...')
        path = os.path.join(bvhsubmissionspath, entry)
        loader.DatasetBVHLoader(name=entry, 
                                path=path, 
                                load=False,
                                pos_mean=None,
                                window=120)