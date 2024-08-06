import os
import numpy as np
import bvhsdk
from tqdm import tqdm

class DatasetBVHLoader():
    """
    Class that stores a motion dataset. It can be used in conjunction to Torch Dataset and Torch DataLoader.

    This class can process BVH files and convert them to numpy arrays (rotations and positions), including the datasets' standard deviation and mean.
    After processing, it loads those numpy arrays. The data can be acessed through the __getitem__ method or the arrays pos and rot3d.

    Parameters:
    name: str
        Name of the dataset.
    path: str
        Path to the BVH files. All files in this path will be processed unles a list of files is passed as the 'files' parameter. In this case, only the files in the list will be processed.
    data_rep: str
        Data representation. Can be 'pos' or 'rot3d'. Default is 'pos'. (*)
    step: int
        Step size for the sliding window. Default is 60. (*)
    window: int
        Window size for the sliding window. Default is 120. (*)
    fps: int
        Frames per second. Default is 30. (*)
    skipjoint: int
        Number of joints to skip. Default is 1. Might be useful to skip joints such as the body_world as in the TWH dataset. (*)
    njoints: int
        Number of joints in the skeleton hierarchy. Default is 83, following the TWH dataset. (*)
    pos_mean: str
        Path to the numpy array with the mean of the positions. If not passed, all stats will be computed and saved. (+)
    pos_std: str
        Path to the numpy array with the standard deviation of the positions. If not passed, all stats will be computed and saved. (+)
    rot3d_mean: str
        Path to the numpy array with the mean of the 3D rotations. If not passed, all stats will be computed and saved. (+)
    rot3d_std: str
        Path to the numpy array with the standard deviation of the 3D rotations. If not passed, all stats will be computed and saved. (+)
    files: list
        List of files to process. If not passed, all files in the path will be processed.
    load: bool
        If True, loads the already processed data. If False or not passed, processes the data. Default is False.
    
    (*) Parameters are not required for processing, although window might impact the mean and std computation. It is possible to load the data with different parameters than the ones used for processing.
    (+) Mean and standard deviation paths can be loaded from different dataset, e.g. it might be desirable to use the mean and std from the training set in the validation or test set.
    """
    def __init__(self,
                 name,
                 path,
                 data_rep = 'pos',
                 step=60,
                 window=120,
                 fps=30,
                 skipjoint = 1,
                 njoints = 83,
                 pos_mean = 'dataset/Genea2023/trn/main-agent/bvh/processed/trn_bvh_positions_mean.npy',
                 pos_std = 'dataset/Genea2023/trn/main-agent/bvh/processed/trn_bvh_positions_std.npy',
                 rot3d_mean = 'dataset/Genea2023/trn/main-agent/bvh/processed/trn_bvh_3drotations_mean.npy',
                 rot3d_std = 'dataset/Genea2023/trn/main-agent/bvh/processed/trn_bvh_3drotations_std.npy',
                 **kwargs) -> None:
        
        self.step = step
        self.window = window
        self.fps = fps
        self.name = name
        self.path = path
        self.skipjoint = skipjoint
        self.data_rep = data_rep
        self.njoints = njoints
        self.processed_path = os.path.join(self.path, "processed")
        self.computeMeanStd = True
        # If mean and std paths are passed, load them
        if pos_mean and pos_std and rot3d_mean and rot3d_std:
            self.computeMeanStd = False
            self.pos_mean = np.load(pos_mean)
            self.pos_std = np.load(pos_std)
            self.rot3d_mean = np.load(rot3d_mean)
            self.rot3d_std = np.load(rot3d_std)
        
        # Create processed path if it does not exist
        if not os.path.isdir(self.processed_path):
            os.mkdir(self.processed_path)

        # Compose list of files to process
        self.files = kwargs.pop('files', [file for file in os.listdir(path) if file.endswith('.bvh')])
        self.files.sort()

        # Get parents vector (skeleton hierarchy) using the first bvh file
        # If you prefer it hard-coded, this is the array for the TWH (Genea) dataset:
        # [-1,  0,  1,  2,  3,  4,  5,  6,  7,  7,  7,  7, 11, 11, 11, 14, 15, 16, 17, 18, 18, 17, 17, 16, 16, 15, 15,  5, 27, 28, 29, 30, 31, 32, 33, 34, 35, 33, 37, 38, 33, 40, 41, 33, 43, 44, 33, 46, 47, 48,  5, 50, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 57, 62, 63, 57, 65, 66, 57, 68, 69, 57, 71, 72,  4,  1, 75, 76, 77,  1, 79, 80, 81]
        # For the ZEGGS dataset:
        # [-1,  0,  1,  2,  3,  4,  5,  6,  7,  4,  9, 10, 11, 12, 13, 14, 15, 12, 17, 18, 19, 12, 21, 22, 23, 12, 25, 26, 27, 12, 29, 30, 31, 12, 11,  4, 35, 36, 37, 38, 39, 40, 41, 38, 43, 44, 45, 38, 47, 48, 49, 38, 51, 52, 53, 38, 55, 56, 57, 38, 37,  0, 61, 62, 63, 64, 63, 62, 0, 68, 69, 70, 71, 70, 69]
        aux = bvhsdk.ReadFile(os.path.join(self.path,self.files[0]))
        self.parents = aux.arrayParent()

        # If load = True, loads already processed data
        if kwargs.pop('load', False):
            self.pos = np.load(os.path.join(self.processed_path, self.name + "_bvh_positions.npy"), allow_pickle = True)
            self.rot3d = np.load(os.path.join(self.processed_path, self.name + "_bvh_3drotations.npy"), allow_pickle = True)
        else:
            # Process data
            self.__data2samples()
            # Save processed data
            # This does not actually save a numpy array due to different lengths of each take
            np.save(file = os.path.join(self.processed_path, self.name + "_bvh_positions.npy"),
                    arr = self.pos,
                    allow_pickle = True)
            np.save(file = os.path.join(self.processed_path, self.name + "_bvh_3drotations.npy"),
                    arr = self.rot3d,
                    allow_pickle = True)
            
            # Compute mean and std if needed
            if self.computeMeanStd:
                self.pos_mean, self.pos_std = self.__computeMeanStd(self.pos)
                self.rot3d_mean, self.rot3d_std = self.__computeMeanStd(self.rot3d)
                np.save(file = os.path.join(self.processed_path, self.name + "_bvh_positions_mean.npy"),
                        arr = self.pos_mean,
                        allow_pickle = True)
                np.save(file = os.path.join(self.processed_path, self.name + "_bvh_positions_std.npy"),
                        arr = self.pos_std,
                        allow_pickle = True)
                np.save(file = os.path.join(self.processed_path, self.name + "_bvh_3drotations_mean.npy"),
                        arr = self.rot3d_mean,
                        allow_pickle = True)
                np.save(file = os.path.join(self.processed_path, self.name + "_bvh_3drotations_std.npy"),
                        arr = self.rot3d_std,
                        allow_pickle = True)
            
        self.rot3d_std[self.rot3d_std==0] = 1
        self.pos_std[self.pos_std==0] = 1
        # Get frames for each take in the dataset
        self.frames = [len(take) for take in self.pos]
        # Get number of samples for each take in the dataset (sliding window)
        self.samples_per_take = [len( [i for i in np.arange(0, n, self.step) if i + self.window <= n] ) for n in self.frames]
        # Get cumulative sum of samples for each take in the dataset
        self.samples_cumulative = [np.sum(self.samples_per_take[:i+1]) for i in range(len(self.samples_per_take))]
        # Get total number of samples in the dataset
        self.length = self.samples_cumulative[-1]
        
    def __getitem__(self, index):
        file_idx = np.searchsorted(self.samples_cumulative, index+1, side='left')
        sample = index - self.samples_cumulative[file_idx-1] if file_idx > 0 else index
        b, e = sample*self.step, sample*self.step+self.window
        if self.data_rep == 'pos':
            sample = self.norma(self.pos[file_idx][b:e, self.skipjoint:, :], self.pos_mean[self.skipjoint:], self.pos_std[self.skipjoint:]).reshape(-1, (self.njoints-self.skipjoint)*3)
        elif self.data_rep == 'rot3d':
            sample = self.norma(self.rot3d[file_idx][b:e, self.skipjoint:, :], self.rot3d_mean[self.skipjoint:], self.rot3d_std[self.skipjoint:]).reshape(-1, (self.njoints-self.skipjoint)*3)
        return sample, self.files[file_idx] + f"_{b}_{e}"
    
    def norma(self, arr_, mean, std):
        return (arr_-mean) / std
    
    def inv_norma(self, arr_, mean, std):
        return (arr_*std) + mean
    
    def __len__(self):
        return self.length
    
    def posLckHips(self):
        """
        Locks the root position to the origin. Does not change the original pos array.
        """
        return [pos-np.tile(pos[:,0,:][:, np.newaxis], (1,self.njoints,1)) for pos in self.pos]

    def __data2samples(self):
        # Converts all files (takes) to samples
        self.pos, self.rot3d = [], []
        print('Preparing samples...')
        for i, file in enumerate(tqdm(self.files)):
            anim = bvhsdk.ReadFile(os.path.join(self.path,file))
            p, r = self.__loadtake(anim)
            self.pos.append(p)
            self.rot3d.append(r)
        print('Done. Converting to numpy.')
        
    def __loadtake(self, anim):
        # Converts a single file (take) to samples
        # Compute joint position
        joint_positions, joint_rotations = [], []
        for frame in range(anim.frames):
            joint_positions.append([joint.getPosition(frame) for joint in anim.getlistofjoints()])
            joint_rotations.append([joint.rotation[frame] for joint in anim.getlistofjoints()])
        
        return np.asarray(joint_positions), np.asarray(joint_rotations)
        
    def __computeMeanStd(self, arr):
        window = self.window
        mean, m2, counter = 0.0, 0.0, 0
        for i, take in enumerate(arr):
            duration = take.shape[0]
            for frame in range(0, duration-duration%window, window):
                mean += np.sum(take[frame:frame+window]     , axis = 0)
                m2   += np.sum(take[frame:frame+window] ** 2, axis = 0)
            counter += np.floor(duration/window)

        mean = mean/(counter*window)
        m2   = m2  /(counter*window)
        std = np.sqrt(m2 - mean ** 2)
        return mean, std