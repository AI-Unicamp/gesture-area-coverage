# Gesture Area Coverage

Implementaion of the paper Gesture Area Coverage to Assess Gesture Expressiveness and Human-Likeness

## Overview

This repository provides:
* Dockerfile to replicate the results
* Code for processing the original data. The processed data is available here
* Code for computing the Gesture Area Coverage (GAC). GAC results used to create the figures and tables in the paper are provided in /GAC/output/ as npy and csv files
* Modified version used for computing the Fréchet Gesture Distance (FGD). The /FGD/output/ folder provides the FGD checkpoint used in the paper and the training log
* Code for plotting the figures used in the paper. The figures are also provided in /figures/

## 1. Preparing environment

The environment is available using Docker. 

To create the environment in Docker:

1. Create docker image using:

```sh
docker build -t gac .
```

2. Run container. Example:

```sh
docker run --rm -it --gpus device=0 --userns=host --shm-size 64G -v /my_dir/gesture-area-coverage:/workspace/gac/ -p '8880:8880' --name gac_container gac:latest /bin/bash
```

3. Enter container and go to the mapped folder ` cd /workspace/gac `.

## Data pre-processing

1. Get the GENEA Challenge 2023 dataset [here](https://zenodo.org/records/8199133), the Submitted Entries to the challenge [here](https://zenodo.org/records/8146028), follow the procedures detailed in the ZEGGS official repository [here](https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS/tree/main?tab=readme-ov-file#zeggs-dataset) and put everything into ` /dataset/ ` as:

📂 dataset\
 ┣ 📂 Genea2023\
 ┃ ┣ 📂 trn\
 ┃ ┃ ┣ 📂 ...\
 ┃ ┣ 📂 val\
 ┃ ┃ ┣ 📂 ...\
 ┃ ┣ 📂 tst\
 ┃ ┃ ┣ 📂 ...\
 ┣ 📂 SubmittedGenea2023\
 ┃ ┣ 📂 BVH\
 ┃ ┃ ┣ 📂 BD\
 ┃ ┃ ┣ 📂 BM\
 ┃ ┃ ┣ 📂 SA\
 ┃ ┃ ┣ 📂 ...\
 ┣ 📂 ZEGGS\
 ┃ ┣ 📄 001_Neutral_0_x_1_0.bvh\
 ┃ ┣ 📄 001_Neutral_1_x_1_0.bvh\
 ┃ ┣ 📄 ...\

2. Run:

```sh
python -m process_dataset
```

The script will create a ` /processed/ ` folder inside each folder that contains BVH files. The processed folder will contain npy files with the 3D positions of each joint of all BVH files in the respective folder.

## Train FGD

To train the FGD from scratch:

1. Run

```sh
python -m FGD.train_AE
```

The training log and the model checkpoint with lowest loss value in the validation set of the GENEA Challenge 2023 will be saved in the folder ` /FGD/output/ `. 

## Compute FGD and GAC

To compute all the results presented in the paper:

1. Run:

```sh
python -m main
```

The scipt will create the figures in the ` /figures/ ` folder and save the results of the metrics as csv and npy in the ` /GAC/output/ ` folder.

## Cite

Please consider citing our paper:

```text
@inproceedings{tonoli2023gesture,
  title={Gesture Area Coverage to Assess Gesture Expressiveness and Human-Likeness},
  author={Tonoli, Rodolfo L and Costa, Paula Dornhofer Paro and Marques, Leonardo B de MM and Ueda, Lucas H},
  booktitle={Companion Publication of the 26th International Conference on Multimodal Interaction},
  pages={N/A},
  year={2024}
}
```