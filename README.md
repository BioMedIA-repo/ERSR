# ERSR

This is the official implementation of [ERSR: An Ellipse-constrained pseudo-label refinement and symmetric regularization framework for semi-supervised fetal head segmentation in ultrasound images] at JBHI-2025.

## Table of Contents
- [Requirements](#requirements)
- [Download](#download)
- [Train](#train)

## Requirements
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

## Download
You can download the supervised models we pre-trained in the corresponding datasets from [here](https://github.com/BioMedIA-repo/ERSR/tree/main/pretrained_ckpt).

## Train
### 1. Dataset Preparation
Please organise the dataset according to the following structure,where the npz file stores the images and their corresponding segmentation labels with the key name {image,label}:
```angular2
root:[data]
+--data_name
| +--train.txt
| +--val.txt
| +--test.txt
| +--00001.npz
```

### 2. Unsupervised training
Now you can start to fine-tune the supervised model:
```angular2
python train.py --exp se/train --data_name fetal_head_se --labeled_ratio 0.1
```

## Acknowledgement
The code is based on [SSL4MIS]([https://github.com/HiLab-git/SSL4MIS]).
We thank the authors for their open-sourced code and encourage users to cite their works when applicable.

## Citations
