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
Please organise the dataset according to the following structure,where the npz file stores the images and their corresponding segmentation labels with the key name {image, label}:
```angular2
root:[data]
+--data_name
| +--train.txt
| +--val.txt
| +--test.txt
| +--00001.npz
| +--...
```

### 2. Unsupervised training
Now you can start to fine-tune the supervised model:
```angular2
python train.py --exp <experiments name> --data_name <dataset name> --labeled_ratio <0.1 or 0.2> --pth <pretrained_ckpt_pth>
```
For PSFH dataset with 10% labeled:
```angular2
python train.py --exp se/train01 --data_name fetal_head_se --labeled_ratio 0.1 --pth se_01
```
For PSFH dataset with 20% labeled:
```angular2
python train.py --exp se/train02 --data_name fetal_head_se --labeled_ratio 0.2 --pth se_02
```
For HC18 dataset with 10% labeled:
```angular2
python train.py --exp cu/train01 --data_name fetal_head_cu --labeled_ratio 0.1 --pth cu_01
```
For HC18 dataset with 20% labeled:
```angular2
python train.py --exp cu/train02 --data_name fetal_head_cu --labeled_ratio 0.2 --pth cu_02
```
## Acknowledgement
The code is based on [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).
We thank the authors for their open-sourced code and encourage users to cite their works when applicable.

## Citations

If the code is helpful for your research, please consider citing:
```angular2
@ARTICLE{11141370,
  author={Zhou, Linkuan and Chen, Zhexin and Shen, Yufei and Xu, Junlin and Xuan, Ping and Zhu, Yixin and Fang, Yuqi and Cong, Cong and Wei, Leyi and Su, Ran and Zhou, Jia and Jin, Qiangguo},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={ERSR: An Ellipse-constrained pseudo-label refinement and symmetric regularization framework for semi-supervised fetal head segmentation in ultrasound images}, 
  year={2025},
  pages={1-11}
}
```
## Social media

<p align="center"><img width="600" alt="image" src="https://github.com/BioMedIA-repo/.github/blob/052046a248d3831a599e11c85ff94cdd658c5abc/pic/wechat.png" height=""></p> 
Welcome to follow our [Wechat official account: iBioMedInfo] and [Xiaohongshu official account: iBioMedInfo], we will share recent studies on biomedical image and bioinformation analysis there.

## Global Collaboration & Questions

**Global Collaboration:** We're on a mission to biomedical research, aiming for artificial intelligence and its
applications to biomedical image and bioinformation analysis, promoting the development of the medical community.
Collaborate with us to increase competitiveness.

**Questions:** General questions, please contact 'zlinkw@mail.nwpu.edu.cn'
