# Remote Biosensing
__________________________
<p align="center">
 <img src="logo.png">
</p>

Our community is eagerly waiting for researchers and developers interested in non-contact/non-invasive algorithm
research and development
to [join](https://join.slack.com/t/remobebiosensing/shared_invite/zt-1u3kjfhf9-xWw_XQ8hGd7qFZymCSzUtg) us.

[![GitHub license](https://img.shields.io/github/license/remotebiosensing/rppg)](https://github.com/remotebiosensing/rppg/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](https://join.slack.com/t/remobebiosensing/shared_invite/zt-1u3kjfhf9-xWw_XQ8hGd7qFZymCSzUtg)
[![Tutorial](https://img.shields.io/badge/Tutorial-doc-blue)](https://github.com/remotebiosensing/rppg/wiki/rPPG-Documentation)

Remote Biosensing (`rPPG`) is a framework for non-contact algorithms for remote photoplethysmography (rPPG) and for
non-invasive blood pressure measurement algorithms (CNIBP) technology.
We aim to implement a deep learning-based remote photoplethysmography (rPPG) model and continuous non-invasive blood
pressure (CNIBP) using PyTorch.

### Quick Environment Setting with ANACONDA

`conda env create -f rppg.yaml`

`conda activate rppg`

### Quick Start with our examples
 
- #### rPPG( remote PPG) models
| year | type |     **model**     | **implement**  |                                                                                       paper                                                                                       | 
|:----:|:----:|:-----------------:|:--------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 2018 |  DL  |     DeepPhys      |       O        |                                                                     [paper](https://arxiv.org/abs/1805.07888)                                                                     |
| 2020 |  DL  |       MTTS        |                |                                            [paper](https://papers.nips.cc/paper/2020/file/e1228be46de6a0234ac22ded31417bc7-Paper.pdf)                                             |
 | 2020 |  DL  |     MetaPhys      |       O        |                                                                     [paper](https://arxiv.org/abs/2010.01773)                                                                     |
| 2021 |  DL  |   EfficentPhys    |                |                                                                     [paper](https://arxiv.org/abs/2110.04447)                                                                     |
| 2023 |  DL  |     BIGSMALL      |       O        |                                                                     [paper](https://arxiv.org/abs/2303.11573)                                                                     |
| 2019 |  DL  |   STVEN_rPPGNET   |                |                                                                   [paper](https://arxiv.org/pdf/1907.11921.pdf)                                                                   |
| 2019 |  DL  |      PhysNet      |       O        |                                                                     [paper](https://arxiv.org/abs/1905.02419)                                                                     |
| 2019 |  DL  | 2D PhysNet + LSTM |                |                                                                     [paper](https://arxiv.org/abs/1905.02419)                                                                     |
| 2022 |  DL  |    PhysFormer     |       O        |                                                                   [paper](https://arxiv.org/pdf/2111.12082.pdf)                                                                   |
| 2023 |  DL  |   PhysFormer++    |                |                                                       [paper](https://link.springer.com/article/10.1007/s11263-023-01758-1)                                                       |
| 2022 |  DL  |       APNET       |       O        |                                                               [paper](https://europepmc.org/article/pmc/pmc9687348)                                                               |
| TBD  |  DL  |      APNETv2      |                |                                                                                       paper                                                                                       |
| 2019 |  DL  |     RhythmNet     |                |                                                                     [paper](https://arxiv.org/abs/1910.11515)                                                                     |
| 2022 |  DL  |      JAMSNet      |                |       [paper](https://ieeexplore.ieee.org/abstract/document/9973323/?casa_token=YE0aZV2EVRcAAAAA:s8ShA85zLSSZgZq9nmsa2imtZc8HbvOdhHfReYYg5_hEG6HPTYBcnjwj6yTRibCngr80hkI-)        |
| 2023 |  DL  |     CRGB rPPG     |                |                                                                 [paper](https://www.mdpi.com/2306-5354/10/2/243)                                                                  |
| 2023 | DL + TR | rPPG-MAE |  |[paper](https://arxiv.org/pdf/2306.02301.pdf) |
| 2023 | DL | LSTC-rPPG | need to verify |[paper](https://openaccess.thecvf.com/content/CVPR2023W/CVPM/papers/Lee_LSTC-rPPG_Long_Short-Term_Convolutional_Network_for_Remote_Photoplethysmography_CVPRW_2023_paper.pdf) |
| 2008 |  TR  |       GREEN       |                |                                                           [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2717852)                                                           |
| 2010 |  TR  |        ICA        |                |                                                                [paper](https://pubmed.ncbi.nlm.nih.gov/20588929/)                                                                 |
| 2011 |  TR  |        PCA        |                |               [paper](https://www.researchgate.net/publication/220726433_Measuring_Pulse_Rate_with_a_Webcam_-_a_Non-contact_Method_for_Evaluating_Cardiac_Activity)               |
| 2013 |  TR  |       CHROM       |                |                                                               [paper](https://ieeexplore.ieee.org/document/6523142)                                                               |
| 2014 |  TR  |        PBV        |                |                                                                [paper](https://pubmed.ncbi.nlm.nih.gov/25159049/)                                                                 |
| 2016 |  TR  |        POS        |                |                                                               [paper](https://ieeexplore.ieee.org/document/7565547)                                                               |
| 2015 |  TR  |        SSR        |                |                                                               [paper](https://ieeexplore.ieee.org/document/7355301)                                                               |
| 2018 |  TR  |        LGI        |                |                           [paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w27/Pilz_Local_Group_Invariance_CVPR_2018_paper.pdf)                           |
| 2023 |  TR  |  EEMD + FastICA   |                | [paper](https://iopscience.iop.org/article/10.1088/1361-6579/accefd/meta?casa_token=EVo9N2t0kekAAAAA:rUcw8D-6qGzT0dQZtBfgW0w2dVy-6p7kyHT3RV1q0YZMmEvQXpUoA-HaaO-K4m0aqiW-twzWWfmwXw) |

- #### rPPG 
2023/CVPRW/Real-Time Estimation of Heart Rate in Situations Characterized by Dynamic Illumination using Remote Photoplethysmography/[paper](https://openaccess.thecvf.com/content/CVPR2023W/CVPM/papers/Hansen_Real-Time_Estimation_of_Heart_Rate_in_Situations_Characterized_by_Dynamic_CVPRW_2023_paper.pdf)

2023/IEEE Access/Heart Rate Estimation From Remote Photoplethysmography Based on Light-Weight U-Net and Attention Modules/[paper](https://ieeexplore.ieee.org/abstract/document/10141618)

- #### CNIBP (Continuous non-invasive blood pressure)
- [ ] PP-Net exmaple [paper](https://ieeexplore.ieee.org/document/9082808)

### datasets
You can find information about datasets at the following [link](https://github.com/remotebiosensing/rppg/tree/main/rppg/datasets).

### Documentation(TBD)

### Performance Comparison
#### - rPPG

- All evaluations are based on the model with the lowest loss value during validation.

|   MODEL   | Train/val Dataset | Test Dataset |   lr   |  optim  |  lr-sch  | HR - MAE | HR - RMSE | HR - MAPE | HR -corr |
|:---------:|:-----------------:|:------------:|:------:|:-------:|:--------:|:--------:|:---------:|:---------:|:--------:|
 |  DeepPhys |       UBFC        |     UBFC     |  1e-3  |  AdamW  |   oneCycle   |   3.71  |   13.82    |   4.03    |   0.81   |
 |  DeepPhys |       PURE        |     PURE     |  1e-3  |  AdamW  |   oneCycle   |   1.78   |   7.72    |   1.86    |   0.91   |
 |  PhysNet  |       UBFC        |     PURE     |  1e-3  |  Adam   |   None   |   1.74   |   8.40    |   1.75    |   0.92   |
 |  PhysNet  |       PURE        |     UBFC     |  1e-3  |  Adam   |   None   |   1.90   |   7.02    |   2.11    |   0.87   |
 

- CNIBP

### Bench Mark Git


## Community

Our community is eagerly waiting for researchers and developers interested in non-contact/non-invasive algorithm
research and development
to [join](https://join.slack.com/t/remobebiosensing/shared_invite/zt-1u3kjfhf9-xWw_XQ8hGd7qFZymCSzUtg) us.

<a href="https://github.com/remotebiosensing/rppg/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=remotebiosensing/rppg" />
</a>

## Contacts

- Dae Yeol Kim, spicyyeol@gmail.com
- Kwangkee Lee, kwangkeelee@gmail.com

## Funding

This work was partly supported by the ICT R&D program of
MSIP/IITP. [2021(2021-0-00900), Adaptive Federated Learning in Dynamic Heterogeneous Environment]

## Reference
If you use this code before our paper is published, please cite the GitHub link.
https://github.com/remotebiosensing/rppg
