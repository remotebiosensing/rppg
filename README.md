# Remote Biosensing 
__________________________
<p align="center">
 <img src="logo.png">
</p>

Feel free to contact us with any questions and suggestions. We welcome your contributions and cooperation.

[![GitHub license](https://img.shields.io/github/license/remotebiosensing/rppg)](https://github.com/remotebiosensing/rppg/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](https://join.slack.com/t/remobebiosensing/shared_invite/zt-1u3kjfhf9-xWw_XQ8hGd7qFZymCSzUtg)
[![Tutorial](https://img.shields.io/badge/Tutorial-doc-blue)](https://github.com/remotebiosensing/rppg/wiki/rPPG-Documentation)

Remote Biosensing (`rPPG`) is an open-source framework for remote photoplethysmography (rPPG) and non-invasive blood pressure measurement (CNIBP) technology.
We aim to implement, evaluate, and benchmark DNN models for remote photoplethysmography (rPPG) and continuous non-invasive blood pressure (CNIBP). Our code is based on PyTorch.

### Quick Environment Setting with ANACONDA

`conda env create -f rppg.yaml`

`conda activate rppg`

### Quick Start with our examples
 
- #### rPPG( remote PPG) models
| year |  type   |     **model**     |       **implement**       |                                                                                           paper                                                                                            | 
|:----:|:-------:|:-----------------:|:-------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 2018 |   DL    |     DeepPhys      |             O             |                                                                         [paper](https://arxiv.org/abs/1805.07888)                                                                          |
| 2020 |   DL    |       MTTS        |              O             |                                                 [paper](https://papers.nips.cc/paper/2020/file/e1228be46de6a0234ac22ded31417bc7-Paper.pdf)                                                 |
 | 2020 |   DL    |     MetaPhys      |             O             |                                                                         [paper](https://arxiv.org/abs/2010.01773)                                                                          |
| 2021 |   DL    |   EfficentPhys    |             O              |                                                                         [paper](https://arxiv.org/abs/2110.04447)                                                                          |
| 2023 |   DL    |     BIGSMALL      |             O             |                                                                         [paper](https://arxiv.org/abs/2303.11573)                                                                          |
| 2019 |   DL    |   STVEN_rPPGNET   |                           |                                                                       [paper](https://arxiv.org/pdf/1907.11921.pdf)                                                                        |
| 2019 |   DL    |      PhysNet      |             O             |                                                                         [paper](https://arxiv.org/abs/1905.02419)                                                                          |
| 2019 |   DL    | 2D PhysNet + LSTM |                           |                                                                         [paper](https://arxiv.org/abs/1905.02419)                                                                          |
| 2020 |   DL    |   Siamese-rPPG    |                           |            [paper](https://dl.acm.org/doi/abs/10.1145/3341105.3373905?casa_token=db7gt2WxLMkAAAAA:G-H8UJbB5TumnogFbXqeMayChD2-wTy6qsWBxuRuylW_IOg3wOSwDdPwDona9xs03DxoOgYnuWnP)            |
| 2022 |   DL    |    PhysFormer     |             O             |                                                                       [paper](https://arxiv.org/pdf/2111.12082.pdf)                                                                        |
| 2023 |   DL    |   PhysFormer++    |                           |                                                           [paper](https://link.springer.com/article/10.1007/s11263-023-01758-1)                                                            |
| 2022 |   DL    |       APNET       |             O             |                                                                   [paper](https://europepmc.org/article/pmc/pmc9687348)                                                                    |
| TBD  |   DL    |      APNETv2      |                           |                                                                                           paper                                                                                            |
| 2019 |   DL    |     RhythmNet     |                           |                                                                         [paper](https://arxiv.org/abs/1910.11515)                                                                          |
| 2020 |    DL    |    HeartTrack     |                           | [paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w19/Perepelkina_HeartTrack_Convolutional_Neural_Network_for_Remote_Video-Based_Heart_Rate_Monitoring_CVPRW_2020_paper.pdf) |
| 2021 |   DL    |     TransrPPG     |                           |                                                         [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9460762)                                                          |
| 2022 |   DL    |     And-rPPG      |                           |    [paper](https://www.sciencedirect.com/science/article/pii/S0010482521009409?casa_token=RgFh-yohlBYAAAAA:Yzu6STC_nuKwMemvJkbPknsi5eYwoXPCk8EPLZFSlIEE82Ob5Z85NuRQu4OegoiUJWEJNAPJKA)     |
| 2022 |   DL    |      JAMSNet      |             O             |            [paper](https://ieeexplore.ieee.org/abstract/document/9973323/?casa_token=YE0aZV2EVRcAAAAA:s8ShA85zLSSZgZq9nmsa2imtZc8HbvOdhHfReYYg5_hEG6HPTYBcnjwj6yTRibCngr80hkI-)            |
| 2023 |   DL    |     CRGB rPPG     |                           |                                                                      [paper](https://www.mdpi.com/2306-5354/10/2/243)                                                                      |
| 2023 |   DL    | Skin + Deep Phys  |                           |  [paper](https://openaccess.thecvf.com/content/CVPR2023W/CVPM/papers/Deshpande_Camera-Based_Recovery_of_Cardiovascular_Signals_From_Unconstrained_Face_Videos_Using_CVPRW_2023_paper.pdf)  |
| 2023 | DL + TR |     rPPG-MAE      |                           |                                                                       [paper](https://arxiv.org/pdf/2306.02301.pdf)                                                                        |
| 2023 |   DL    |     LSTC-rPPG     |      need to verify       |       [paper](https://openaccess.thecvf.com/content/CVPR2023W/CVPM/papers/Lee_LSTC-rPPG_Long_Short-Term_Convolutional_Network_for_Remote_Photoplethysmography_CVPRW_2023_paper.pdf)        |
| 2008 |   TR    |       GREEN       |             O             |                                                               [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2717852)                                                                |
| 2010 |   TR    |        ICA        |                           |                                                                     [paper](https://pubmed.ncbi.nlm.nih.gov/20588929/)                                                                     |
| 2011 |   TR    |        PCA        | O #Need to change to cuda |                   [paper](https://www.researchgate.net/publication/220726433_Measuring_Pulse_Rate_with_a_Webcam_-_a_Non-contact_Method_for_Evaluating_Cardiac_Activity)                    |
| 2013 |   TR    |       CHROM       |             O             |                                                                   [paper](https://ieeexplore.ieee.org/document/6523142)                                                                    |
| 2014 |   TR    |        PBV        |             O             |                                                                     [paper](https://pubmed.ncbi.nlm.nih.gov/25159049/)                                                                     |
| 2016 |   TR    |        POS        |             O             |                                                                   [paper](https://ieeexplore.ieee.org/document/7565547)                                                                    |
| 2015 |   TR    |        SSR        |             O             |                                                                   [paper](https://ieeexplore.ieee.org/document/7355301)                                                                    |
| 2018 |   TR    |        LGI        |             O             |                               [paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w27/Pilz_Local_Group_Invariance_CVPR_2018_paper.pdf)                                |
| 2021 |   TR    |     EEMD-MCCA     |                           |                                                                   [paper](https://ieeexplore.ieee.org/document/9382335)                                                                    |
| 2023 |   TR    |  EEMD + FastICA   |                           |    [paper](https://iopscience.iop.org/article/10.1088/1361-6579/accefd/meta?casa_token=EVo9N2t0kekAAAAA:rUcw8D-6qGzT0dQZtBfgW0w2dVy-6p7kyHT3RV1q0YZMmEvQXpUoA-HaaO-K4m0aqiW-twzWWfmwXw)    |

- #### rPPG 
2023/CVPRW/Real-Time Estimation of Heart Rate in Situations Characterized by Dynamic Illumination using Remote Photoplethysmography/[paper](https://openaccess.thecvf.com/content/CVPR2023W/CVPM/papers/Hansen_Real-Time_Estimation_of_Heart_Rate_in_Situations_Characterized_by_Dynamic_CVPRW_2023_paper.pdf)

2023/IEEE Access/Heart Rate Estimation From Remote Photoplethysmography Based on Light-Weight U-Net and Attention Modules/[paper](https://ieeexplore.ieee.org/abstract/document/10141618)

- #### CNIBP (Continuous non-invasive blood pressure)
- [ ] PP-Net exmaple [paper](https://ieeexplore.ieee.org/document/9082808)

DATASET INFO
------------------- 


- #### rPPG datasets
| #   | Must Need | year | subject |  video  |     label      | **Dataset**  | **example** | **config** |                                                            **paper**                                                             |                                                                                                                                       **download or apply**                                                                                                                                        |
|-----|:---------:|:----:|:-------:|:-------:|:--------------:|:------------:|:-----------:|:----------:|:--------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     |           |      |         |         |                |     ALL      |   example   |   config   |                                                                                                                                  |                                                                                                                                                                                                                                                                                                    |
| 1   |     △     | 2011 |   27    |   RGB   |      ECG       |  MAHNOB_HCI  |   example   |   config   |                                       [link](https://ieeexplore.ieee.org/document/5975141)                                       |                                                                                                                                   [link](https://mahnob-db.eu/)                                                                                                                                    |
| 2   |           | 2014 |   25    |   RGB   |      PPG       |     AFRL     |   example   |   config   |                                       [link](https://ieeexplore.ieee.org/document/6974121)                                       |                                                                                                                                                link                                                                                                                                                |
| 3   |     O     | 2014 |   10    |   RGB   |    PPG/SPo2    |     PURE     |   example   |   config   |                                       [link](https://ieeexplore.ieee.org/document/6926392)                                       | [link](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure) |
| 4   |           | 2016 |   140   | RGB/NIR |   PPG/HR/BP    |    BP4D+     |   example   |   config   |     [link](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Multimodal_Spontaneous_Emotion_CVPR_2016_paper.pdf)      |                                                                                                           [link](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)                                                                                                            |
| 5   |     O     | 2016 |   40    |   RGB   |     HR/BP      |   MMSE-HR    |   example   |   config   |     [link](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Multimodal_Spontaneous_Emotion_CVPR_2016_paper.pdf)      |                                                                                   [link](https://binghamton.technologypublisher.com/tech/MMSE-HR_dataset_(Multimodal_Spontaneous_Expression-Heart_Rate_dataset))                                                                                   |
| 6   |     O     | 2017 |   40    |   RGB   |   PPG/HR/RR    |   COHFACE    |   example   |   config   |                                             [link](https://arxiv.org/abs/1709.00962)                                             |                                                                                                                          [link](https://www.idiap.ch/en/dataset/cohface)                                                                                                                           |
| 7   |           | 2017 |    -    |    -    |     PPG/BP     |    BIDMC     |   example   |   config   |                             [link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7748483)                             |                                                                                                                                                link                                                                                                                                                |
| 8   |     △     | 2018 |   25    |   RGB   |       -        |     LGGI     |   example   |   config   |   [link](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w27/Pilz_Local_Group_Invariance_CVPR_2018_paper.pdf)   |                                                                                                                       [link](https://github.com/partofthestars/LGI-PPGI-DB)                                                                                                                        |
| 9   |     O     | 2018 |   107   |    -    |     PPG/HR     |   VIPL-HR    |   example   |   config   |                                             [link](https://arxiv.org/abs/1810.04927)                                             |                                                                                                                                                link                                                                                                                                                |
| 10  |           | 2018 |   100   | RGB/NIR | PPG/HR/HRV/ECG |     OBF      |   example   |   config   |                                  [link](http://jultika.oulu.fi/files/nbnfi-fe2019080623583.pdf)                                  |                                                                                                                                                link                                                                                                                                                |
| 11  |           | 2018 |    8    | RGB/NIR |     PPG/HR     | MR-NIRP(ind) |   example   |   config   | [link](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w27/Nowara_SparsePPG_Towards_Driver_CVPR_2018_paper.pdf) |                                                                                                                   [link](https://computationalimaging.rice.edu/mr-nirp-dataset/)                                                                                                                   |
| 12  |     O     | 2019 |   42    |   RGB   |     PPG/HR     |  UBFC-rppg   |   example   |   config   |                         [link](https://www.sciencedirect.com/science/article/abs/pii/S0167865517303860)                          |                                                                                                                      [link](https://sites.google.com/view/ybenezeth/ubfcrppg)                                                                                                                      |
| 13  |           | 2020 |   10    |   RGB   |   PPG/HR/ECG   |   VicarPPG   |   example   |   config   |                                        [link](https://www.mdpi.com/2076-3417/10/23/8630)                                         |                                                                                            [link](https://docs.google.com/forms/d/e/1FAIpQLScwnW_D5M4JVovPzpxA0Bf1ZCTaG5vh7sYu48I0MVSpgltvdw/viewform)                                                                                             |
| 14  |           | 2020 |   18    | RGB/NIR |     PPG/HR     | MR-NIRP(DRV) |   example   |   config   |                                       [link](https://ieeexplore.ieee.org/document/9275394)                                       |                                                                                                                   [link](https://computationalimaging.rice.edu/mr-nirp-dataset/)                                                                                                                   |
| 15  |     △     | 2021 |   56    |   RGB   |   PPG/HR/EDA   |  UBFC-phys   |   example   |   config   |                                       [link](https://ieeexplore.ieee.org/document/9346017)                                       |                                                                                                                     [link](https://ieee-dataport.org/open-access/ubfc-phys-2)                                                                                                                      |
| 16  |           | 2021 |    9    |   RGB   |   PPG/HR/HRV   |  MPRSC-rPPG  |   example   |   config   |                                                                                                                                  |                                                                                                                [link](https://ieee-dataport.org/documents/mpsc-rppg-dataset#files)                                                                                                                 |
| 17  |     △     | 2021 |   140   | RGB/NIR |    HR/RR/BP    |     V4V      |   example   |   config   |                                          [link](https://arxiv.org/pdf/2109.10471v1.pdf)                                          |                                                                                                                         [link](https://vision4vitals.github.io/index.html)                                                                                                                         |
| 18  |           | 2022 |   62    |   RGB   |     PPG/RR     |     MTHS     |   example   |   config   |                                          [link](https://arxiv.org/pdf/2204.08989v2.pdf)                                          |                                                                                                                                                link                                                                                                                                                |
| 19  |     △     | 2023 |   33    |   RGB      |       PPG         |     MMPD     |   example   |   config   |                                             [link](https://arxiv.org/abs/2302.03840)                                             |                                                                                                                                                [link](https://github.com/McJackTang/MMPD_rPPG_dataset)                                                                                                                                                |
| 20  |           |      |   20    |   RGB   |     PPG/HR     |  EatingSet   |   example   |   config   |                                                                                                                                  |                                                                                                                                                link                                                                                                                                                |
| 21  |           |      |   24    |   RGB   |   HR/HRV/ECG   |  StableSet   |   example   |   config   |                                                                                                                                  |                                                                                                                                                link                                                                                                                                                |
| 22  |           |      |   37    |   RGB   |      PPG       |  BSIPL-RPPG  |   example   |   config   |                                                                                                                                  |                                                                                                                                                link                                                                                                                                                |
| 23  |           |      |   14    |    -    |     PPG/HR     |  BAMI-rPPG   |   example   |   config   |                                                                                                                                  |                                                                                                                                                link                                                                                                                                                |
| 24  |           |      |   900    |    -    |         |  Vital Videos   |   example   |   config   |                                                   [link](https://arxiv.org/abs/2306.11891)                                                                               |                                                                                                                                                link                                                                                                                                                |


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

Feel free to contact us with any questions and suggestions. We welcome your contributions and cooperation.

<a href="https://github.com/remotebiosensing/rppg/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=remotebiosensing/rppg" />
</a>

Please feel free to contact us and [join Slack](https://join.slack.com/t/remobebiosensing/shared_invite/zt-1u3kjfhf9-xWw_XQ8hGd7qFZymCSzUtg).

## Contacts

- Dae Yeol Kim, spicyyeol@gmail.com
- Kwangkee Lee, kwangkeelee@gmail.com

## Funding

This work was partly supported by the ICT R&D program of
MSIP/IITP. [2021(2021-0-00900), Adaptive Federated Learning in Dynamic Heterogeneous Environment]

## Reference
We will publish the paper introducing this repo soon, please cite the GitHub link by then.
https://github.com/remotebiosensing/rppg
