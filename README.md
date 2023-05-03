# Remote Biosensing
<p align="center">
 <img src="logo.png">
</p>

[![GitHub license](https://img.shields.io/github/license/remotebiosensing/rppg)](https://github.com/remotebiosensing/rppg/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](https://join.slack.com/t/remobebiosensing/shared_invite/zt-1u3kjfhf9-xWw_XQ8hGd7qFZymCSzUtg)

Remote Biosensing (`rPPG`) is a framework for non-contact algorithms for remote photoplethysmography (rPPG) and for
non-invasive blood pressure measurement algorithms (CNIBP) technology.
We aim to implement a deep learning-based remote photoplethysmography (rPPG) model and continuous non-invasive blood
pressure (CNIBP) using PyTorch.

### Quick Start with our examples
 
- #### rPPG( remote PPG) models
| **Implementation** | year | type |     **model**     |                                            **example**                                            |                                              **config**                                               |                                           paper                                            | 
|:------------------:|:----:|:----:|:-----------------:|:-------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|
|      &#10004;      |      |  DL  |     DeepPhys      | [example](https://github.com/remotebiosensing/rppg/blob/main/rppg/examples/physnet_ubfc_ubfc.py)  | [config](https://github.com/remotebiosensing/rppg/blob/main/rppg/configs/FIT_DEEPPHYS_UBFC_UBFC.yaml) |                         [paper](https://arxiv.org/abs/1805.07888)                          |
|      &#10005;      |      |  DL  |       MTTS        |                                              example                                              |                                                config                                                 | [paper](https://papers.nips.cc/paper/2020/file/e1228be46de6a0234ac22ded31417bc7-Paper.pdf) |
 |      &#10004;      |      |  DL  |     MetaPhys      | [example](https://github.com/remotebiosensing/rppg/blob/main/rppg/examples/metaphys_ubfc_ubfc.py) | [config](https://github.com/remotebiosensing/rppg/blob/main/rppg/configs/FIT_METAPHYS_UBFC_UBFC.yaml) |                         [paper](https://arxiv.org/abs/2010.01773)                          |
|      &#10004;      |      |  DL  |      PhysNet      | [example](https://github.com/remotebiosensing/rppg/blob/main/rppg/examples/physnet_ubfc_ubfc.py)  | [config](https://github.com/remotebiosensing/rppg/blob/main/rppg/configs/FIT_PHYSNET_UBFC_UBFC.yaml)  |                         [paper](https://arxiv.org/abs/1905.02419)                          |
|      &#10005;      |      |  DL  | 2D PhysNet + LSTM |                                              example                                              |                                                config                                                 |                         [paper](https://arxiv.org/abs/1905.02419)                          |
 |      &#10004;      |      |  DL  |      APNETv2      | [example](https://github.com/remotebiosensing/rppg/blob/main/rppg/examples/apnetv2_ubfc_ubfc.py)  |                                                config                                                 |                                           paper                                            |

- #### rPPG datasets
| **Implementation** | year | subject |  video  |     label      | **Dataset**  | **DIFF** | **CONT** | **example** | **config** | **paper** | **link** |
|:------------------:|:----:|:-------:|:-------:|:--------------:|:------------:|:--------:|:--------:|:-----------:|:----------:|:---------:|:--------:|
|      &#10005;      |      |         |         |                |     ALL      | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10004;      |      |   42    |   RGB   |     PPG/HR     |  UBFC-rppg   | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10005;      |      |   56    |   RGB   |   PPG/HR/EDA   |  UBFC-phys   | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10005;      |      |   140   | RGB/NIR |   PPG/HR/BP    |    BP4D+     | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10005;      |      |   20    |   RGB   |     PPG/HR     |  EatingSet   | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10005;      |      |   24    |   RGB   |   HR/HRV/ECG   |  StableSet   | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10005;      |      |   10    |   RGB   |   PPG/HR/ECG   |   VicarPPG   | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10005;      |      |   40    |   RGB   |     HR/BP      |   MMSE-HR    | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10004;      |      |   40    |   RGB   |   PPG/HR/RR    |   COHFACE    | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10004;      |      |   25    |   RGB   |       -        |     LGGI     | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10005;      |      |   37    |   RGB   |      PPG       |  BSIPL-RPPG  | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10004;      |      |   27    |   RGB   |      ECG       |  MAHNOB_HCI  | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10004;      |      |   107   |    -    |     PPG/HR     |   VIPL-HR    | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10004;      |      |   10    |   RGB   |    PPG/SPo2    |     PURE     | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10004;      |      |   140   | RGB/NIR |    HR/RR/BP    |     V4V      | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10004;      |      |   14    |    -    |     PPG/HR     |  BAMI-rPPG   | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10004;      |      |   100   | RGB/NIR | PPG/HR/HRV/ECG |     OBF      | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10004;      |      |   18    | RGB/NIR |     PPG/HR     | MR-NIRP(DRV) | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10004;      |      |    8    | RGB/NIR |     PPG/HR     | MR-NIRP(ind) | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10004;      |      |   25    |   RGB   |      PPG       |     AFRL     | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10004;      |      |   62    |   RGB   |     PPG/RR     |     MTHS     | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10004;      |      |    -    |    -    |     PPG/BP     |    BIDMC     | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10004;      |      |    9    |   RGB   |   PPG/HR/HRV   |  MPRSC-rPPG  | &#10004; | &#10004; |   example   |   config   |           |   link   |
|      &#10005;      |      |         |         |                |     MMPD     | &#10004; | &#10004; |   example   |   config   |           |   link   |


- #### CNIBP (Continuous non-invasive blood pressure)
- [ ] PP-Net exmaple [paper](https://ieeexplore.ieee.org/document/9082808)

### Documentation(TBD)

### Performance Comparison

- rPPG

| MODEL | Train/val Dataset | Test Dataset | lr  | MAE | RMSE | MAPE | r   |
|-------|-------------------|--------------|-----|-----|------|------|-----|

- CNIBP

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

