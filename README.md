# Remote Biosensing
[![GitHub license](https://img.shields.io/github/license/remotebiosensing/rppg)](https://github.com/remotebiosensing/rppg/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](https://join.slack.com/t/remobebiosensing/shared_invite/zt-1u3kjfhf9-xWw_XQ8hGd7qFZymCSzUtg)

Remote Biosensing (`rPPG`) is a framework for non-contact algorithms for remote photoplethysmography (rPPG) and for non-invasive blood pressure measurement algorithms (CNIBP) technology.
We aim to implement a deep learning-based remote photoplethysmography (rPPG) model and continuous non-invasive blood pressure (CNIBP) using PyTorch.

### Quick Start with our examples
- #### rPPG( remote PPG)
- [ ] DeepPhys example [paper](https://arxiv.org/abs/1805.07888)
- [ ] MTTS example [paper](https://papers.nips.cc/paper/2020/file/e1228be46de6a0234ac22ded31417bc7-Paper.pdf)
- [ ] DeepPhys + LSTM example paper
- [x] physNet [example](https://github.com/remotebiosensing/rppg/blob/main/rppg/examples/physnet_ubfc_ubfc.py) [paper](https://arxiv.org/abs/1905.02419)
- [ ] 2D phsyNet example [paper](https://arxiv.org/abs/1905.02419)
- [ ] APNETv2 [example](https://github.com/remotebiosensing/rppg/blob/main/rppg/examples/apnetv2_ubfc_ubfc.py) paper

- #### CNIBP (Continuous non-invasive blood pressure)
- [ ] PP-Net exmaple [paper](https://ieeexplore.ieee.org/document/9082808)

### Documentation(TBD)

### Performance Comparison
- rPPG

| MODEL | Train/val Dataset | Test Dataset | lr  | MAE | RMSE | MAPE | r   |
|-------|-------------------|--------------|-----|-----|------|------|-----|

- CNIBP


[//]: # ()
[//]: # (## )

[//]: # ()
[//]: # (## Additional info)

[//]: # ()
[//]: # (#####  *  How to test  &#40;Assessment of ROI selection for facial video based rPPG&#41;)

[//]: # ()
[//]: # (- before test modify sample2.cfg&#40;./pyVHR/analysis/sample2.cfg&#41;)

[//]: # ()
[//]: # (~~~)

[//]: # ([DEFAULT])

[//]: # (''')

[//]: # (methods         = ['POS','CHROM','ICA','SSR','LGI','PBV','GREEN'] # Change Method)

[//]: # (''')

[//]: # ([VIDEO])

[//]: # (dataset     = LGI_PPGI # change dataset)

[//]: # (videodataDIR= /media/hdd1/LGGI/ # change dataset path)

[//]: # (BVPdataDIR  = /media/hdd1/LGGI/)

[//]: # (;videoIdx    = all)

[//]: # (videoIdx    = [1,2,5,6] # change test video idx)

[//]: # (detector    = media-pipe # use media-pipe, it's proposed ROI option)

[//]: # (~~~)

[//]: # ()
[//]: # (- before test, modify test suit file&#40;./pyVHR/analysis/testsuite.py&#41;, all regions one-hot mapping.)

[//]: # ()
[//]: # (~~~)

[//]: # (   ''')

[//]: # (   test for all region)

[//]: # (    ''')

[//]: # (    # tmp = bin&#40;test&#41;)

[//]: # (    # binary = '')

[//]: # (    # for i in range&#40;mask_num-len&#40;tmp[2:]&#41;&#41;:)

[//]: # (    #     binary += '0')

[//]: # (    # binary += tmp[2:])

[//]: # (    ''')

[//]: # (    test for top-5 & bot -5)

[//]: # (    ''')

[//]: # (    if test_case == 0 :)

[//]: # (        binary = '0011000000000000000100000001001')

[//]: # (    else :)

[//]: # (        binary = '0000000001100001011000000000000')

[//]: # (~~~)

[//]: # ()
[//]: # (* run _1_rppg_assesment.py)

[//]: # ()
[//]: # (* all mask information found at video.py's make_mask function &#40;./pyVHR/signals/video.py&#41;)

## Community

Our community is eagerly waiting for researchers and developers interested in non-contact/non-invasive algorithm research and development to [join](https://join.slack.com/t/remobebiosensing/shared_invite/zt-1u3kjfhf9-xWw_XQ8hGd7qFZymCSzUtg) us.

<a href="https://github.com/remotebiosensing/rppg/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=remotebiosensing/rppg" />
</a>


## Contacts

- Dae Yeol Kim, spicyyeol@gmail.com  
- Kwangkee Lee, kwangkeelee@gmail.com  

## Funding

This work was partly supported by the ICT R&D program of
MSIP/IITP. [2021(2021-0-00900), Adaptive Federated Learning in Dynamic Heterogeneous Environment]

## reference

1. [ZitongYu/PhysNet](https://github.com/ZitongYu/PhysNet)
