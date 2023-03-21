# Implement Deep Learning based Rppg Model & PPG 2 ABP using pytorch

### model list (TODO : UPDATE)

- #### Facial Image Based ppg measurement algorithm
- [x] [DeepPhys : DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks](https://arxiv.org/abs/1805.07888)
- [ ] [MTTS  :Multi-Task Temporal Shift Attention Networks for
  On-Device Contactless Vitals Measurement](https://papers.nips.cc/paper/2020/file/e1228be46de6a0234ac22ded31417bc7-Paper.pdf)
    + need to verification
- [x] DeepPhys + LSTM
- [x] [3D physNet :  Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks](https://arxiv.org/abs/1905.02419)
- [x] [2D phsyNet + LSTM](https://arxiv.org/abs/1905.02419)

- #### PPG Based Blood Pressure estimation algorithm
- [x] [PP-Net: A Deep Learning Framework for PPG based Blood Pressure and Heart Rate Estimation](https://ieeexplore.ieee.org/document/9082808)

## TODO

## Additional info

#####  *  How to test  (Assessment of ROI selection for facial video based rPPG)

- before test modify sample2.cfg(./pyVHR/analysis/sample2.cfg)

~~~
[DEFAULT]
'''
methods         = ['POS','CHROM','ICA','SSR','LGI','PBV','GREEN'] # Change Method
'''
[VIDEO]
dataset     = LGI_PPGI # change dataset
videodataDIR= /media/hdd1/LGGI/ # change dataset path
BVPdataDIR  = /media/hdd1/LGGI/
;videoIdx    = all
videoIdx    = [1,2,5,6] # change test video idx
detector    = media-pipe # use media-pipe, it's proposed ROI option
~~~

- before test, modify test suit file(./pyVHR/analysis/testsuite.py), all regions one-hot mapping.

~~~
   '''
   test for all region
    '''
    # tmp = bin(test)
    # binary = ''
    # for i in range(mask_num-len(tmp[2:])):
    #     binary += '0'
    # binary += tmp[2:]
    '''
    test for top-5 & bot -5
    '''
    if test_case == 0 :
        binary = '0011000000000000000100000001001'
    else :
        binary = '0000000001100001011000000000000'
~~~

* run _1_rppg_assesment.py

* all mask information found at video.py's make_mask function (./pyVHR/signals/video.py)

## Contacts

- Dae Yeol Kim, spicyyeol@gmail.com  
- Kwangkee Lee, kwangkeelee@gmail.com  
- Jin Soo Kim, wlstn25092303@gmail.com  

## Funding

This work was partly supported by the ICT R&D program of
MSIP/IITP. [2021(2021-0-00900), Adaptive Federated Learning in Dynamic Heterogeneous Environment]

## reference

1. [ZitongYu/PhysNet](https://github.com/ZitongYu/PhysNet)
2. [phuselab/pyVHR](https://github.com/phuselab/pyVHR)
