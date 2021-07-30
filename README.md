# Pytorch_rppgs
implement rppg model  using pytorch
### model list
- [x] DeepPhys
- [ ] MTTS
- [ ] DeepPhys + LSTM
- [x] 3D physNet

### preprocessor list
- \_\_TIME__ : check features running time
  + preprocessing time
  + model init time
  + setting loss func time
  + setting optimizer time
  + training time per 1epoch

## file list
~~~
|-- bvpdataset.py    : Data loader
|-- model.py         : Deep Learning Modules
|-- parser.py        : Parser (Main)
|-- preprocessing.py : Data preprocessing
|-- test.py          : test model (load saved model)
`-- train.py         : train model (save model)
~~~
## Contacts
TVSTORM inc.\
Kim Dae Yeol &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Kim Jin Soo\
wagon0004@tvstorm.com &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;wlstn25092303@tvstorm.com

## Funding
 This work was supported by the ICT R&D program of MSIP/IITP. [2021(2021-0-00900), Adaptive Federated Learning in Dynamic Heterogeneous Environment]
