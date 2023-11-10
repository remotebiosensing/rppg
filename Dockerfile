FROM xychelsea/anaconda3:latest-gpu


USER root
RUN apt-get -y update && apt-get -y upgrade
RUN apt-get install -y git libgl1-mesa-glx



CMD ["/bin/bash"]

USER anaconda
WORKDIR /home/anaconda

RUN git clone https://github.com/remotebiosensing/rppg.git -b main_docker_cohface

RUN cd rppg && conda env create -y -f rppg.yaml
RUN conda activate rppg
RUN conda install -y wandb neurokit2 h5py
RUN conda install scipy