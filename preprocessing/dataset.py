from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib_inline
import numpy as np
import shutil
import os
import posixpath
import wfdb

fname = '/tmp/pycharm_project_811/VBPNet/039'


def read_ann(annfile):
    annotation = wfdb.rdann(annfile, 'not')
    annotation.extension = 'cpy'

    annotation.wrann()
    print(annfile.split('/')[-1])
    annotation_cpy = wfdb.rdann(annfile.split('/')[-1], 'cpy')

    print(annotation_cpy)


# read_ann(fname)
"""
wfdb.io.rdrecord()
Read a WFDB record and return the signal and record descriptors 
as attributes in a Record or MultiRecord object.

"""


def find_sig_idx(path):
    sig_list = []
    record = wfdb.rdrecord(path)
    sig_name = record.sig_name
    sig_list.append(sig_name.index('ABP'))
    sig_list.append(sig_name.index('PLETH'))
    return sig_list


def read_data(path, sampfrom=0, sampto=None):
    record = wfdb.rdrecord(path, channels=find_sig_idx(path),
                           sampfrom=sampfrom, sampto=sampto)
    wfdb.plot_wfdb(record=record, title='Record 039 from PhysioNet MIMIC Dataset')
    display(record.__dict__)
    return record

# read_hea(hpath)
