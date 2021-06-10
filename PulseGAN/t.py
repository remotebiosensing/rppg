import csv
import matplotlib.pyplot as plt
import numpy as np
import os

def train_result(file_name):
	b=open('./results/generate_{}_5.csv'.format(file_name),'r')
	rdr2 = csv.reader(b)
	result2 = []
	for i,line in enumerate(rdr2):
	  for j in line:
	      j = list(j)
	      j[0]=j[-1]=''
	      new = ''.join(j)
	      result2.append(new)
	  result2 = list(map(float, result2))
	  return result2

def clean_result(file_name):
	rdr3 = csv.reader(open('/home/js/Desktop/Data/Pytorch_rppgs_save/preprocessing/test/result/compare/clean/{}.csv'.format(file_name),'r'))
	for data in rdr3:
	  return list(map(float, data))

def noisy_result(file_name): 
	rdr4 = csv.reader(open('/home/js/Desktop/Data/Pytorch_rppgs_save/preprocessing/test/result/compare/noisy/{}.csv'.format(file_name),'r'))
	for data in rdr4:	  
	  return list(map(float, data))
	  

file_list = sorted(os.listdir('./data/serialized_test_data/'))
for f in file_list:
    f = f.split('.')[0]
    a=train_result(f)
    b=clean_result(f)
    c=noisy_result(f)
    plt.plot(a,label = 'train_25',color='b')
    plt.plot(b,label = 'clean',color='salmon')
    plt.plot(c,label = 'noisy',color='g')
    plt.savefig('./results_img/{}.png'.format(f))
    plt.clf()

