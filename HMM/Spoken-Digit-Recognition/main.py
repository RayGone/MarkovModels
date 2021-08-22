##--- data repo: https://github.com/Jakobovski/free-spoken-digit-dataset.git

from scipy.io import wavfile
import scipy.stats as st
import os
import numpy as np
from python_speech_features import mfcc
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cbook import flatten
from GMMHMM import GHMM

def plot_mfcc(feat):
    fig, ax = plt.subplots()
    mfcc_data= np.swapaxes(feat, 0 ,1)
    cax = ax.imshow(mfcc_data, interpolation='none', cmap=cm.coolwarm, origin='lower')
    ax.set_title('MFCC')
    plt.show()

def plot_histogram(feat):
    plt.hist(feat)    
    plt.show()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
data_path = '../../../recordings'

file_list = []
for i in range(10):
    file_list.append([ f for f in os.listdir(data_path) if f.split('_')[0] == str(i)])

mfeatures = np.zeros((len(file_list[0]),16000))
i=0
max_size = 0
sr = 0
for fi in file_list:
    for f in fi:
        _, d = wavfile.read(data_path+"/"+f)
        sr = _
        mfeatures[i,:d.shape[0]] = d
        i+=1
        if(max_size < d.shape[0]):
            max_size = d.shape[0]        
    break

mfeatures = mfeatures[:,:max_size]

mfcc_feat = []
for i in mfeatures:
    mfcc_feat.append(np.array(mfcc(i,sr,numcep=6)).T)
    
mfcc_feat = np.array(mfcc_feat)

model = GHMM(6)
model.train(mfcc_feat[0],10)
print(model.score(mfcc_feat[2]))

# plot_mfcc(mfeatures[1][0])
# flat0 = list(flatten(mfeatures[0]))
# flat1 = list(flatten(mfeatures[1]))

# plot_histogram(flat0)
# plot_histogram(flat1)
