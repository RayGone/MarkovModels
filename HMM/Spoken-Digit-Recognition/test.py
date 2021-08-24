##--- data repo: https://github.com/Jakobovski/free-spoken-digit-dataset.git

from numpy.lib.function_base import append
from scipy.io import wavfile
import scipy.stats as st
import os
import numpy as np
from python_speech_features import mfcc
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cbook import flatten
from GMMHMM import GMMHMM

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

mfeatures = np.zeros((len(file_list[0])-1,16000))
i=0
max_size = 0
sr = 0
all_data = []
for fi in file_list:
    for f in fi:
        _, d = wavfile.read(data_path+"/"+f)
        if(d.shape[0] > 9300):
            print('skip')
            continue
        sr = _
        mfeatures[i,:d.shape[0]] = d
        i+=1
        if(max_size < d.shape[0]):
            max_size = d.shape[0]    
    mfeatures = mfeatures[:,:max_size]
    all_data.append(mfeatures)


mfcc_feat = []
for digit in all_data:
    tmp = []
    for i in digit:
        tmp.append(np.array(mfcc(i,sr,numcep=4)).T)
    mfcc_feat.append(tmp)
    
mfcc_feat = np.array(mfcc_feat)


model = []
test_index = []
# train a model for each digit using 70%< of training data; and 30%> as test data
for digit in mfcc_feat:
    hmm_model = GMMHMM(6)
    size = int(digit.shape[0] * 0.7)
    test_index.append(size+1)
    hmm_model.train(digit[:size],10)
    model.append(hmm_model)

#testing
likelihoods = []
for i in range(10):
    test_data = mfcc_feat[i][test_index[i]:]
    likelihoods.append[[]]
    for feat in test_data:
        score = model[i].score(feat)
        likelihoods[i].append(score)
        
