##--- data repo: https://github.com/Jakobovski/free-spoken-digit-dataset.git

from scipy.io import wavfile
import scipy.stats as st
import os
import numpy as np
from python_speech_features import mfcc
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cbook import flatten

def plot_mfcc(feat):
    fig, ax = plt.subplots()
    mfcc_data= np.swapaxes(feat, 0 ,1)
    cax = ax.imshow(mfcc_data, interpolation='none', cmap=cm.coolwarm, origin='lower')
    ax.set_title('MFCC')
    plt.show()

def plot_histogram(feat):
    plt.hist(feat)    
    plt.show()


pi = np.random.RandomState(0).rand(6, 1)
A = np.random.RandomState(0).rand(6, 6)
pi = (pi + (pi==0))/np.sum(pi)
print(pi)
A = (A+(A==0))/np.sum(A,axis=1)
print(A)

exit()

os.chdir(os.path.dirname(os.path.abspath(__file__)))

file_list = []
for i in range(10):
    file_list.append([ f for f in os.listdir('../recordings') if f.split('_')[0] == str(i)])

mfeatures = []
for fi in file_list:
    temp = []
    for f in fi:
        _, d = wavfile.read('../recordings/'+f)
        temp.append(mfcc(d,_))
    mfeatures.append(temp)
    break

plot_mfcc(mfeatures[0][0])
# plot_mfcc(mfeatures[1][0])

# flat0 = list(flatten(mfeatures[0]))
# flat1 = list(flatten(mfeatures[1]))

# plot_histogram(flat0)
# plot_histogram(flat1)
