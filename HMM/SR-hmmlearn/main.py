from hmmlearn import hmm
from python_speech_features import mfcc
import numpy as np
import os
from scipy.io import wavfile

#--------------------------------

os.chdir(os.path.dirname(os.path.abspath(__file__)))
data_path = '../../../recordings'

file_list = []
for i in range(10):
    file_list.append([ f for f in os.listdir(data_path) if f.split('_')[0] == str(i)])

max_size = 0
sr = 0
all_data = []
num_cep = 6
print('reading audio files')
for fi in file_list:
    i=0    
    mfeatures = np.zeros((len(fi),32000))
    for f in fi:
        _, d = wavfile.read(data_path+"/"+f)
        sr = _
        mfeatures[i,:d.shape[0]] = d
        if(max_size < d.shape[0]):
            max_size = d.shape[0]
        i+=1    
    mfeatures = mfeatures[:,:max_size]
    all_data.append(mfeatures)

print('File read complete')

mfcc_feat = []
lengths = []
j=0
test_feat = []
print('\nComputing MFCC Features')
for digit in all_data:
    lengths.append([])
    k=0
    print('Digit',j)    
    size = int(digit.shape[0] * 0.98) # training size - per digit
    for i in digit:
        mfc = np.array(mfcc(i,sr,numcep=num_cep))
        if k==0:
            tmp = mfc
        else:           
            tmp = np.concatenate([tmp,mfc])
        
        if k > size:
            test_feat.append(mfc)
        lengths[j].append(mfc.shape[0])
        k+=1
    j+=1
    tmp = tmp.T
    mfcc_feat.append(tmp)
    print('test size',len(test_feat))
    
print('MFCC features computed')
model = []
j=0
# train a model for each digit using 70%< of training data; and 30%> as test data
print('\nStarting Training of the Model')
for digit in mfcc_feat:
    learn = hmm.GMMHMM(n_components=6,n_mix=1)
    learn.fit(digit.T,lengths=lengths[j])
    print('Digit {} Model Trained'.format(j))
    j+=1
    model.append(learn)
print('Training Complete')

print('\nStarting Testing')
predictions = []
j=0
for m in model:
    print('---------- Model - Digit ',j)
    j+=1
    likelihood = []
    for test in test_feat:
        likelihood.append(m.score(test))
    predictions.append(np.array(likelihood))
print('Tests Completed')
predictions = np.array(predictions).T
max = np.max(predictions,axis=1)
result = []
for i in range(0,predictions.shape[0]):
    for j in range(0,predictions[i].shape[0]):
        if(predictions[i][j]==max[i]):
            result.append(j)
            break

print('predictions: ', result)

n = -1
res = []
for i in range(len(result)):
    if i%5 == 0:
        n+=1
    if(result[i] == n):
        res.append(1)
    else:
        res.append(0)
res = np.array(res)

print("accuracy: ",np.mean(res))


    