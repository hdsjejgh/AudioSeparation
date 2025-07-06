import random
import math
from scipy.io import wavfile
from scipy.io.wavfile import write
import numpy as np
from mixed_maker import normalize

_,data1 = wavfile.read('mixed/mixed_1.wav')
_,data2 = wavfile.read('mixed/mixed_2.wav')
_,data3 = wavfile.read('mixed/mixed_3.wav')
_,data4 = wavfile.read('mixed/mixed_4.wav')

audio_mat = np.array([data1,data2,data3,data4])
audio_mat = audio_mat.T
means = np.mean(audio_mat, axis=0)
std_dev = np.std(audio_mat, axis=0)
audio_mat = (audio_mat - means) / (std_dev+10e-5)
Taudio = audio_mat
audio_mat=audio_mat.T

LEARNING_RATE = 0.001
EPOCHS=500000
NUM_AUDIOS = 4

W = np.random.rand(NUM_AUDIOS,NUM_AUDIOS)

def g(X):
    X = X[0,0]
    return 1/(1+math.exp(-X))

for i in range(EPOCHS):
    ind = random.randint(0,Taudio.shape[0]-1)
    x = np.expand_dims(Taudio[ind],axis=0).T


    dldW = np.expand_dims(np.array([1-2*g(np.expand_dims(W[ii],axis=0) @ x) for ii in range(NUM_AUDIOS)]),axis=1)


    dldW=dldW @ x.T
    dldW += np.linalg.inv(W.T)
    W += LEARNING_RATE*dldW

unmixed = W @ audio_mat
print(W.shape)
print(W)
write('1.wav', _, normalize(unmixed[0]))
write('2.wav', _, normalize(unmixed[1]))
write('3.wav', _, normalize(unmixed[2]))
write('4.wav', _, normalize(unmixed[3]))