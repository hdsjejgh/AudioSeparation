from scipy.io import wavfile
from scipy.io.wavfile import write
import numpy as np

_,data1 = wavfile.read('base/1.wav')
_,data2 = wavfile.read('base/2.wav')
_,data3 = wavfile.read('base/3.wav')
_,data4 = wavfile.read('base/4.wav')

audio_mat = np.array([data1,data2,data3,data4])
mixing_mat = np.array([[0.6,0.1,0.2,0.2],
                       [0.25,0.5,0.15,0.125],
                       [0.1,0.05,0.5,0.6],
                       [0.05,0.45,0.1,0.4]])
new_audios = mixing_mat @ audio_mat

def normalize(audio):
    max_val = np.max(np.abs(audio))
    audio = audio / max_val
    return (audio * 32767).astype(np.int16)

write('mixed/mixed_1.wav', _, normalize(new_audios[0]))
write('mixed/mixed_2.wav', _, normalize(new_audios[1]))
write('mixed/mixed_3.wav', _, normalize(new_audios[2]))
write('mixed/mixed_4.wav', _, normalize(new_audios[3]))
