from scipy.io import wavfile
import numpy as np

# read in multichannel WAV file
sample_rate, wavdata = wavfile.read('input1.wav')

# convert to mono
if wavdata.ndim > 1:
    mono_wavdata = np.mean(wavdata, axis=1)
else:
    mono_wavdata = wavdata

# write mono data to new WAV file
wavfile.write('input1_mono.wav', sample_rate, mono_wavdata.astype(np.float32))