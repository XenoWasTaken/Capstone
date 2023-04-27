from scipy.io import wavfile
import numpy as np

# read in multichannel WAV file
sample_rate, wavdata = wavfile.read('test.wav')

# convert to mono
if wavdata.ndim > 1:
    mono_wavdata = np.mean(wavdata, axis=1)
else:
    mono_wavdata = wavdata

# write mono data to new WAV file
wavfile.write('test.wav', sample_rate, mono_wavdata.astype(np.float32))