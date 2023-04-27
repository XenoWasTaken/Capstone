#imports
import math
import numpy as np
#from matplotlib import pyplot as plt
import librosa
import sys
from scipy import signal
import scipy.io.wavfile as wav
from scipy.fft import fft, ifft
import os
import soundfile as sf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import dump, load
import time
import math

#########################################
# Reading
def mono_read(wavname):
    sample_rate, wavdata = wav.read(wavname)
    if wavdata.ndim > 1:
        wavdata = np.mean(wavdata, axis=1)
    return sample_rate, wavdata
#########################################
# Filter and splitting

def decibel_calc(val):
    return 20*math.log10(val)

def average_decibel_calc(data):
    volume = np.sqrt(np.mean(np.square(data)))
    volume = 20*np.log10(volume)
    return volume

#Bandpass
def bandpass(wavdata, sample_rate):
    low = 50
    high = 3000
    nyq = 0.5*sample_rate
    lowpass = low/nyq
    highpass = high/nyq
    b, a = signal.butter(3, [lowpass, highpass], btype="band")
    filtered_wav = signal.filtfilt(b, a, wavdata, padtype=None)
    return filtered_wav

def wiener_filter(wavdata):
    return signal.wiener(wavdata)

def short_sound_removal(wavdata, pwa_volume, smoothing_factor, sample_rate):
    # pwa_volume -- average volume of the previous 5sec clip.
    # smoothing factor -- float between 0 and 1, 0.3 is empirically tested to be best
    volume = average_decibel_calc(wavdata)
    refnoise = smoothing_factor*volume - (1-smoothing_factor)*pwa_volume
    # Then scan for segments greater than 300ms of nonzeros
    final_sounds = get_long_sounds(wavdata, refnoise, 0.3, sample_rate)
    return final_sounds

def get_long_sounds(samples, threshold, min_duration, sample_rate):
    # min_duration in seconds
    # Find all non-ambient samples
    #print("Length of sample with environmental: %f", len(samples))
    filterout = np.power(10, threshold*1.3/20)
    print(filterout)
    non_ambient = np.where(abs(samples) > filterout, samples, 0)
    #print("Length of sample without environmental: %f", len(test_non_ambient))
    ########################################################
    # Short sound removal.  Instead, just get rid of quiet content
    longer_than = min_duration*sample_rate
    found = False
    start = 0
    window_length = 1
    long_sounds = list()
    for i in range(window_length, len(non_ambient)+1, window_length):
        #New start
        if (average_decibel_calc(non_ambient[i-window_length+1:i]) > 0) and (not found):
            start = i-window_length+1
            found = True
        #Have started, found quiet after long enough time
        elif (average_decibel_calc(non_ambient[i-window_length+1:i]) < 0) and (i-start >= longer_than) and (found):
            found = False
            #record the sound
            long_sounds.append(np.array(non_ambient[start:i-window_length]))
        #Have started, found a 0 before threshold
        elif (average_decibel_calc(non_ambient[i-window_length+1:i]) < 0) and (i-start < longer_than) and (found):
            found = False
    ##############################################################
    # Grab last clip
    if found and (len(non_ambient) - start >= longer_than):
        long_sounds.append(np.array(non_ambient[start:len(non_ambient)-1]))
    return non_ambient

def splitter(wavdata, sample_rate):
    # read
    # Define the clip length in seconds
    clip_length = 0.01

    # Calculate the number of clips
    num_clips = int(np.ceil(len(wavdata) / (sample_rate * clip_length)))

    # Create an empty list to store the clips
    clips = []

    # Loop over the clips
    for i in range(num_clips):
        # Calculate the start and end samples of the clip
        start_sample = int(i * sample_rate * clip_length)
        end_sample = min(int((i + 1) * sample_rate * clip_length), len(wavdata))

        # Extract the clip from the data
        clip = wavdata[start_sample:end_sample]

        # Append the clip to the list
        clips.append(clip)

    #pad final clip to be same length as all other clips
    if len(clips[-1]) != len(clips[0]):
        clips[-1] = np.pad(clips[-1], (0, len(clips[0]) - len(clips[-1])), mode='constant', constant_values=0 )
    return clips

#########################################
# Statistic functions


def compute_entropy(hist): 
    # Calculate the entropy of the histogram
    return -np.sum(hist * np.log2(hist + 1e-6)) #Safeguard value to prevent log of 0

def compute_signal_energy(histogram):
    signal_energy = np.sum(np.square(histogram))/len(histogram)
    return signal_energy

def compute_zero_cross(clip):
    # Set indicator function values of main hist
    signage = np.sign(clip)
    #Find sum
    total = np.sum(np.abs(np.diff(signage)))
    return total/((len(clip)-1))

def compute_spectral_rolloff(fourier, sample_rate):
    # calculate the power spectrum
    magnitude = np.abs(fourier) ** 2
    #accumulate
    cumulative_sum = np.cumsum(magnitude, axis=0)
    # find the frequency bin index at which 85% of the power is contained
    total_power = np.sum(magnitude)
    cutoff_index = np.argmax(cumulative_sum > 0.85 * total_power)
    # calculate the spectral rolloff frequency as the frequency bin center
    # corresponding to the cutoff index
    bins = np.fft.rfftfreq(n=441, d=1/sample_rate)
    return bins[cutoff_index]


def compute_spectral_centroid(clip, sample_rate):
    magnitudes = np.abs(np.fft.rfft(clip, n=441))  # Compute the magnitude spectrum
    freqs = np.fft.rfftfreq(n=441, d=1/sample_rate)  # Compute the frequency bins
    return np.sum(magnitudes * freqs) / np.sum(magnitudes)  # Compute the weighted average of frequency bins

def compute_spectral_flux(fourier_curr, fourier_prev):
    # compute magnitude spectrum for each clip
    magnitude1 = np.abs(fourier_curr)
    magnitude2 = np.abs(fourier_prev)

    # compute spectral flux between the two clips
    spectral_flux = np.sum(np.abs(magnitude2 - magnitude1))

    return spectral_flux
    
#########################################

# Data Driver
def collect_data(wavname, ref_noise=0):
    sample_rate, wavdata = mono_read(wavname)
    #filter the audio bandpass
    bandpass_wav = bandpass(wavdata, sample_rate)
    #Wiener filter
    wiener = wiener_filter(bandpass_wav)
    
    #######################
    #Remove short sounds
    #Comments left in case future improvements are made
    candidates = short_sound_removal(wiener, ref_noise, 0.3, sample_rate)
    print(candidates)
    if (len(candidates) == 0):
        return (10000, 10000, 10000, 10000, 10000, 10000)
        
    ##################################
    features = list() #stores for each clip
    #Preprocessing of original audio complete, begin splitting and feature definition.
    #for candidate in candidates:
    #split into 10ms sections, make sure only 500
    splits = splitter(candidates, sample_rate)
    if len(splits) > 1:
        #take fourier of first clip for flux purposes
        prev_fourier = np.fft.rfft(splits[0], n=441)
        max_prev = max(prev_fourier)
        if max_prev == 0:
            prev_fourier = np.zeros(len(prev_fourier))
        else:
            prev_fourier = [(x/max_prev) for x in prev_fourier]
        #start at second clip for flux purposes
        for clip in splits[1:]:
            fourier = np.fft.rfft(clip, n=441)
            max_fourier = max(fourier)
            if max_fourier == 0:
                normalized_fourier = np.zeros(len(fourier))
            else:
                normalized_fourier = [(x/max_fourier) for x in fourier]
            # Calculate the decibel levels for the clip
            db = librosa.amplitude_to_db(np.abs(clip), ref=np.max)
            # Calculate the histogram of the decibel levels
            hist, bins = np.histogram(db, bins='auto', density=True)
            features.append(np.array([compute_entropy(hist), compute_signal_energy(hist), compute_zero_cross(hist), compute_spectral_rolloff(fourier, sample_rate), 
                                    compute_spectral_centroid(fourier, sample_rate), compute_spectral_flux(normalized_fourier, prev_fourier)]))
            #save for next clips flux
            prev_fourier = normalized_fourier

    # For entropy, ZCT, Roll Off, Centroid, compute SD
    std_entropy = np.std([clip_features[0] for clip_features in features])
    std_ZCT = np.std([clip_features[2] for clip_features in features])
    std_rolloff = np.std([clip_features[3] for clip_features in features])
    std_centroid = np.std([clip_features[4] for clip_features in features])
    # For Signal Energy, Flux, compute SD by Squared Mean
    # Not sure about the difference between these two, ask Prof Murphy maybe? For now, just use regular std
    std_sig_nrg = np.std([clip_features[1] for clip_features in features])
    std_flux = np.std([clip_features[5] for clip_features in features])
    #Features prepped for SVM
    return (std_entropy, std_sig_nrg, std_ZCT, std_rolloff, std_centroid, std_flux)
        
#########################################


# Load Model
model = load('model.joblib')

# Begin cycle
while True:
    os.system("arecord -D plughw:1 -c2 -r 44100 -f FLOAT_LE -t wav -V stereo --duration=5 -v a.wav")
    time.sleep(1)
    reading_file = "a.wav"
    score_tuple = collect_data(reading_file)
    if score_tuple == (10000, 10000, 10000, 10000, 10000, 10000):
        print("nada")
    elif ((model.predict(score_tuple).reshape(1,-1)) == 1):
        print("possible causalty")
    else:
        print("nada")
    print("To begin again, type go and hit enter.")
    input1 = input()
    while(input1 != "go"):
        input1 = input

