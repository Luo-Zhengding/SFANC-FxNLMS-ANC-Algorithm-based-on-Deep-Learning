import numpy as np
import scipy.signal as signal
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from scipy.fft import fft, fftfreq, ifft

str(torchaudio.get_audio_backend())

import os, sys
import math
import pandas as pd

def BandlimitedNoise_generation(f_star, Bandwidth, fs, N):
    # f_star indecats the start of frequency band (Hz)
    # Bandwith denots the bandwith of the boradabnd noise 
    # fs denots the sample frequecy (Hz)
    # N represents the number of point
    len_f = 1024 
    f_end = f_star + Bandwidth
    b2 = signal.firwin(len_f, [f_star, f_end], pass_zero='bandpass', window ='hamming',fs=fs)
    xin = np.random.randn(N)
    Re = signal.lfilter(b2,1,xin)
    Noise = Re[len_f-1:]
    #----------------------------------------------------
    return Noise/np.sqrt(np.var(Noise))

def additional_noise(signal, snr_db):
    signal_power = signal.norm(p=2)
    length = signal.shape[1]
    additional_noise = np.random.randn(length)
    additional_noise = torch.from_numpy(additional_noise).type(torch.float32).unsqueeze(0)
    noise_power = additional_noise.norm(p=2)
    snr = math.exp(snr_db / 10)
    scale = snr * noise_power / signal_power
    noisy_signal = (scale * signal + additional_noise) / 2
    return noisy_signal

class SoundGenerator:
    def __init__(self, fs, folder):
        self.fs = fs 
        self.len = fs + 1023 
        self.folder = folder 
        self.Num = 0 
        try: 
            os.mkdir(folder)
        except:
            print("folder exists")
    
    def _construct_(self):
        self.Num = self.Num + 1 
        f_star = np.random.uniform(20, 7880, 1)
        bandWidth = np.random.uniform(1,7880-f_star,1)
        f_end = f_star + bandWidth
        filename = f'{self.Num}_Frequency_from_'+ f'{f_star[0]:.0f}_to_{f_end[0]:.0f}_Hz.wav'
        filePath = os.path.join(self.folder, filename)
        noise = BandlimitedNoise_generation(f_star[0], bandWidth[0], self.fs, self.len)
        noise = torch.from_numpy(noise).type(torch.float32).unsqueeze(0)
        torchaudio.save(filePath, noise, self.fs)
        return f_star[0], f_end[0], filename
    
    def _construct_A(self):
        self.Num  = self.Num + 1 
        f_star = np.random.uniform(20, 7880, 1)
        bandWidth = np.random.uniform(1,7880-f_star,1)
        f_end = f_star + bandWidth
        filename = f'{self.Num}_Frequency_from_'+ f'{f_star[0]:.0f}_to_{f_end[0]:.0f}_Hz_A.wav'
        filePath = os.path.join(self.folder, filename)
        noise = BandlimitedNoise_generation(f_star[0], bandWidth[0], self.fs, self.len)
        noise = torch.from_numpy(noise).type(torch.float32).unsqueeze(0)
        snr_db = np.random.uniform(3, 60, 1)
        noise = additional_noise(noise, snr_db) # add additional noise
        torchaudio.save(filePath, noise, self.fs)
        return f_star[0], f_end[0], filename

class DatasetSheet:
    
    def __init__(self, folder, filename):
        self.filename = filename 
        self.folder = folder
        try: 
            os.mkdir(folder, 755)
        except:
            print("folder exists")
        self.path = os.path.join(folder, filename)
    
    def add_data_to_file(self, wave_file, class_ID):
        dict = {'File_path': [wave_file], 'Class_ID': [class_ID]}
        df = pd.DataFrame(dict)
        
        with open(self.path, mode = 'a') as f:
            df.to_csv(f, header=f.tell()==0)
        
    def flush(self):
        dc = pd.read_csv(self.path, index_col=0)
        dc.index = range(len(dc))
        dc.to_csv(self.path)

def plot_specgram(waveform, sample_rate, filepath, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.savefig(filepath)
    plt.close('all')

#-------------------------------------------------------------
# Function: frequencyband_design(level,fs)
# Description: Creating the bound of the frequnecy. 
#-------------------------------------------------------------
def frequencyband_design(level,fs):
    # the number of filter equals 2^level.
    # fs represents the sampling rate.
    Num = 2**level
    # Computing the start and end of the frequency band.
    #----------------------------------------------------
    F_vector = []
    f_start = 20
    f_marge = 20 
    # the wideth of the frequency band
    width = (fs/2-f_start-f_marge)//Num 
    #----------------------------------------------------
    for ii in range(Num):
        f_end = f_start + width 
        F_vector.append([f_start,f_end])
        f_start = f_end 
    #----------------------------------------------------
    return F_vector, width

#-------------------------------------------------------------
# Function: SimilarityRato(f1_min, f1_max, f2_min, f2_max)
# Description: Geting Class ID of generated noise through computing Similarity between bandpass filter and noise.  
#-------------------------------------------------------------
def SimilarityRato(f1_min, f1_max, f2_min, f2_max):
    if (f1_min <= f2_min):
        if (f1_max <= f2_min):
            return 0
        elif (f2_min <= f1_max) & (f1_max <= f2_max):
            return (f1_max-f2_min)/(f2_max-f1_min)
        else:
            return (f2_max-f2_min)/(f1_max-f1_min)
    else:
        if (f2_max <= f1_min):
            return 0
        elif (f1_min <= f2_max)&(f2_max <= f1_max):
            return (f2_max-f1_min)/(f1_max-f2_min)
        else:
            return (f1_max-f1_min)/(f2_max-f2_min)


class ClassID_Calculator:
    
    def __init__(self, levels, fs):
        self.f_vector = []
        for level in range(levels):
            a_vector,_ = frequencyband_design(level,fs) 
            self.f_vector = self.f_vector + a_vector 
        self.len = len(self.f_vector)
            
    def _get_ID_(self, f_low, f_high):
        SimlarityRatio = []
        for ii in range(self.len):
            SimlarityRatio.append(SimilarityRato(f_low, f_high, self.f_vector[ii][0],self.f_vector[ii][1]))
        ID = SimlarityRatio.index(max(SimlarityRatio))
        return ID, SimlarityRatio

#--------------------------------------------------------------------------------------
# Function: Generating Dataset as given frequency level (It comes from main function)
#--------------------------------------------------------------------------------------
def Generating_Dataset_as_Given_Frequencylevels(N_sample, level, Folder_name):
    import progressbar
    bar = progressbar.ProgressBar(maxval=N_sample, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    file_name = "Index.csv"
    
    Generator = SoundGenerator(fs=16000, folder = Folder_name)
    datasheet = DatasetSheet(Folder_name, file_name)
    ID_calculator = ClassID_Calculator(levels=level, fs=16000)
    
    bar.start()
    for ii in range(N_sample):
        f_star, f_end, filePath = Generator._construct_()
        ID, SR = ID_calculator._get_ID_(f_low=f_star, f_high=f_end)
        datasheet.add_data_to_file(filePath,ID)
        
        f_star, f_end, filePath = Generator._construct_A()
        ID, SR = ID_calculator._get_ID_(f_low=f_star, f_high=f_end)
        datasheet.add_data_to_file(filePath,ID)
        bar.update(ii+1)
    datasheet.flush()
    bar.finish()