import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.io import savemat
import pandas as pd

#-----------------------------------------------------------------------------------
# Class: frequencyband_design()
# Description : The function is utilized to devide the full frequency band into several equal frequency components.
#-----------------------------------------------------------------------------------
def frequencyband_design(level, fs):
    # the number of filter equals 2^level.
    # fs represents the sampling rate. 
    Num = 2**level
    # Computing the start and end of the frequency band.
    #----------------------------------------------------
    F_vector = []
    f_start  = 20
    f_marge  = 20 
    # the wideth of thefrequency band
    width = (fs/2-f_start-f_marge)//Num 
    #----------------------------------------------------
    for ii in range(Num):
        f_end   = f_start + width 
        F_vector.append([f_start,f_end])
        f_start = f_end 
    #----------------------------------------------------
    return F_vector, width

#-----------------------------------------------------------------------------------
# Class type: Filter design
# Description: Design filter group by the configure vector 
#-----------------------------------------------------------------------------------
class Filter_designer():
    
    def __init__(self, filter_len, F_vector, fs):

        self.filter_len = filter_len
        self.filter_num = len(F_vector)
        self.wc = np.zeros((self.filter_num, self.filter_len))
        for i in range(self.filter_num):
            self.wc[i,:] = signal.firwin(self.filter_len, F_vector[i], pass_zero='bandpass', window ='hamming',fs=fs) 
        
    
    def __save_mat__(self, FILE_NAME_PATH):
        mdict = {'Wc_v': self.wc}
        savemat(FILE_NAME_PATH, mdict)
        
#-----------------------------------------------------------------------------------
# Function: Broadband Filter design by given freqency bands 
#-----------------------------------------------------------------------------------
def Boardband_Filter_Desgin_as_Given_Freqeuencybands(MAT_filename, F_bands, fs):
    Filters = Filter_designer(filter_len=1024, F_vector=F_bands, fs=fs)
    Filters.__save_mat__(MAT_filename)
    print(Filters.filter_num)

#-----------------------------------------------------------------------------------
# Function: Broadband Filter design by give Frequency levels
#-----------------------------------------------------------------------------------
def Broadband_Filter_Design_as_Given_F_levles(MAT_filename, level, fs):
    F_vector = []
    for i in range(level):
        F_vec, _ = frequencyband_design(i, fs)
        F_vector += F_vec
    Filters = Filter_designer(filter_len=1024, F_vector=F_vector, fs=fs)
    Filters.__save_mat__(MAT_filename)
    print(f' There are {Filters.filter_num} pre-train filters has been created!')