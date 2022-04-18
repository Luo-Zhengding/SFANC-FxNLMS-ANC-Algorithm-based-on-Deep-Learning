import numpy as np
import math 
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import repeat 
from scipy import signal, misc
import torch

#-------------------------------------------------------------
# Function: Disturbance_reference_generation()
# Description: Generate the disturbance and reference by using the defuat paramters.
#-------------------------------------------------------------
def Disturbance_reference_generation():
    # Defined the configuration for the ANC system
    fs = 16000 # The system sampling rate 
    T = 5 # The duraiton of the simulation 
    t = np.arange(0,T,1/fs).reshape(-1,1)
    f0 = 500
    
    # Constructing the refererence signal
    Re = np.random.randn(len(t))
    
    # Define the low-pass filter
    f_cutoff = 2000 
    N_fc = f_cutoff/fs 
    b1, b2 = signal.firwin(128, N_fc), signal.firwin(128, 2*N_fc)
    
    # Constructing primary path
    Pri_path = signal.convolve(b1,b2)
    w1, h1 = signal.freqz(Pri_path)
    w2, h2 = signal.freqz(b2)

    # Drawing the frequency response of the low-pass filter 
    plt.title('Digital filter frequency response')
    plt.plot(w1, 20*np.log10(np.abs(h1)),'b')
    plt.plot(w2, 20*np.log10(np.abs(h2)),'r')
    plt.ylabel('Amplitude Response (dB)')
    plt.xlabel('Frequency (rad/sample)')
    plt.grid()
    plt.show()

    # Drawing the impulse response of the primary path
    plt.title('The response of the primary path')
    plt.plot(b1)
    plt.ylabel('Amplitude')
    plt.xlabel('Length (taps)')
    plt.grid()
    plt.show()
    
    # Construting the desired signal 
    Dir, Fx = signal.lfilter(Pri_path, 1, Re), signal.lfilter(b2, 1, Re)
    print(Fx[1])
    print(Dir.shape, Fx.shape)

    # Drawing the frequency spectrum of the disturbance 
    f, Pper_spec = signal.periodogram(Dir, fs, 'flattop', scaling='spectrum')
    plt.semilogy(f, Pper_spec)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.grid()
    plt.show()

    # Drawing the frequency spectrum of the Fx
    f, Pper_spec = signal.periodogram(Fx, fs, 'flattop', scaling='spectrum')
    plt.semilogy(f, Pper_spec)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.grid()
    plt.show()
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float)

#-------------------------------------------------------------
# Function: Disturbance_reference_generation_from_Fvector()
# Discription : Generating the distubrane and reference signal from the defined parameters
#-------------------------------------------------------------
def Disturbance_reference_generation_from_Fvector(fs, T, f_vector, Pri_path, Sec_path):
    # Pri_path and Sec_path are One dimension arraies
    # constructing bandpass filter based on f_vector
    t = np.arange(0,T,1/fs).reshape(-1,1)
    len_f = 1024
    b2 = signal.firwin(len_f, [f_vector[0],f_vector[1]], pass_zero='bandpass', window ='hamming', fs=fs)
    
    xin = np.random.randn(len(t))
    Re = signal.lfilter(b2,1,xin) # random signal pass the bandpass filter to get the noise
    Noise = Re[len_f-1:]
    
    # Construting the desired signal
    Dir, Fx = signal.lfilter(Pri_path, 1, Noise), signal.lfilter(Sec_path, 1, Noise)
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float)

#-------------------------------------------------------------
# Function: Disturbance_generation_from_real_noise()
# Descritption: Generating the disturbance and filtered reference from the raw waveform
#-------------------------------------------------------------
def Disturbance_generation_from_real_noise(fs, Repet, wave_form, Pri_path, Sec_path):
    wave = wave_form[0,:].numpy()
    wavec = wave
    for ii in range(Repet):
        wavec = np.concatenate((wavec,wave),axis=0) # add the length of the wave_form through repetition
    pass

    # Construting the desired signal
    Dir, Fx = signal.lfilter(Pri_path, 1, wavec), signal.lfilter(Sec_path, 1, wavec)
    
    N = len(Dir)
    N_z = N//fs
    Dir, Fx = Dir[0:N_z*fs], Fx[0:N_z*fs]
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float), torch.from_numpy(wavec).type(torch.float)

#-------------------------------------------------------------
# Function: Varied_distrubance_reference_generation_from_Fvector()
# Discription : Generating the distubrane and reference signal from the defined parameters
#-------------------------------------------------------------
def Varied_distrubance_reference_generation_from_Fvector(fs, T, f_vector, Pri_path, Sec_path):
    t = np.arange(0,T,1/fs).reshape(-1,1)
    len_f = 1024
    for ii in range(len(f_vector)):
        b2 = signal.firwin(len_f, f_vector[ii], pass_zero='bandpass', window ='hamming',fs=fs)
        xin = np.random.randn(len(t))
        Re = signal.lfilter(b2,1,xin)
        if ii == 0:
            Noise = Re[fs:]
        else:
            if ii ==2 :
                Noise = np.concatenate((Noise, 4*Re[fs:]),axis=0)
            else: 
                Noise = np.concatenate((Noise, Re[fs:]),axis=0)
        
    # Construting the desired signal 
    Dir, Fx = signal.lfilter(Pri_path, 1, Noise), signal.lfilter(Sec_path, 1, Noise)
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float), torch.from_numpy(Noise).type(torch.float)