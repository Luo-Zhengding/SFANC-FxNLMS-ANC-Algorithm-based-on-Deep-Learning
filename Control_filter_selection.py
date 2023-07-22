import torch
import os
import numpy as np
from torch import nn
import scipy.signal as signal

from Network import m6_res


#-------------------------------------------------------------
# Function: load_weight_for_model()
# Loading pre-trained weights to model
#-------------------------------------------------------------
def load_weigth_for_model(model, pretrained_path, device):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    for k, v in model_dict.items():
        model_dict[k] = pretrained_dict[k]
    model.load_state_dict(model_dict)


def minmaxscaler(data):
    min = data.min()
    max = data.max()    
    return (data)/(max-min)


#-------------------------------------------------------------
# Function: multiple length of samples
#-------------------------------------------------------------
def Casting_multiple_time_length_of_primary_noise(primary_noise, fs):
    assert  primary_noise.shape[0] == 1, 'The dimension of the primary noise should be [1 x samples] !!!'
    cast_len = primary_noise.shape[1] - primary_noise.shape[1]%fs
    return primary_noise[:,:cast_len] # make the length of primary_noise is an integer multiple of fs

def Casting_single_time_length_of_training_noise(filter_training_noise, fs):
    assert filter_training_noise.dim() == 3, 'The dimension of the training noise should be 3 !!!'
    print(filter_training_noise[:,:,:fs].shape)
    return filter_training_noise[:,:,:fs]


#------------------------------------------------------------
# Function : Generating the testing bordband noise 
#------------------------------------------------------------
def Generating_boardband_noise_wavefrom_tensor(Wc_F, Seconds, fs):
    filter_len = 1024 
    bandpass_filter = signal.firwin(filter_len, Wc_F, pass_zero='bandpass', window ='hamming',fs=fs) 
    N = filter_len + Seconds*fs
    xin = np.random.randn(N)
    y = signal.lfilter(bandpass_filter,1,xin)
    yout = y[filter_len:]
    # Standarlize 
    yout = yout/np.sqrt(np.var(yout))
    # return a tensor of [1 x sample rate]
    return torch.from_numpy(yout).type(torch.float).unsqueeze(0)


#-------------------------------------------------------------
# Class : Control_filter_Index_predictor
#-------------------------------------------------------------
class Control_filter_Index_predictor():
    
    def __init__(self, MODEL_PATH, device, fs):
        model = m6_res
        load_weigth_for_model(model, MODEL_PATH, device)
        model = model.to(device)
        model.eval()
        
        self.device = device
        self.model = model
        self.fs = fs
    
    def predic_ID(self, noise): # predict the noise index
        noise = noise.to(self.device) # torch.Size([1, 16000])
        noise = noise.unsqueeze(0) # torch.Size([1, 1, 16000])
        noise = minmaxscaler(noise) # minmax normalization
        prediction = self.model(noise) # torch.Size([15])
        _, pred = prediction.max(0)
        return pred.item() # tensor to int
    
    def predic_ID_vector(self, primary_noise):
        # Checking the length of the primary noise.
        assert  primary_noise.shape[0] == 1, 'The dimension of the primary noise should be [1 x samples] !!!'
        assert  primary_noise.shape[1] % self.fs == 0, 'The length of the primary noise is not an integral multiple of fs.'
        
        # Computing how many seconds the primary noise contain.
        Time_len = int(primary_noise.shape[1]/self.fs) 
        print(f'The primary nosie has {Time_len} seconds !!!')
        
        # Bulding the matric of the primary noise [times x 1 x fs]
        primary_noise_vectors = primary_noise.reshape(Time_len, self.fs).unsqueeze(1)
        
        # Implementing the noise classification for each frame whose length is 1 second. 
        ID_vector = []
        for ii in range(Time_len):
            ID_vector.append(self.predic_ID(primary_noise_vectors[ii]))
        return ID_vector


def Control_filter_selection(fs=16000, Primary_noise=None):
    
    # pretrained CNN model path
    MODEL_PTH = 'Trained models/M6_res.pth'
    device = torch.device('cuda')
    
    # Construct control_filter_ID_pridector based on the pretrained cnn model
    Pre_trained_control_filter_ID_pridector = Control_filter_Index_predictor(MODEL_PATH=MODEL_PTH, device=device, fs=fs)
    
    Primary_noise = Casting_multiple_time_length_of_primary_noise(Primary_noise, fs=fs) # torch.Size([1, 320000]) to torch.Size([1, 320000])
    
    Id_vector = Pre_trained_control_filter_ID_pridector.predic_ID_vector(Primary_noise)
    
    return Id_vector
