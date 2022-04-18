import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from Reading_path_test import loading_paths_from_MAT
from Disturbance_generation import Disturbance_reference_generation_from_Fvector
from FxLMS_algorithm import FxLMS_algroithm, train_fxlms_algorithm

def save_mat__(FILE_NAME_PATH, Wc):
    mdict= {'Wc_v': Wc}
    savemat(FILE_NAME_PATH, mdict)
    
#-----------------------------------------------------------------------------------
# Class: frequencyband_design() is also used in Filter_design.py
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

def main():
    FILE_NAME_PATH = 'models/Pretrained_Control_filters.mat'
    
    # Configurating the system parameters
    fs = 16000
    T = 30
    Len_control = 1024
    level = 4
    
    # F_levles converted to Frequecy_band
    Frequecy_band = []
    for i in range(level):
        F_vec, _ = frequencyband_design(i, fs)
        Frequecy_band += F_vec # F_vector is the same as Frequecy_band
    print(len(Frequecy_band)) # 15
    
    # Loading the primary and secondary path
    Pri_path, Secon_path = loading_paths_from_MAT(folder='Pz and Sz', subfolder='Dongyuan', Pri_path_file_name='Primary_path.mat', Sec_path_file_name='Secondary_path.mat')
    
    # Training the control filters from the defined mat file
    num_filters = 15
    Wc_matrix = np.zeros((num_filters, Len_control), dtype=float)
    
    for ii, F_vector in enumerate(Frequecy_band):
        Dis, Fx = Disturbance_reference_generation_from_Fvector(fs=fs, T=T, f_vector=F_vector, Pri_path=Pri_path, Sec_path=Secon_path)
        controller = FxLMS_algroithm(Len=Len_control)
        
        Erro = train_fxlms_algorithm(Model=controller, Ref=Fx, Disturbance=Dis, Stepsize=0.0001)
        Wc_matrix[ii] = controller._get_coeff_()
        
        # Drawing the impulse response of the primary path
        plt.title('The error signal of the FxLMS algorithm')
        plt.plot(Erro)
        plt.ylabel('Amplitude')
        plt.xlabel('Time')
        plt.grid()
        plt.show()
        
    save_mat__(FILE_NAME_PATH, Wc_matrix)

if __name__ == "__main__":
    main()