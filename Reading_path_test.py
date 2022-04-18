import os 
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal, misc
import scipy.io as sio


def loading_paths(folder="Duct_path", Pri_path_file_name = "Primary Path.csv", Sec_path_file_name="Secondary Path.csv"):
    Primay_path_file, Secondary_path_file = os.path.join(folder,Pri_path_file_name), os.path.join(folder,Sec_path_file_name)
    Pri_dfs, Secon_dfs   = pd.read_csv(Primay_path_file), pd.read_csv(Secondary_path_file)
    Pri_path, Secon_path = np.array(Pri_dfs['Amplitude - Plot 0']), np.array(Secon_dfs['Amplitude - Plot 0'])
    return Pri_path, Secon_path

# Pz1.mat saves the primary path, Sz.mat saves the Secondary Path
def loading_paths_from_MAT(folder, subfolder, Pri_path_file_name, Sec_path_file_name):
    Primay_path_file, Secondary_path_file = os.path.join(folder, subfolder, Pri_path_file_name), os.path.join(folder,subfolder, Sec_path_file_name)
    Pri_dfs, Secon_dfs = sio.loadmat(Primay_path_file), sio.loadmat(Secondary_path_file)
    Pri_path, Secon_path = Pri_dfs['Pz1'].squeeze(), Secon_dfs['S'].squeeze()
    return Pri_path, Secon_path