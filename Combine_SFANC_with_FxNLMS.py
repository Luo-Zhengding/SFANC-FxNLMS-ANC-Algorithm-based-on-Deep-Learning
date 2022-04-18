import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import scipy.signal as signal
import scipy.io as sio

#------------------------------------------------------------------------------
# Class: FxNLMS algorithm with initial coefficients determined by SFANC
#------------------------------------------------------------------------------
class FxNLMS():
    
    def __init__(self, Len, Ws):
        self.Wc = torch.tensor(Ws, requires_grad=True) # Ws: initial coefficients determined by SFANC
        self.Xd = torch.zeros(1, Len, dtype=torch.float)
    
    def feedforward(self,Xf):
        self.Xd = torch.roll(self.Xd,1,1)
        self.Xd[0,0] = Xf 
        yt = self.Wc @ self.Xd.t()
        power = self.Xd @ self.Xd.t() # different from FxLMS
        return yt, power
    
    def LossFunction(self, y, d, power):
        e = d-y
        return e**2/(2*power), e
    
    def _get_coeff_(self):
        return self.Wc.detach().numpy()

#----------------------------------------------------------------
# Function: SFANC_FxNLMS
# Description: Using FxNLMS to optimize the control filter, the initial weights come from SFANC
#----------------------------------------------------------------
class SFANC_FxNLMS():
    def __init__(self, MAT_FILE, fs):
        self.Wc = self.Load_Pretrained_filters_to_tensor(MAT_FILE) # torch.Size([15, 1024])
        Len = self.Wc.shape[1]
        self.fs = fs
        self.Current_Filter = torch.zeros(1, Len, dtype=torch.float)
    
    def noise_cancellation(self, Dis, Fx, filter_index, Stepsize):
        Error = []
        j = 0
        model = FxNLMS(Len=1024, Ws=self.Current_Filter)
        optimizer = optim.SGD([model.Wc], lr=Stepsize) # Stepsize is learning_rate
        
        for ii, dis in enumerate(Dis):
            y,power = model.feedforward(Fx[ii])
            loss,e = model.LossFunction(y,dis,power)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Error.append(e.item())
            
            if (ii + 1) % self.fs == 0:
                print(j)
                if self.Current_Filter[0].equal(self.Wc[filter_index[j]]) == False: 
                    # if prediction index is changed, change initial weights of FxNLMS
                    print('change the initial weights of FxNLMS')
                    self.Current_Filter = self.Wc[filter_index[j]].unsqueeze(0) # torch.Size([1, 1024])
                    model = FxNLMS(Len=1024, Ws=self.Current_Filter)
                    optimizer = optim.SGD([model.Wc], lr=Stepsize) # Stepsize is learning_rate
                j += 1
        return Error
        
    def Load_Pretrained_filters_to_tensor(self, MAT_FILE): # Loading the pre-trained control filter from the mat file
        mat_contents = sio.loadmat(MAT_FILE)
        Wc_vectors = mat_contents['Wc_v']
        return torch.from_numpy(Wc_vectors).type(torch.float)