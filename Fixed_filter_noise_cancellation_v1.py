import scipy.io as sio
import torch

#----------------------------------------------------------------
# Function: Fixed_filter_controller
# Description: Conducting fixed-filter ANC based on the control filter
#----------------------------------------------------------------
class Fixed_filter_controller():
    def __init__(self, MAT_FILE, fs):
        self.Wc = self.Load_Pretrained_filters_to_tensor(MAT_FILE) # torch.Size([15, 1024])
        Len = self.Wc.shape[1]
        self.fs = fs
        self.Xd = torch.zeros(1, Len, dtype=torch.float)
        self.Current_Filter = torch.zeros(1, Len, dtype=torch.float)
    
    def noise_cancellation(self, Dis, Fx, filter_index):
        Erro = torch.zeros(Dis.shape[0])
        j = 0
        for ii, dis in enumerate(Dis):
            self.Xd = torch.roll(self.Xd,1,1)
            self.Xd[0,0] = Fx[ii] # Fx[ii]: fixed-x signal
            yt = self.Current_Filter @ self.Xd.t()
            e = dis - yt
            Erro[ii] = e.item()
            if (ii + 1) % self.fs == 0 :
                self.Current_Filter = self.Wc[filter_index[j]]
                j += 1
        return Erro
        
    def Load_Pretrained_filters_to_tensor(self, MAT_FILE): # Loading the control filter from the mat file
        mat_contents = sio.loadmat(MAT_FILE)
        Wc_vectors = mat_contents['Wc_v']
        return torch.from_numpy(Wc_vectors).type(torch.float)