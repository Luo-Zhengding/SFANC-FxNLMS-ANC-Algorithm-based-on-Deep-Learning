Description：
This is the code of SPL paper "A Hybrid SFANC-FxNLMS Algorithm for Active Noise Control based on Deep Learning".
The paper proposes a hybrid SFANC-FxNLMS approach to overcome the adaptive algorithm's slow convergence and provide a better noise reduction level than the SFANC method. A lightweight one-dimensional convolutional neural network (1D CNN) is designed to automatically select the most suitable pre-trained control filter for each frame of the primary noise. Meanwhile, the FxNLMS algorithm continues to update the coefficients of the chosen pre-trained control filter at the sampling rate.


Platform: NVIDIA-SMI 466.47, Driver Version: 466.47, CUDA Version: 11.3
Environment: Jupyter Notebook 6.4.5, Python 3.9.7, Pytorch 1.10.1


Run Instructions:
1.Training and tesing 1D network used for classifying noises:
run "Train_Testing_1D_Network.ipynb"

To train the 1D CNN model, we generated 80,000 broadband noise tracks with various frequency bands, amplitudes, and background noise levels at random. Each track has a duration of 1 second.
The synthetic noise dataset was subdivided into three subsets: 80,000 noise tracks for training, 2,000 noise tracks for validation, and 2,000 noise tracks for testing.
The trained 1D model is stored in "Trained models/model.pth"

2. Active noise control based on the proposed hybrid SFANC-FxNLMS algorithm on real-record noises.
run "SFANC-FxNLMS for ANC.ipynb"
The real noises are provided in "Real Noise Examples/"


Citation: 
If you find the hybrid SFANC-FxNLMS algorithm useful in your research, please consider citing.


Contact Information:
Authors: Zhengding Luo, Dongyuan Shi, Woon-Seng Gan
The School of Electrical and Electronic Engineering, Nanyang Technological University, Singapore.
(e-mail: LUOZ0021@e.ntu.edu.sg; dongyuan.shi@ntu.edu.sg)

