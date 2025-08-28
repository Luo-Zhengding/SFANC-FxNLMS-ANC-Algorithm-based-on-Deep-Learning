# A-Hybrid-SFANC-FxNLMS-Algorithm
Description：
This is the code of SPL paper "A Hybrid SFANC-FxNLMS Algorithm for Active Noise Control based on Deep Learning".
You can find the paper at https://arxiv.org/pdf/2208.08082.pdf or at IEEE Xplore.

The paper proposes a hybrid SFANC-FxNLMS approach to overcome the adaptive algorithm's slow convergence and provide a better noise reduction level than the SFANC method. A lightweight one-dimensional convolutional neural network (1D CNN) is designed to automatically select the most suitable pre-trained control filter for each frame of the primary noise. Meanwhile, the FxNLMS algorithm continues to update the coefficients of the chosen pre-trained control filter at the sampling rate.
![Fig1_00](https://user-images.githubusercontent.com/95018034/163777818-985cac62-74fb-4585-84d4-c4d9b29fc0e6.png)


Platform: NVIDIA-SMI 466.47, Driver Version: 466.47, CUDA Version: 11.3

Environment: Jupyter Notebook 6.4.5, Python 3.9.7, Pytorch 1.10.1


Run Instructions:

To train the 1D CNN model, we generated 80,000 broadband noise tracks with various frequency bands, amplitudes, and background noise levels at random. Each track has a duration of 1 second. The synthetic noise dataset was subdivided into three subsets: 80,000 noise tracks for training, 2,000 noise tracks for validation, and 2,000 noise tracks for testing. The entire dataset is available at https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/ETJWLU

If you don't want to train the model. The trained 1D model stored in "Trained models/model.pth" can be used directly.

Active noise control based on the proposed hybrid SFANC-FxNLMS algorithm on real-record noises. You can easily run "SFANC-FxNLMS for ANC.ipynb"
The real noises are provided in "Real Noise Examples/"


Citation: 
If you find the hybrid SFANC-FxNLMS algorithm useful in your research, please consider citing:
@ARTICLE{9761749,
  author={Luo, Zhengding and Shi, Dongyuan and Gan, Woon-Seng},
  journal={IEEE Signal Processing Letters}, 
  title={A Hybrid SFANC-FxNLMS Algorithm for Active Noise Control Based on Deep Learning}, 
  year={2022},
  volume={29},
  pages={1102-1106},
  doi={10.1109/LSP.2022.3169428}}

## Related Publications
1. **Real-time implementation and explainable AI analysis of delayless CNN-based selective fixed-filter active noise control**  
   *Journal*: Mechanical Systems and Signal Processing, 2024, 214: 111364.  
   *Paper Link*: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0888327024002620) &nbsp; *Code Link*: [GitHub](https://github.com/Luo-Zhengding/SFANC-Window)

2. **Frequency-Direction Aware Multichannel Selective Fixed-Filter Active Noise Control based on Multi-Task Learning**  
   *Journal*: IEEE Transactions on Audio, Speech and Language Processing, 2025, 33: 3137-3147.
   *Paper Link*: [IEEE](https://ieeexplore.ieee.org/document/11082568) &nbsp; *Code Link*: [GitHub](https://github.com/Luo-Zhengding/Frequency-Direction-MCSFANC)

3. **Performance Evaluation of Selective Fixed-filter Active Noise Control based on Different Convolutional Neural Networks**  
   *Conference*: The 51st International Congress and Exposition on Noise Control Engineering (Inter-Noise 2022)
   *Paper Link*: [arXiv](https://arxiv.org/pdf/2208.08440)

**If you are interested in our works, please consider citing our papers. Thanks! Have a great day!**
