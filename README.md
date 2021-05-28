# NeuralNILM_Pytorch

- **Introduction**

  Given that it is difficult to reproduce some NILM algorithms, we create this code repositories implementing some state of the art (or classical) **energy disaggregation algorithms** by the means of prevailing deep learning toolkit **Pytorch** [1] and noted NILM toolkit **nilmtk** [2].

------

- **Set up**

  - Create your own virtual environment with **Python > 3.6**

  - Configure deep learning environment with **pytorch (GPU edition) ≥ 1.3.0 + cuDNN**

  - Install other necessary dependencies, such as **Matplotlib, Scikit-learn** etc.

  - Clone this repository (Please notice that **code in the folder `\nilmtk\.` is slightly different from the code in [2], so please clone this folder too**)

  The environments we used are listed in the file `environment.yml`. If you use `conda`, you can use `conda env create -f environment.yml` to set up the environment.

------

- **File Directory**

  Since the original nilmtk toolkit may seem to be redundant for testing Neural NILM algorithms (Deep Learning method), we sorely use it  for the generation of power data within a specific period of time. Thus you can **only focus on  these files or folders**: `\nilmtk\api.py`, `\nilmtk\loss.py`, `\nilmtk\disaggregate`, `\tutorial\experiment_example.ipynb` and  `\tutorial\code_example.ipynb`
  
  The whole file directory is as follow (We omit some unimportant details)：
  
  ```
  ├── README.md							                   
  ├── environment.yml						//Environment dependencies
  ├── nilm_metadata          				
  │   └── *								//Some details are omitted
  ├── tutorial                     
  │   ├── experiment_example.ipynb     	//How to carry out your own NILM experiment
  │   ├── code_example.ipynb			    //How to write your own algorithms or metrics
  ├── nilmtk                     
  │   ├── api.py							//The core api to carry out NILM experiment
  │   ├── loss.py							//The evaluation Metrics
  │   ├── diaggregate
  |   |	├── __init__.py
  |   |	├── attention_pytorch.py		   //Seq2Seq with Attention
  |   |	├── bilstm_pytorch.py			   //BiLSTM
  |   |	├── dae_pytorch.py				   //Denoising AutoEncoder
  |   |	├── disaggregator.py			   //Base Class
  |   |	├── energan_pytorch.py			   //EnerGAN
  |   |	├── seq2point_pytorch.py		   //Seq2Point
  |   |   ├── attention_cnn_pytorch.py       //CNN_Attention
  |   |   ├── seq2seqcnn_pytorch.py          //CNN_Seq2Seq
  |   |   ├── bilstm_pytorch_multidim.py     //Multiple input features BiLSTM
  |   |   ├── dae_pytorch_multidim.py        //Multiple input features DAE
  |   |   ├── seq2point_pytorch_multidim.py  //Multiple input features Seq2Point
  |   |	└── sgn_pytorch.py				   //SGN
  │   └── *								   //Some details are omitted
  ```

------

- **Algorithm Details**

  In the folder `\nilmtk\disaggregate`, you may find several NILM algorithms, they are listed as follow:

  - Denoising AutoEncoder [3]
- BiLSTM [3]
  - Seq2Point [4]
- Seq2Seq [4]
  - Seq2Seq with Attention [5]
- SGN [6]
  - EnerGAN [7]
- CNN_Attention[8]
  
And several NILM algorithms with '_**multidim**' suffix, such as bilstm_pytorch_multidim, ae_pytorch_multidim, seq2point_pytorch_multidim. They are original algorithms with multiple input features(P or P + Q or P + S O or  P + Q + S), which are **not included in nilmtk**[2]
  
  Notice that our implementations of BiLSTM[3] and EnerGAN[7] are **slightly different from original papers**, and the experiment results have shown that our method will result in improved accuracy. To avoid confusing users, we will list our implementation details as follow: 
  
  - For BiLSTM,  the input of the first fully connected layer is the concat of **all the hidden states** instead of the last hidden state which was the way Kelly used.
  - For EnerGAN, we incorporate **reconstruction loss**(L1 loss) while training Generator, which is proved to be valid in most **pix2pix tasks**[9].

------

- **Tutorial**

  - Refer to  `\tutorial\experiment_example.ipynb` to know how to **carry out your own NILM experiment**.

  - Refer to  `\tutorial\code_example.ipynb` to know how to **write your own energy disaggregation algorithms** or **evaluation metrics**.

------
- **References**

  [1] https://pytorch.org/

  [2] https://github.com/nilmtk/nilmtk

  [3] KELLY J, KNOTTENBELT W. Neural NILM: Deep neural networks applied to energy disaggregation[C]//The 2nd ACM International Conference on Embedded Systems for Energy-Efficient Built Environments. November 4-5, 2015, Seoul, South Korea: 55-64.

  [4] ZHANG C, ZHONG M, WANG Z, et al. Sequence-to-point learning with neural networks for non-intrusive load monitoring[C]//The 32nd AAAI Conference on Artificial Intelligence. February 2–7, 2018, Louisiana, USA: 2604–2611.

  [5] WANG Ke, ZHONG Haiwang, YU Nanpeng, et al. Nonintrusive Load Monitoring based on Sequence-to-sequence Model With Attention Mechanism[J]. Proceedings of the CSEE,2019,39(1):75-83.

  [6] SHIN C, JOO S, YIM J. Subtask Gated Networks for Non-Intrusive Load Monitoring[J]. Proceedings of the AAAI Conference on Artificial Intelligence, 2019, 33(1): 1150–1157.

  [7] KASELIMI M, VOULODIMOS A, PROTOPAPADAKIS E, et al. EnerGAN: A GENERATIVE ADVERSARIAL NETWORK FOR ENERGY DISAGGREGATION[C/OL]//ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). Barcelona, Spain: IEEE, 2020: 1578–1582.

  [8] XU Xiaohui, ZHAO Shutao, CUI Kebin. Non-intrusive Load Disaggregate Algorithm Based on Convolutional Block Attention Module[J/0L]. Power System Technology, 2021: 1-8.

  [9] ISOLA P, ZHU JY, ZHOU T, et al. Image-to-Image Translation with Conditional Adversarial Networks[C/OL]//2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Honolulu, HI: IEEE, 2017: 5967–5976.

  ------

- **Future Work To Do** 

  Override some of the functions which are repeated in the current framework with sealed APIs.
