# Image-Captions-Generator
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Markdown](https://img.shields.io/badge/markdown-%23000000.svg?style=for-the-badge&logo=markdown&logoColor=white) ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)  ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

# Introduction

In this project, scripts are provided to download data from the [COCO dataset](http://cocodataset.org/#home) and train a CNN-RNN model for automatically generating image captions. Also inference can be made to predict an input image captions.

# Scripts
| Script Name        | Description                                                                                                 | Required Arguments                                                       |   |   |   |   |   |   |   |
|--------------------|-------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|---|---|---|---|---|---|---|
| data\_download\.py | When run, the COCO training and validation \(test\) dataset files will be downloaded in the specified path  | Dataset Path \(str\): Path to save the downloaded images and annotations |   |   |   |   |   |   |   |
| train\.py          | When run, the neural networks will start training\. The final model will be saved in the specified path     | Model Path \(str\): Path to save the train neural network                |   |   |   |   |   |   |   |
| inference\.py      | TBD                                                                                                         | TBD                                                                      |   |   |   |   |   |   |   |
| model\.py          | Includes the encoder CNN and decoder RNN definition                                                         | None                                                                     |   |   |   |   |   |   |   |
| data\_loader\.py   | Includes the methods needed to load images as batches for training/testing                                  | TBD                                                                      |   |   |   |   |   |   |   |

# How to Run
- Install requirements/CONDA env
- run the dataset download
- Run the training script
- Use inference.py to predict choosen images captions


# Tech Description
There are important implemented functionalilties in this project to be explained:

- In the `data_loader.py`, the function `get_loader()` works on creating a data loader for the training/test set files to generate batches consisting of images and the corresponding captions. 
- In the `model.py`, the `EncoderCNN` involves transfer learning from the pre-trained ResNet50 neural netowrk to achive excellent features extraction. As usual, the last fully connected layer was removed and a new layer was added instead to be trained on the COCO images and generate the desired word embbiding vector.
- The `DecoderRNN` architcture involves using the LSTM units to generate the captions. This netork takes the generated word embbiding vector from the `EncoderCNN` as input and predict the captions accordingly.

## 🌐 Socials:
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/https://www.linkedin.com/in/zaid-ghazal/) 

### ✍️ Random Dev Quote
![](https://quotes-github-readme.vercel.app/api?type=vetical&theme=tokyonight)

---
[![](https://visitcount.itsvg.in/api?id=ZaidGhazal&icon=0&color=0)](https://visitcount.itsvg.in)

<!-- Proudly created with GPRM ( https://gprm.itsvg.in ) -->
