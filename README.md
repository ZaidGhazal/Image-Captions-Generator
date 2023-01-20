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

# Running the App

There are 2 options to use the images cations generator:

**Option 1:** Use the deployed app version in the link. Note that the deployed version supports genarating captions for the uploaded image(s) and download the results as CSV file. These captions are generated using a pre-trained model. Training new models is not supported on this version.

**Option 2:** Clone/Download this repo files locally and run the app using the execuatable files. The one for macOS/Linux users is `run_macos.command`, or `run_windows.bat` for Windows. Using the local version, training a new model is available besids the captions generations as mentioned in option 1.

Bellow provided the insturctions to run the app using both options.

## Option 1: Using the Deployed App

The deployed app can be accessed through the link.


## Option 2: Using the App Locally
Alternatively, the repo files can be cloned/downloeded, and the app can be run using either files `run_macos.command` for macOS users, or `run_windows.bat` for Windows users. Images caption generation is available as in the deployed app. The extra features is the ability to train new model. This can be done by following these steps:

1- First, download the [COCO dataset]("https://cocodataset.org/#download"). Download only the files marked in the bellow picture:


2- Download/Clone the repo files locally.


3- Create a new folder called `cocoapi` inside the downloaded/cloned repo local directory. Inside the created folder, also the `images` and `annotations` folders must be created.


4- After the COCO dataset files download is done, move the `train2017` and `test2017` folders into the created `images` folder (`cocoapi/images`). Also move the content of the downloaded annotations folder (for the train and test sets) into the created annotations folder (`cocoapi/annotations`).


5- Run the suitable execuatble file (`run_macos.command` OR `run_windows.bat`) to start the app.


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
