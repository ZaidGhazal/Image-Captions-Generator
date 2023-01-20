# Image-Captions-Generator
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Markdown](https://img.shields.io/badge/markdown-%23000000.svg?style=for-the-badge&logo=markdown&logoColor=white) ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)  ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

# Introduction
<p style="text-align:justify;">
Inspired by the <a href="https://arxiv.org/pdf/1411.4555.pdf">Show and Tell: A Neural Image Caption Generator</a> paper, a neural network model was built and trained to extract features from the images and generate text captions. To facilitate model usage, a <strong>Web App</strong> was created and designed to allow users to upload image(s) to be captioned by the model. The captions generated can be downloaded in a CSV file including each image name and the generated caption. 

Moreover, the app makes it possible for interested users to train new models (rather than the pre-trained one provided in this repo).
</p>

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

*Note: to use GPU for training, a pyTorch version supporting CUDA must be installed in the python environment created by running the execuatble file. see [PyTorch Download Page](https://pytorch.org) for the pip command*

# Network Architecture
<p style="text-align:justify;">
        The network architecture consists of two parts: The Encoder and Decoder.
        The Encoder consists of the pretrained ResNet-50 model layers (except the last fully connected linear layer) and an embedding linear layer used to get the extracted features vector and produce the embedding vector in a configurable size.
        ResNet-50 was chosen due to its power in classifying objects, which makes it excellent in extracting complex features through its convolutional network. 
        <br><br>
        The Decoder has three main parts: an embedding layer, an LSTM layer, and a fully connected output layer.
        The embedding layer is used to convert the input word indices to dense vectors of fixed size. The LSTM layer is a type of recurrent neural network that is used for processing sequential data. It has a configurable number of hidden states and a configurable number of layers. The fully connected output layer used to generate the final caption.
        <br><br>
        The deployed model has trained on the <a href="https://cocodataset.org/#home">COCO-2017</a> dataset. The training process was done on a machine equipped with a GeForce RTX 2080 Ti GPU. Training time was around 3 hours and 30 minutes. Training configurable parameters were as follows:
        <br><br>
        <ul>
        <li>Batch Size: 128</li>
        <li>Vocabulary Threshold: 8</li>
        <li>Embedding Size: 400</li>
        <li>Hidden States Size: 356</li>
        <li>Learning Rate: 0.001</li>
        <li>Number of Epochs: 2</li>
        </ul>
        </p>

-----------------------------------------------
## 🌐 Socials:
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/https://www.linkedin.com/in/zaid-ghazal/) 

### ✍️ Random Dev Quote
![](https://quotes-github-readme.vercel.app/api?type=vetical&theme=tokyonight)

---
[![](https://visitcount.itsvg.in/api?id=ZaidGhazal&icon=0&color=0)](https://visitcount.itsvg.in)

<!-- Proudly created with GPRM ( https://gprm.itsvg.in ) -->


# Scripts
| Script Name        | Description                                                                                                 | Required Arguments                                                       |   |   |   |   |   |   |   |
|--------------------|-------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|---|---|---|---|---|---|---|
| data\_download\.py | When run, the COCO training and validation \(test\) dataset files will be downloaded in the specified path  | Dataset Path \(str\): Path to save the downloaded images and annotations |   |   |   |   |   |   |   |
| train\.py          | When run, the neural networks will start training\. The final model will be saved in the specified path     | Model Path \(str\): Path to save the train neural network                |   |   |   |   |   |   |   |
| inference\.py      | TBD                                                                                                         | TBD                                                                      |   |   |   |   |   |   |   |
| model\.py          | Includes the encoder CNN and decoder RNN definition                                                         | None                                                                     |   |   |   |   |   |   |   |
| data\_loader\.py   | Includes the methods needed to load images as batches for training/testing                                  | TBD                                                                      |   |   |   |   |   |   |   |