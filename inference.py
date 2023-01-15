import sys
import os
from tkinter import filedialog
import tkinter
from model import EncoderCNN, DecoderRNN
from pycocotools.coco import COCO
from data_loader import get_loader
from torchvision import transforms
import torch
from PIL import Image, ImageTk
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a transform to pre-process the testing images.
transform_test = transforms.Compose([ 
    transforms.Resize((224,224)),                          # smaller edge of image resized to 256
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])
# # Create the data loader.
# # Check if the file ./vocab.pkl exists
# path="vocab.pkl"
# isExist = os.path.exists(path)

# if isExist:
#     data_loader = get_loader(transform=transform_test,    
#                             mode='test')
# else:
#     raise FileNotFoundError(f"The {path} file is not found. This file is created when the models were trained.")


class Inference:

    def __init__(self, models_saving_directory: str, 
    embed_size: int, hidden_size: int
    ):
        
        # The size of the vocabulary.
        vocab_size = len(data_loader.dataset.vocab)

        # # Initialize the encoder and decoder, and set each to inference mode.
        encoder = EncoderCNN(embed_size)
        encoder.eval()
        decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
        decoder.eval()

        # Load the trained weights.
        encoder.load_state_dict(torch.load(os.path.join(models_saving_directory, 'encoder.pkl')))
        decoder.load_state_dict(torch.load(os.path.join(models_saving_directory, 'decoder.pkl')))

        # Move models to GPU if CUDA is available.
        encoder.to(device)
        decoder.to(device)

        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def clean_sentence(output):
        sentence = ''
        for x in output:
            sentence = sentence + ' ' + data_loader.dataset.vocab.idx2word[x]
            sentence = sentence.strip()
            sentence = sentence.replace("<start>", "")
            sentence = sentence.replace("<end>", "")
        return sentence

    # def generate_caption(self, images_directory):
    #     """
    #     Generate a caption from the image.

    #     parameters:
    #     images_directory: 
    #         The directory of the image to be captioned.
    #     """
    #     files = os.listdir(images_directory)

    #     #Filtering only the files.
    #     images_files = [f for f in files if os.path.isfile(os.path.join(images_directory, f))] 


    #     # Read the image and convert to tensor
    #     for image_file in images_files:
    #     # Convert image to tensor and pre-process using transform
    #         PIL_image = Image.open(os.path.join(images_directory, image_file)).convert('RGB')
    #         orig_image = np.array(PIL_image)
    #         image = transform_test(PIL_image)
    #         image = image.reshape(1, 3, 224, 224)
    #         # Move the model to GPU, if available.
    #         image = image.to(device)
    #         # Pass the image through the encoder.
    #         features = self.encoder(image).unsqueeze(1)
    #         # Pass the encoder output through the decoder.
    #         output = self.decoder.sample(features)    
    #         # clean captions from the <start> and 
    #         # <end> tokens and return the result.
    #         sentence = self.clean_sentence(output)
    #         print("----> The caption for the image is:")
    #         print(sentence)
    #         break

    def generate_caption(self, images_path):
        """
        Generate a caption from the image.

        parameters:
        images_path: 
            The path of the image to be captioned.
        """
        # Read the image and convert to tensor
        # Convert image to tensor and pre-process using transform
        PIL_image = Image.open(images_path).convert('RGB')
        orig_image = np.array(PIL_image)
        image = transform_test(PIL_image)
        image = image.reshape(1, 3, 224, 224)
        # Move the model to GPU, if available.
        image = image.to(device)
        # Pass the image through the encoder.
        features = self.encoder(image).unsqueeze(1)
        # Pass the encoder output through the decoder.
        output = self.decoder.sample(features)    
        # clean captions from the <start> and 
        # <end> tokens and return the result.
        sentence = self.clean_sentence(output)
        print("----> The caption for the image is:")
        print(sentence)
        return PIL_image , sentence

