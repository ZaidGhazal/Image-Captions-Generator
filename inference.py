import pickle
import sys
import os
from model import EncoderCNN, DecoderRNN
from data_loader import get_loader
from torchvision import transforms
import torch
from PIL import Image, ImageTk
import numpy as np
import yaml


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# read the embed_size and hidden_size from config.yml
file_directory = os.path.realpath(__file__).rsplit("inference.py",1)[0]
path_to_models = os.path.join(file_directory, "models")
path_to_config = os.path.join(path_to_models, "config.yaml")




class Inference:

    def __init__(self):
        
        # checking if the directory demo_folder 
        # exist or not.
        if not os.path.exists(path_to_models):
            # if the demo_folder directory is not present 
            # then create it.
            os.makedirs(path_to_models)

        with open(path_to_config, "r") as stream:
            try:
                net_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        hidden_size = net_config["hidden_size"]
        new_img_size = net_config["new_img_size"]

        self.hidden_size = hidden_size

        # Define a transform to pre-process the testing images.
        self.transform_test = transforms.Compose([ 
            transforms.Resize((new_img_size,new_img_size)),                          # smaller edge of image resized to 256
            transforms.ToTensor(),                           # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                (0.229, 0.224, 0.225))])


        # Load the vocab file
        vocab_file_path = os.path.join(path_to_models, "vocab.pkl")
        with open(vocab_file_path, "rb") as file:
            self.vocab = pickle.load(file)

        

        # Load the trained weights.
        encoder = torch.load(os.path.join(path_to_models, 'encoder.pkl'))
        decoder = torch.load(os.path.join(path_to_models, 'decoder.pkl'))
        encoder.eval()
        decoder.eval()
        # Move models to GPU if CUDA is available.
        encoder.to(device)
        decoder.to(device)

        self.encoder = encoder
        self.decoder = decoder

    def clean_sentence(self, output):
        sentence = ''
        for x in output:
            sentence = sentence + ' ' + self.vocab.idx2word[x]
            sentence = sentence.strip()
            sentence = sentence.replace("<start>", "")
            sentence = sentence.replace("<end>", "")
        return sentence

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
        image = self.transform_test(PIL_image)
        image = image.reshape(1, 3, 224, 224)
        # Move the model to GPU, if available.
        image = image.to(device)
        # Pass the image through the encoder.
        features = self.encoder(image).unsqueeze(1)
        # Pass the encoder output through the decoder.
        output = self.decoder.sample(features, self.hidden_size)    
        # clean captions from the <start> and 
        # <end> tokens and return the result.
        sentence = self.clean_sentence(output)
        print("----> The caption for the image is:")
        print(sentence)
        return PIL_image , sentence

def run_inference(imgs_path):
        inference = Inference()
        resukts_list = []
        for path in imgs_path:
            img, caption = inference.generate_caption(path)
            resukts_list.append((img, caption))
        return resukts_list