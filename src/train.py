"""This module contains the code for training the CNN-RNN model for image captioning."""
import math
import os
import sys
import time
from typing import Any

import nltk
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.utils.data as data
import yaml
from torch.optim import Adam
from torchvision import transforms

from data_loader import get_loader
from model import ImageEncoder, TextDecoder

nltk.download("punkt")


image_new_size = 224
# Define a transform to pre-process the training images.
transform_train = transforms.Compose(
    [
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(image_new_size),  # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize(
            (0.485, 0.456, 0.406),  # normalize image for pre-trained model
            (0.229, 0.224, 0.225),
        ),
    ]
)

saving_directory = "./models"

# checking if the directory demo_folder
# exist or not.
if not os.path.exists(saving_directory):
    # if the demo_folder directory is not present
    # then create it.
    os.makedirs(saving_directory)


class Model:
    """
    Model Class.

    This class contains methods for training a CNN-RNN model for image captioning.
    The class initializes with the following default parameters:
    saving_directory = "./models" # Path to save the trained models
    batch_size = 128, # batch size
    vocab_threshold = 8, # minimum word count threshold
    embed_size = 300, # dimensionality of image and word embeddings
    hidden_size = 128, # number of features in hidden state of the RNN decoder
    num_epochs = 2, # number of training epochs
    save_every = 1, # determines frequency of saving model weights
    print_every = 100, # determines window for printing average loss
    log_file = 'training_log.txt' # name of file with saved training loss and perplexity

    Methods:

    get_data_loader(batch_size, vocab_threshold): A static method that returns the data loader for the model
    train(learning_rate=0.001): trains the CNN-RNN model for image captioning with the specified learning rate

    Note: the models will be saved in the ./models directory
    """

    def __init__(
        self,
        batch_size: int = 128,  # batch size
        vocab_threshold: int = 8,  # minimum word count threshold
        embed_size: int = 300,  # dimensionality of image and word embeddings
        hidden_size: int = 128,  # number of features in hidden state of the RNN decoder
        num_epochs: int = 2,  # number of training epochs
        save_every: int = 1,  # determines frequency of saving model weights
        print_every: int = 100,  # determines window for printing average loss
        log_file: str = "training_log.txt",
        models_saving_directory: str = "./models",  # Path to save the trained models
    ):
        """
        This method initializes the Model class.

        Parameters
        ----------
        batch_size : int, optional
            The number of samples per batch, by default 128
        vocab_threshold : int, optional
            The minimum word count threshold, by default 8
        embed_size : int, optional
            The dimensionality of image and word embeddings, by default 300
        hidden_size : int, optional
            The number of features in hidden state of the RNN decoder, by default 128
        num_epochs : int, optional
            The number of training epochs, by default 2
        save_every : int, optional
            Determines frequency of saving model weights, by default 1
        print_every : int, optional
            Determines window for printing average loss, by default 100
        log_file : str, optional
            The name of file with saved training loss and perplexity, by default "training_log.txt"
        models_saving_directory : str, optional
            The path to save the trained models, by default "./models"
        """
        self.saving_directory = models_saving_directory
        self.batch_size = batch_size
        self.vocab_threshold = vocab_threshold
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.print_every = print_every
        self.log_file = log_file
        self.vocab_saving_path = os.path.join(self.saving_directory, "vocab.pkl")

    @staticmethod
    def get_data_loader(
        batch_size: int, vocab_threshold: int, vocab_saving_path: str
    ) -> Any:
        """
        This function is a static method that obtains the data loader for the model.

        Parameters
        ----------
        batch_size : int
            The number of samples per batch
        vocab_threshold : int
            The minimum word count threshold
        vocab_saving_path : str
            The path to save the vocabulary

        Returns
        -------
        Any
            The data loader for the model
        """
        # Obtain the data loader.
        data_loader = get_loader(
            transform=transform_train,
            mode="train",
            batch_size=batch_size,
            vocab_file=vocab_saving_path,
            vocab_threshold=vocab_threshold,
            vocab_from_file=False,
        )

        return data_loader

    def train(self, learning_rate: float = 0.001):
        """
        This method trains the CNN-RNN model for image captioning with the specified learning rate.

        Note: the models will be saved in the ./models directory

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate, by default 0.001
        """
        data_loader = self.get_data_loader(
            self.batch_size, self.vocab_threshold, self.vocab_saving_path
        )

        # The size of the vocabulary.
        vocab_size = len(data_loader.dataset.vocab)

        # Initialize the encoder and decoder.
        encoder = ImageEncoder(self.embed_size)
        decoder = TextDecoder(self.embed_size, self.hidden_size, vocab_size)

        # Move models to GPU if CUDA is available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("-------> Trianing on:", device)
        encoder.to(device)
        decoder.to(device)

        # Define the loss function.
        criterion = (
            nn.CrossEntropyLoss().cuda()
            if torch.cuda.is_available()
            else nn.CrossEntropyLoss()
        )

        # Specify the learnable parameters of the model.
        parameters = list(decoder.parameters()) + list(
            encoder.embedding_layer.parameters()
        )

        # Define the optimizer.
        optimizer = Adam(parameters, lr=learning_rate)

        # Set the total number of training steps per epoch.
        total_step = math.ceil(
            len(data_loader.dataset.caption_lengths)
            / data_loader.batch_sampler.batch_size
        )

        # Open the training log file.
        f = open(self.log_file, "w")

        old_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            for i_step in range(1, total_step + 1):
                if time.time() - old_time > 60:
                    old_time = time.time()

                # Randomly sample a caption length, and sample indices with that length.
                indices = data_loader.dataset.get_train_indices()
                # Create and assign a batch sampler to retrieve a batch with the sampled indices.
                new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
                data_loader.batch_sampler.sampler = new_sampler

                # Obtain the batch.
                images, captions = next(iter(data_loader))

                # Move batch of images and captions to GPU if CUDA is available.
                images = images.to(device)
                captions = captions.to(device)

                # Zero the gradients.
                decoder.zero_grad()
                encoder.zero_grad()

                # Pass the inputs through the CNN-RNN model.
                features = encoder(images)
                outputs = decoder(features, captions)

                # Calculate the batch loss.
                loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

                # Backward pass.
                loss.backward()

                # Update the parameters in the optimizer.
                optimizer.step()

                # Get training statistics.
                stats = (
                    "Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f"
                    % (
                        epoch,
                        self.num_epochs,
                        i_step,
                        total_step,
                        loss.item(),
                        np.exp(loss.item()),
                    )
                )

                # Print training statistics (on same line).
                print("\r" + stats, end="")
                sys.stdout.flush()

                # Print training statistics to file.
                f.write(stats + "\n")
                f.flush()

                # Print training statistics (on different line).
                if i_step % self.print_every == 0:
                    print("\r" + stats)

                # Save the weights.
            if epoch % self.save_every == 0:
                # if i_step % 10 == 0:
                torch.save(decoder, os.path.join(self.saving_directory, "decoder.pkl"))
                torch.save(encoder, os.path.join(self.saving_directory, "encoder.pkl"))

        # Close the training log file.
        f.close()


def run_train(
    batch_size: int = 128,
    vocab_threshold: int = 8,
    embed_size: int = 256,
    hidden_size: int = 512,
    learning_rate: float = 0.001,
    num_epochs: int = 2,
    status: Any = None,
):
    """
    This function runs the training process for the CNN-RNN model for image captioning.

    Parameters
    ----------
    batch_size : int, optional
        The batch size, by default 128
    vocab_threshold : int, optional
        The minimum word count threshold, by default 8
    embed_size : int, optional
        The dimensionality of the image and word embeddings, by default 256
    hidden_size : int, optional
        The number of features in the hidden state of the RNN decoder, by default 512
    learning_rate : float, optional
        The learning rate, by default 0.001
    num_epochs : int, optional
        The number of training epochs, by default 2
    status : Any, optional
        The status of the training process, by default None
    """
    model = Model(
        batch_size=batch_size,
        vocab_threshold=vocab_threshold,
        embed_size=embed_size,
        hidden_size=hidden_size,
        models_saving_directory=saving_directory,
        num_epochs=num_epochs,
    )
    try:
        if status:
            status.value = 200

        model.train(learning_rate)

        if status:
            status.value = 210
    except Exception as e:
        print(type(e).__name__, e)
        if type(e).__name__.strip() == "FileNotFoundError" and status:
            status.value = 510
            return
        if status:
            status.value = 500
        return

    # Save model configrations config.yaml
    config = {
        "embed_size": embed_size,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        # 'training_time_minutes': training_time,
        "img_size": image_new_size,
    }

    with open(os.path.join(saving_directory, "config.yaml"), "w") as f:
        yaml.dump(config, f)
