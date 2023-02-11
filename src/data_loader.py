"""This file contains the code for the data loader."""
import json
import os
import random

import nltk
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from vocabulary import Vocabulary


def get_loader(
    transform,
    vocab_file="./models/vocab.pkl",
    mode="train",
    batch_size=1,
    vocab_threshold=None,
    start_word="<start>",
    end_word="<end>",
    unk_word="<unk>",
    vocab_from_file=True,
    num_workers=0,
):
    """
    Returns the data loader.

    Parameters
    ----------
    transform : torchvision.transforms
        Image transform.
    vocab_file : str, optional
        Path for vocabulary wrapper, by default "./models/vocab.pkl"
    mode : str, optional
        One of 'train', 'test', by default "train"
    batch_size : int, optional
        Batch size, by default 1
    vocab_threshold : int, optional
        Minimum word count threshold, by default None
    start_word : str, optional
        Special word denoting sentence start, by default "<start>"
    end_word : str, optional
        Special word denoting sentence end, by default "<end>"
    unk_word : str, optional
        Special word denoting unknown words, by default "<unk>"
    vocab_from_file : bool, optional
        If True, load vocab wrapper from file, by default True
    num_workers : int, optional
        Number of subprocesses to use for data loading, by default 0

    Returns
    -------
    data_loader : torch.utils.data.DataLoader
        Data loader for custom coco dataset.
    """
    assert mode in ["train", "test"], "mode must be one of 'train' or 'test'."
    if vocab_from_file == False:
        assert (
            mode == "train"
        ), "To generate vocab from captions file, must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    if mode == "train":
        if vocab_from_file == True:
            assert os.path.exists(
                vocab_file
            ), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        img_folder = "./cocoapi/images/train2017/"
        annotations_file = "./cocoapi/annotations/captions_train2017.json"
    if mode == "test":
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(
            vocab_file
        ), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        img_folder = "./cocoapi/images/test2017/"
        annotations_file = "./cocoapi/annotations/image_info_test2017.json"

    # COCO caption dataset.
    dataset = CoCoDataset(
        transform=transform,
        mode=mode,
        batch_size=batch_size,
        vocab_threshold=vocab_threshold,
        vocab_file=vocab_file,
        start_word=start_word,
        end_word=end_word,
        unk_word=unk_word,
        annotations_file=annotations_file,
        vocab_from_file=vocab_from_file,
        img_folder=img_folder,
    )

    if mode == "train":
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        # data loader for COCO dataset.
        data_loader = data.DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=data.sampler.BatchSampler(
                sampler=initial_sampler, batch_size=dataset.batch_size, drop_last=False
            ),
        )
    else:
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=dataset.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    return data_loader


class CoCoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(
        self,
        transform,
        mode,
        batch_size,
        vocab_threshold,
        vocab_file,
        start_word,
        end_word,
        unk_word,
        annotations_file,
        vocab_from_file,
        img_folder,
    ):
        """
        Set the path for images, captions and vocabulary wrapper.

        Parameters
        ----------
        transform : torchvision.transforms
            Image transform.
        mode : str
            One of 'train', 'test'.
        batch_size : int
            Batch size.
        vocab_threshold : int
            Minimum word count threshold.
        vocab_file : str
            Path for vocabulary wrapper.
        start_word : str
            Special word denoting sentence start.
        end_word : str
            Special word denoting sentence end.
        unk_word : str
            Special word denoting unknown words.
        annotations_file : str
            Path for train annotation json file.
        vocab_from_file : bool
            If True, load vocab wrapper from file.
        img_folder : str
            Path for train images folder.
        """
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(
            vocab_threshold,
            vocab_file,
            start_word,
            end_word,
            unk_word,
            annotations_file,
            vocab_from_file,
        )
        self.img_folder = img_folder
        if self.mode == "train":
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print("Obtaining caption lengths...")
            all_tokens = [
                nltk.tokenize.word_tokenize(
                    str(self.coco.anns[self.ids[index]]["caption"]).lower()
                )
                for index in tqdm(np.arange(len(self.ids)))
            ]
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item["file_name"] for item in test_info["images"]]

    def __getitem__(self, index):
        """Returns one data pair (image and caption).

        Parameters
        ----------
        index : int
            Index in the dataset.

        Returns
        -------
        image : torch.Tensor
            Image.
        caption : torch.Tensor
            Caption.
        """
        # obtain image and caption if in training mode
        if self.mode == "train":
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]["caption"]
            img_id = self.coco.anns[ann_id]["image_id"]
            path = self.coco.loadImgs(img_id)[0]["file_name"]

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            image = self.transform(image)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            # return pre-processed image and caption tensors
            return image, caption

        # obtain image if in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and pre-process using transform
            pil_image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            orig_image = np.array(pil_image)
            image = self.transform(pil_image)

            # return original image and pre-processed image tensor
            return orig_image, image

    def get_train_indices(self):
        """
        Returns training indices that correspond to a caption length of self.batch_size in length.

        Returns
        -------
        indices : list
            List of training indices.
        """
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where(
            [
                self.caption_lengths[i] == sel_length
                for i in np.arange(len(self.caption_lengths))
            ]
        )[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        """Returns the total number of font files.

        Returns
        -------
        length : int
            Total number of font files.
        """
        if self.mode == "train":
            return len(self.ids)
        else:
            return len(self.paths)
