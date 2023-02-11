"""This file contains the Vocabulary class which is used to convert tokens to integers and vice-versa."""
import os.path
import pickle
from collections import Counter

import nltk
from pycocotools.coco import COCO


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(
        self,
        vocab_threshold,
        vocab_file="./models/vocab.pkl",
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>",
        annotations_file="../cocoapi/annotations/captions_train2017.json",
        vocab_from_file=False,
    ):
        """Initialize the vocabulary.

        Parameters
        ----------
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
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, "rb") as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print("Vocabulary successfully loaded from vocab.pkl file!")
        else:
            self.build_vocab()
            with open(self.vocab_file, "wb") as f:
                pickle.dump(self, f)

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary.

        Parameters
        ----------
        word : str
            The token to be added to the vocabulary.
        """
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold."""
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]["caption"])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if i % 100000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        """Converts a token to its integer representation.

        Parameters
        ----------
        word : str
            The token to be converted to an integer.

        Returns
        -------
        int
            The integer representation of the token.
        """
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        """Returns the length of the vocabulary.

        Returns
        -------
        int
            The length of the vocabulary.
        """
        return len(self.word2idx)
