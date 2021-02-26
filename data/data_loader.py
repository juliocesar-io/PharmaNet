import os
import re
import sys
import json
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
import os.path as osp
from skimage import io
from skimage.transform import resize

from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import pdb

class SMILESDataset(Dataset):
    """Description."""

    def __init__(self, data, char_size, smiles_dict, embed_size, neighbours, padding, test = False):
        """
        Args:
            smiles_dir (string): Directory with all the smiles.
        """
        self.smiles = data['Smiles']
        self.test = test
        if test:
            self.id = data['CHEMBL_ID']
        else:    
            self.targets = data['Target']
            self.labels = np.array(data['Label'])
        self.smiles_dict = smiles_dict
        self.embed_size = embed_size
        self.neighbours = neighbours
        self.padding = padding
        self.char_size = char_size

    def smile_to_one_hot(self, Smile):
        embed = np.zeros([self.char_size, self.embed_size])
        for i,c in enumerate(Smile):
            embed[self.smiles_dict[c], i] = 1
    
        return embed
            
    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        smile_one = self.smile_to_one_hot(smile)
        if self.test:
            ids = self.id[idx]
            return smile_one, smile, ids
        else:
            target = self.targets[idx]
            label = self.labels[idx]
            return smile_one, smile, label
