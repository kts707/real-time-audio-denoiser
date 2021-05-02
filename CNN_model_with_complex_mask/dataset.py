import torch
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os

from utils import prepare_input_features

class AudioDataset(Dataset):
    """Noisy Audio Dataset"""

    def __init__(self, data_dir, data_type, transform=None):
        """
        Args:
            data_dir (string): Path to data directory of audio files.
            data_type (string): 'training', 'test' or 'validation'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        clean_data = os.path.join(self.data_dir, "clean")
        noisy_data = os.path.join(self.data_dir, "noisy")

        clean_files = os.listdir(os.path.join(clean_data,data_type))
        noisy_files = os.listdir(os.path.join(noisy_data,data_type))

        # check if the # of noisy files = # of clean files
        assert len(clean_files) == len(noisy_files)
        clean_files_full_path = [os.path.join(clean_data, data_type, filename) for filename in clean_files]
        noisy_files_full_path = [os.path.join(noisy_data, data_type, filename) for filename in noisy_files]

        print("Start processing",data_type,"data")
        for idx in range(len(clean_files)):

            clean_new_array = np.load(clean_files_full_path[idx])
            noisy_new_array = np.load(noisy_files_full_path[idx])

            # Do some of the preprocessing here
            noisy_new_array, clean_new_array = prepare_input_features(noisy_new_array, clean_new_array, 8, 129)
            if idx == 0:
                noisy_final = np.copy(noisy_new_array)
                clean_final = np.copy(clean_new_array)
            else:
                noisy_final = np.concatenate((noisy_final, noisy_new_array))
                clean_final = np.concatenate((clean_final, clean_new_array))

        self.noisy_data = torch.from_numpy(noisy_final).float()
        self.clean_data = torch.from_numpy(clean_final).float()

    def __len__(self):
        return self.noisy_data.shape[0]

    def __getitem__(self, index):
        target = self.clean_data[index]
        data_value = self.noisy_data[index]
        
        return (data_value, target)
