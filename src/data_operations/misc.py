"""
Miscellaneous functions.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader


def get_data(loader: DataLoader,
             only_labels: bool = False) -> Tuple[Tensor, Tensor]:
    """
    Extracts data from the data loader.
    Args:
        only_labels: Whether to only return labels.
        loader: A data loader.

    Returns:
        A tuple of eeg data and labels.
    """
    eeg_data = []
    labels = []
    for _, (eeg, term_label) in enumerate(loader):
        eeg_data.append(eeg)
        labels.append(term_label)

    if not only_labels:
        eeg_data = torch.concat(eeg_data, dim=0)
    labels = torch.concat(labels, dim=0)

    return eeg_data, labels


def load_data(folder_path: str, file_name: str) -> Tuple[Tensor, pd.DataFrame]:
    """
    Loads data.
    Args:
        folder_path: A path to the folder containing all data.
        file_name: A file name to read.

    Returns:
        A tuple of eeg data and text data.
    """
    eeg_data = torch.from_numpy(
        np.load(os.path.join(folder_path, '.'.join([file_name, 'npy']))))
    text_data = pd.read_pickle(
        os.path.join(folder_path, '.'.join([file_name, 'pkl'])))

    return eeg_data, text_data
