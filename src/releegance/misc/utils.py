"""
Miscellaneous utilities.
"""

import argparse
import copy
import logging
import os
import random
import shutil
from enum import Enum

import numpy as np
import torch
from torch.backends import cudnn


def create_folder(output_dir, with_checking=False):
    """
    Creates a folder.
    Args:
        output_dir: Output folder.
        with_checking: Whether to check if the folder exists.

    Returns:
        Output folder.
    """
    if with_checking and os.path.exists(output_dir):
        raise ValueError("The folder exists")
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir


def remove_folder(folder_path: str):
    """
    Removes a folder.
    Args:
        folder_path: A folder.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' has been removed.")
    else:
        print(f"Folder '{folder_path}' does not exist.")


def set_logging(log_dir: str, file_name: str):
    """
    Sets logging.
    Args:
        log_dir: Logging directory.
        file_name: File name where the logs will be stored.

    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/{file_name}.log"),
            logging.StreamHandler()
        ]
    )


def set_seed(seed):
    """
    Sets the seed.
    Args:
        seed: Seed.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


def create_args(seeds_args: bool = True,
                benchmark_args: bool = True,
                data_type_args: bool = False) -> argparse.ArgumentParser:
    """
    Creates an argument parser.
    Args:
        seeds_args: Whether to add the ``seeds`` argument.
        benchmark_args: Whether to add the ``benchmark`` argument.
        data_type_args: Whether to add the ``data_type`` argument.

    Returns:
        An argument parser.

    """
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--project_path',
                        type=str,
                        help='A path to the folder containing the EEG data '
                             'called "raw"')
    if seeds_args:
        parser.add_argument('--seeds',
                            type=int,
                            help='Number of seeds to use.',
                            default=10)
    if benchmark_args:
        parser.add_argument('--benchmark',
                            type=str,
                            default='w',
                            help='"w" or "s"')
    if data_type_args:
        parser.add_argument('--data_type',
                            type=str,
                            default='benchmark',
                            help='"benchmark" or "preprocessed"')

    return parser


class Relevance(Enum):
    """
    Relevance and irrelevance labels.
    """
    RELEVANT = 1
    IRRELEVANT = 0

def _perm_fun(x: np.ndarray, n_test: int, n_control: int):
    n = n_test + n_control
    idx_control = set(random.sample(range(n), n_control))
    idx_test = set(range(n)) - idx_control
    return x[list(idx_test)].mean() - x[list(idx_control)].mean()

def run_permutation_test(control, test):
    a = copy.deepcopy(control)
    b = copy.deepcopy(test)
    observed_difference = np.mean(b) - np.mean(a)
    perm_diffs = [
        _perm_fun(np.concatenate((a, b), axis=None), a.shape[0], b.shape[0]) for
        _ in range(10000)]
    p = np.mean([diff > observed_difference for diff in perm_diffs])
    print("P value: {:.4f}".format(p))

    return p
