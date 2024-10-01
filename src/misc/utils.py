"""
Miscellaneous utilities.
"""

import argparse
import logging
import os
import random
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


def create_args(seeds_args: bool = True, benchmark_args: bool = True) -> argparse.ArgumentParser:
    """
    Creates an argument parser.
    Args:
        seeds_args: Whether to add the ``seeds`` argument.
        benchmark_args: Whether to add the ``benchmark`` argument.

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

    return parser


class Relevance(Enum):
    """
    Relevance and irrelevance labels.
    """
    RELEVANT = 1
    IRRELEVANT = 0
