"""
Loader for words.
"""

import os
import re
from collections import OrderedDict
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data_operations.misc import load_data


class DatasetWords(Dataset):
    """
    Provides the access to the data for the word relevance classification task.

    Args:
        dir_data: Path to the directory containing the prepared data:
        ``data_prepared_for_benchmark``.
    """

    def __init__(self, dir_data: str) -> None:
        """
        Initializes an object.

        Args:
            dir_data: Path to the directory containing the prepared data folder:
            ``data_prepared_for_benchmark``.
        """
        self.dir_data = dir_data
        self.documents = self.get_path_to_documents()
        self.data = self.read_data()
        self.class_weights = self._get_class_weights()
        self.participants, self.unique_participants = self.get_participants()

    def _get_class_weights(self) -> dict:
        """
        Calculates weights for each class.

        Returns:
            Weights for each class.
        """
        n_samples = self.data['text'].shape[0]
        n_classes = 2
        y = self.data['text']['semantic_relevance']
        weights = n_samples / (n_classes * np.bincount(y))
        return {0: weights[0], 1: weights[1]}

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.data['text'])

    def get_path_to_documents(self):
        """
        Gets the path to each document.

        Returns:
            A list where each element contains the path to a document.
        """
        documents = os.listdir(self.dir_data)
        documents = [sample for sample in documents]
        return documents

    def read_data(self):
        """
        Reads the data and returns word-level text data and eeg data.

        Returns:
            A dict containing word-level text data and eeg data.
            The dict contains the following keys: ``text``, ``eeg_tiny``,
            and ``eeg_small``.

            The ``text`` dict contains the word-level text data,
            where a single value corresponds to a single word.

            The ``eeg_tiny`` dict contains word-level eeg recordings,
            where each value contains recordings of a single word.
            The ``eeg_tiny`` represents the tiny version of the eeg data, where
            each word-level recording is represented by 7 values.

            The ``eeg_small`` dict contains word-level eeg recordings,
            where each value contains recordings of a single word.
            The ``eeg_small`` represents the small version of the eeg data, where
            each word-level recording is represented by 151 values.
        """
        eeg_data_tiny = []
        eeg_data_small = []
        text_data = []
        for idx in range(len(self.documents)):
            sample_path = os.path.join(self.dir_data, self.documents[idx])
            eeg, text = load_data(folder_path=sample_path,
                                  file_name=self.documents[idx])

            # 7 sections:
            eeg_data_groups_tiny = np.array_split(eeg, 7, axis=-1)
            eeg_data_mean_groups_tiny = [torch.mean(i, dim=-1) for i in
                                    eeg_data_groups_tiny]
            eeg_tiny = torch.stack(eeg_data_mean_groups_tiny, dim=-1)

            # 151 sections:
            eeg_data_groups_small = np.array_split(eeg, 151, axis=-1)
            eeg_data_mean_groups_small = [torch.mean(i, dim=-1) for i in
                                         eeg_data_groups_small]
            eeg_small = torch.stack(eeg_data_mean_groups_small, dim=-1)

            text.loc[:, 'user'] = re.search('(TRPB)[0-9]{3}',
                                            sample_path).group()
            eeg_data_tiny.append(eeg_tiny)
            eeg_data_small.append(eeg_small)
            text_data.append(text)

        word_data = pd.concat(text_data, axis=0)
        word_data = word_data.reset_index(drop=True)
        word_data['sentence_relevance'] = 0
        word_data.loc[word_data['topic'] == word_data['selected_topic'], 'sentence_relevance'] = 1
        eeg_signals_tiny = torch.vstack(eeg_data_tiny)
        eeg_signals_small = torch.vstack(eeg_data_small)

        data_dict = {'eeg_tiny': eeg_signals_tiny,
                     'eeg_small': eeg_signals_small,
                     'text': word_data}
        return data_dict

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, pd.DataFrame]:
        """
        Returns the item at index idx.

        Args:
            idx: Index of the item to be returned.

        Returns:
            The item at index idx.
        """
        eeg_tiny = self.data.get('eeg_tiny')[idx]
        eeg_small = self.data.get('eeg_small')[idx]
        text = self.data.get('text').iloc[[idx]]

        return eeg_tiny, eeg_small, text

    def get_participants(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the identifiers and names of the participants.

        Returns:
            Identifiers and names of the participants.
        """
        participants = self.data['text'].loc[:, 'user']
        codes, uniques = pd.factorize(participants)
        return codes, uniques

    def get_indices_for_reading_tasks(self, participant_indices: list) -> OrderedDict:
        """
        Returns the indices for each reading task.
        Args:
            participant_indices: indices of data belonging to one participant.

        Returns:
            An ordered dictionary containing the indices for each reading task
            a participant performed.
            The keys are the reading tasks and the values are the
            indices of the data belonging to that reading task.
            The indices are sorted in ascending order.
        """
        d = self.data['text'].iloc[participant_indices]
        if len(d['topic'].unique()) != 16 and len(d['user'].unique()) != 1:
            raise ValueError('16 documents should be present per participant.')

        blocks = d[['topic', 'selected_topic', 'sentence_relevance', 'event']].groupby(by=['event']).min().drop_duplicates().sort_index()
        blocks['Block'] = np.repeat([*range(1, 9)], 2).tolist()
        blocks = blocks[['topic', 'Block']].drop_duplicates()
        d['Block'] = d.apply(
            lambda row: blocks[blocks['topic'] == row['topic']]['Block'].item(), axis=1)

        block_test_indices = OrderedDict()
        blocks = d['Block'].unique()
        blocks.sort()
        for block in blocks:
            block_test_indices[block] = d[d['Block'] == block].index

        return block_test_indices


class CollatorWords(object):
    """
    Collects and combines the EEG recordings and the labels into a single
    batch. The EEG recordings are word-level recordings.
    The labels are the semantic relevance labels.

    This collator is to be used when performing word-level classification.

    """

    def __call__(self, batch):
        (eeg_data, _, text_data) = zip(*batch)

        eeg_batch = torch.stack(eeg_data)
        eeg_batch = eeg_batch.reshape(eeg_batch.shape[0], -1)

        words_data = pd.concat(text_data, axis=0)
        word_labels = words_data['semantic_relevance'].tolist()

        return eeg_batch.float(), torch.FloatTensor(word_labels)


class CollatorEEGNetWord(object):
    """
    Collects and combines the EEG recordings and the labels into a single
    batch. The EEG recordings are word-level recordings.
    The labels are the semantic relevance labels.

    This collator is to be used when performing word-level classification
    with EEGNet.
    """

    def __call__(self, batch):
        (_, eeg_data, text_data) = zip(*batch)

        eeg_batch = torch.stack(eeg_data)
        eeg_batch = eeg_batch.unsqueeze(dim=1)

        words_data = pd.concat(text_data, axis=0)
        word_labels = words_data['semantic_relevance'].tolist()

        return eeg_batch.float(), torch.FloatTensor(word_labels)


class CollatorLSTMWord(object):
    """
    Collects and combines the EEG recordings and the labels into a single
    batch. The EEG recordings are word-level recordings.
    The labels are the semantic relevance labels.

    This collator is to be used when performing word-level classification
    with LSTM.

    The sequence length, required by the LSTM model,
    is defined by the number of EEG recordings for a single word
    and is equal for all words.
    """

    def __call__(self, batch):
        (eeg_data, _, text_data) = zip(*batch)

        eeg_batch = torch.stack(eeg_data)
        eeg_batch = eeg_batch.permute(0, 2, 1)

        words_data = pd.concat(text_data, axis=0)
        word_labels = words_data['semantic_relevance'].tolist()

        return eeg_batch.float(), torch.FloatTensor(word_labels)


class CollatorTransformerWord(object):
    """
    Collects and combines the EEG recordings and the labels into a single
    batch. The EEG recordings are word-level recordings.
    The labels are the semantic relevance labels.

    This collator is to be used when performing word-level classification
    with Transformer.

    The sequence length, required by the Transformer model,
    is defined by the number of EEG recordings for a single word
    and is equal for all words.
    """

    def __call__(self, batch):
        (eeg_data, _, text_data) = zip(*batch)

        eeg_batch = torch.stack(eeg_data)
        eeg_batch = eeg_batch.permute(0, 2, 1)

        words_data = pd.concat(text_data, axis=0)
        word_labels = words_data['semantic_relevance'].tolist()
        mask = None

        return (eeg_batch.float(), mask), torch.FloatTensor(word_labels)
