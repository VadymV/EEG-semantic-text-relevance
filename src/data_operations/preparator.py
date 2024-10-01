"""
Prepares data for benchmarking.
"""

import logging
import os
from pathlib import Path
from typing import Tuple

import mne
import numpy as np
import pandas as pd

from src.misc import utils
from src.misc.utils import Relevance

CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
            'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2',
            'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'FT9', 'O1', 'Oz',
            'O2', 'FT10']


class DataPreparator:
    """
    Prepares data for benchmarking.

    Args:
        data_dir: Folder with cleaned EEG data.
        annotations: Ground truth annotations for words.
    """

    def __init__(self, data_dir: str):
        """
        Initializes an object.
        Args:
            data_dir: Folder with cleaned EEG data.
        """
        self.data_dir = data_dir
        self.annotations = pd.read_csv(os.path.join(Path(data_dir).parent,
                                                    'annotations.csv'))
        logging.info('Data dir is %s', self.data_dir)

    def save_cleaned_data(self):
        """
        Saves cleaned data as numpy array.
        Saves the metadata for the cleaned data as a csv file.
        """

        output_dir = os.path.join(Path(self.data_dir).parent)

        eeg_data_list = []
        metadata_list = []
        for filename in os.listdir(self.data_dir):
            f = os.path.join(self.data_dir, filename)
            if os.path.isfile(f) and f.endswith(
                    '.fif'):  # search fof .fif files
                epochs = mne.read_epochs(f).pick_channels(CHANNELS)
                eeg_data = epochs.crop(tmin=0.0,
                                       tmax=1.0).get_data(copy=False).squeeze()
                eeg_data = eeg_data * 1e6  # µV -> V (convert to voltages)
                metadata = epochs.metadata.reset_index(drop=True)

                # Assert that metadata and events are aligned:
                events_comparison = metadata.iloc[:,
                                        0].to_numpy() == epochs.events[:, 0]
                if np.unique(events_comparison).size != 1 and not np.unique(
                        events_comparison).item():
                    raise ValueError('Metadata and events mismatch!')

                metadata['interestingness'] = metadata['interestingness'].abs()
                metadata['pre-knowledge'] = metadata['pre-knowledge'].abs()
                eeg_data_list.append(eeg_data)
                metadata_list.append(metadata)

        eeg_data = np.concatenate(eeg_data_list)
        metadata = pd.concat(metadata_list).reset_index(drop=True)
        np.save(os.path.join(output_dir, 'cleanedEEG.npy'), eeg_data)
        metadata.to_csv(os.path.join(output_dir, 'metadataForCleanedEEG.csv'))
        metadata.to_pickle(os.path.join(output_dir, 'metadataForCleanedEEG.pkl'))

    def prepare_data_for_benchmark(self):
        """
        Prepares data for benchmarking.
        """

        prepared_dir = os.path.join(Path(self.data_dir).parent, 'data_prepared_for_benchmark')
        utils.create_folder(prepared_dir)

        logging.info('Prepared dir is %s', prepared_dir)

        for filename in os.listdir(self.data_dir):
            f = os.path.join(self.data_dir, filename)
            if os.path.isfile(f) and f.endswith(
                    '.fif'):  # search fof .fif files
                epochs = mne.read_epochs(f).pick_channels(CHANNELS)
                eeg_data = epochs.crop(tmin=0.25,
                                       tmax=0.95).get_data(copy=False).squeeze()
                eeg_data = eeg_data * 1e6  # µV -> V (convert to voltages)
                metadata = epochs.metadata.reset_index(drop=True)

                # Assert that metadata and events are aligned:
                events_comparison = metadata.iloc[:,
                                        0].to_numpy() == epochs.events[:, 0]
                if np.unique(events_comparison).size != 1 and not np.unique(
                        events_comparison).item():
                    raise ValueError('Metadata and events mismatch!')

                idx = filename.split('-epo.fif')[0]
                rel_documents, irr_documents = self._extract_documents(metadata,
                                                                       eeg_data,
                                                                       idx)
                self._write_documents_to_file(rel_documents, prepared_dir,
                                              Relevance.RELEVANT)
                self._write_documents_to_file(irr_documents, prepared_dir,
                                              Relevance.IRRELEVANT)

    def _extract_documents(self, metadata: pd.DataFrame, eeg_data: np.ndarray,
                           idx: str) -> Tuple[dict, dict]:
        """
        Extracts documents for a given participant.
        Args:
            metadata: Metadata dataframe.
            eeg_data: EEG data.
            idx: ID of a participant.

        Returns:
            A tuple of text documents and the corresponding EEG data for a
            given participant.
        """

        relevant_topics = metadata['selected_topic'].unique()
        selected_topics = metadata['topic'].unique()

        relevant_data = {'text': [], 'eeg': [], 'idx': []}
        irrelevant_data = {'text': [], 'eeg': [], 'idx': []}
        for rel_topic in relevant_topics:
            for selected_topic in selected_topics:
                document = metadata[
                    metadata['selected_topic'].isin([rel_topic]) & metadata[
                        'topic'].isin([selected_topic])]
                if document.size == 0:
                    continue
                if rel_topic == selected_topic:
                    relevant_data['text'].append(document)
                    relevant_data['eeg'].append(
                        eeg_data[document.index.tolist()])
                    relevant_data['idx'].append(idx)
                else:
                    irrelevant_data['text'].append(document)
                    irrelevant_data['eeg'].append(
                        eeg_data[document.index.tolist()])
                    irrelevant_data['idx'].append(idx)

        return relevant_data, irrelevant_data

    def _write_documents_to_file(self, documents: dict, output_folder: str,
                                 rel_flag: Relevance):
        """
        Writes documents to a file.
        Args:
            documents: Documents to write.
            output_folder: Output folder.
            rel_flag: Relevance or irrelevance.
        """
        for i in range(len(documents['text'])):
            documents['text'][i] = documents['text'][i].reset_index(drop=True)
            topic = documents['text'][i]['topic'].unique()[0]
            self._save(text_document=documents['text'][i],
                       eeg_document=documents['eeg'][i],
                       idx=documents['idx'][i],
                       topic=topic,
                       output_folder=output_folder,
                       rel_flag=rel_flag)

    def _save(self, text_document: pd.DataFrame,
              eeg_document: np.ndarray,
              idx: str,
              topic: str,
              output_folder: str,
              rel_flag: Relevance):
        """
        Saves text documents and EEG data.
        Args:
            text_document: Text document.
            eeg_document: EEG data.
            idx: ID of a participant.
            topic: Topic.
            output_folder: Output folder.
            rel_flag: Relevance or irrelevance.
        """
        start = 0
        end = len(text_document)
        folder_name = '-'.join([str(rel_flag.value), topic, idx])
        out_folder = os.path.join(output_folder, folder_name)
        utils.create_folder(out_folder, with_checking=True)

        eeg = eeg_document[start:end]
        text_document.to_pickle(
            os.path.join(out_folder, '.'.join([folder_name, 'pkl'])))
        np.save(os.path.join(out_folder, folder_name), eeg)
