"""
EEG pre-processing: filtering, creation of epochs, and cleaning.
"""
import copy
import csv
import logging
import os
from os import path, listdir
from os.path import join

import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne.epochs import EpochsFIF
from nltk.stem.snowball import PorterStemmer

from src.data_operations.preparator import CHANNELS
from src.misc import utils


def get_protocol_data(data_dir: str) -> dict:
    """
    Reading the protocol data.

    Args:
        data_dir: A folder containing the protocol data.

    Returns:
        A dictionary of the protocol data.
    """
    with open(path.join(data_dir, 'mindir-marker-protocol5.csv'),
              mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=';')
        next(reader, None)
        data = {str(int(rows[0][1:])): rows[1] for rows in reader}
        return data


class DataPreprocessor:
    """
    EEG pre-processing.
    """

    def __init__(self, project_path: str):
        """
        Initializes an object.

        Args:
            project_path: A path to the folder containing the EEG data.
        """
        self.raw_data_dir = path.join(project_path, 'raw/')

        self.filtered_data_dir = path.join(project_path, 'filtered/')
        utils.create_folder(self.filtered_data_dir)

        self.epoched_data_dir = path.join(project_path, 'epoched/')
        utils.create_folder(self.epoched_data_dir)

        self.cleaned_data_dir = path.join(project_path, 'cleaned/')
        utils.create_folder(self.cleaned_data_dir)

        self.protocol_data = get_protocol_data(self.raw_data_dir)
        self.annotations = pd.read_csv(os.path.join(project_path,
                                                    'annotations.csv'))

    def _filter(self, file_name: str):
        """
        Filters the EEG data by applying a band-pass filter (0.25-35Hz)
        and saves it into the folder "filtered".

        Args:
            file_name: A name of the file.

        """
        logging.info('\nFiltering%s', file_name)
        logging.info('Output will be saved into %s', self.filtered_data_dir)

        raw = mne.io.read_raw_brainvision(
            path.join(self.raw_data_dir, f'{file_name}/{file_name}.vhdr'),
            eog=('HEOG', 'VEOG'),
            preload=True)
        raw.set_montage('standard_1020')
        events, _ = mne.events_from_annotations(raw)

        # Pruned markers have different frequency:
        events[:, 0] = events[:, 0] * 4

        mne.write_events(
            path.join(self.filtered_data_dir, f'{file_name}-eve.fif'), events,
            overwrite=True)

        raw.filter(0.25, 35., fir_design='firwin', l_trans_bandwidth=0.1,
                   n_jobs=4)
        raw.save(path.join(self.filtered_data_dir, f'{file_name}-raw.fif'),
                 overwrite=True)

        logging.info('Finished filtering %s', file_name)

    def _map_words_events(self, participants_id: str):
        """
        Maps the words to events.

        Args:
            participants_id: Participant's ID.

        Returns:
            Mapped words to events.
        """

        events = mne.read_events(
            path.join(self.filtered_data_dir, f'{participants_id}-eve.fif'))

        # READ EPRIME
        f = open(path.join(self.raw_data_dir,
                           f'{participants_id}/{participants_id}.txt'), 'r',
                 encoding='UTF-8')
        experiment = None
        curr_word = {}
        words = []
        for line in f:
            line = line.translate(str.maketrans(dict.fromkeys('\0'))).strip()
            if 'Procedure: ShowWordProcXP1' in line:
                experiment = 1
            elif experiment and 'WhichWord:' in line:
                curr_word['word'] = line[11:]
            elif experiment and 'RType:' in line[:6]:
                curr_word['type'] = int(line[7:])
            elif experiment and 'X_TOPICREL:' in line:
                curr_word['topicrel'] = line[12:]
            elif experiment and 'X_TOPIC1:' in line:
                curr_word['topic1'] = line[10:]
            elif experiment and 'X_TOPIC2:' in line:
                curr_word['topic2'] = line[10:]
            elif experiment and 'isrelsent:' in line:
                curr_word['isrelsent'] = bool(int(line[11:]))
            elif experiment and 'sentencenumber:' in line:
                curr_word['sentencenumber'] = int(
                    line[16:]) if '???' not in line else -1
            if 'LogFrame End' in line:
                if experiment:
                    words.append(curr_word)
                experiment = None
                curr_word = {}

        # TRANSLATE MARKERS
        labels = [self.protocol_data[str(x)] for x in
                  events[2:, 2].tolist()]  # Skip first 2 events
        timeslabels = list(zip(events[:, 0].tolist(), labels))

        # MATCH STIMULI WITH WORDS
        stimulus_def = False
        trial_info = False
        experiment = None
        task = None
        block_count = 0
        tasks = [None] * 32
        sentence_counts = [-1] * 32
        stream_on = False
        si = 0
        global_word_i = 0
        cur_isrelsent = None
        sen_wid = 0
        cur_topicrel = ''
        doc_wid = [0, 0]
        doc_start_t = 0
        quest_def = False
        quest_show = False
        knowq = False
        interestq = False

        types = {
            1: 'visrel',
            2: 'semrel',
            3: 'funcword',
            4: 'semirr',
            5: 'visirr',
            6: 'semanti',
            7: 'id',
            8: 'unknown'
        }

        newevents = []
        w2edtype = [('event', 'int'),
                    ('word', 'U40'),
                    ('type', 'U40'),
                    ('topicrel', 'U40'),
                    ('topic', 'U40'),
                    ('isrelsent', 'bool'),
                    ('sentencenumber', 'int'),
                    ('sen_wid', 'int'),
                    ('doc_wid', 'int'),
                    ('know', 'int'),
                    ('interest', 'int')]
        words2events = np.empty(0, dtype=w2edtype)
        for t, line in timeslabels:

            if 'StimulusDefBegin' in line:
                stimulus_def = True
            if 'StimulusDefEnd' in line:
                stimulus_def = False
                continue
            if stimulus_def:
                continue

            if line == 'BlockStart':
                logging.info(
                    '------------------ START BLOCK %s ------------------',
                    block_count)
                si = 0
            elif line == 'BlockEnd':
                if experiment:
                    sentence_counts[block_count] = si
                block_count += 1
                logging.info(line)

            elif line == 'TrialInfoBegin':
                si += 1
                trial_info = True
            elif line == 'TrialInfoEnd':
                trial_info = False

            elif 'TrialTopic' in line:
                continue

            elif trial_info:
                if line == 'Experiment_0':
                    logging.info('Experiment 0, skipping')
                    experiment = 0
                elif line == 'Experiment_1':
                    logging.info(line)
                    experiment = 1
                elif experiment and 'Task_' in line:
                    task = line
                    tasks[block_count] = line
                if experiment:
                    logging.info(line)

            if experiment:
                if 'Streamofwordsbegins' in line:
                    stream_on = True
                elif 'Streamofwordsends' in line:
                    stream_on = False
                if stream_on and 'StimulusOn' in line:
                    word = words[global_word_i]
                    if cur_isrelsent != word['isrelsent']:
                        cur_isrelsent = word['isrelsent']
                        sen_wid = 0
                    if word['topicrel'] not in cur_topicrel:
                        cur_topicrel = word['topicrel']
                        doc_wid = [0, 0]

                        doc_start_t = t
                    typ = word['type'] + 8 if task == 'Task_CountRel' else word[
                        'type']
                    typ = typ + 16 if word['isrelsent'] else typ

                    # Figure out topic
                    t_n = 2 if word['topic1'] == word['topicrel'] else 1
                    word['topic'] = word['topicrel'] if word['isrelsent'] else \
                        word[f'topic{t_n:d}']

                    logging.info(
                        '%s %s', line.ljust(32), word['word'].ljust(15)
                    )
                    newevents.append([t, 1, typ])
                    words2events = np.append(words2events, np.array((
                        t,
                        word['word'],
                        types[word['type']],
                        word['topicrel'],
                        word['topic'],
                        word['isrelsent'],
                        word['sentencenumber'],
                        sen_wid,
                        doc_wid[cur_isrelsent],
                        -1,
                        -1
                    ), dtype=w2edtype))

                    global_word_i += 1
                    if word['type'] != 8:
                        doc_wid[cur_isrelsent] += 1
                        sen_wid += 1

                # QUESTIONNAIRE
                if 'QuestionnaireItemDEFStarts' in line:
                    quest_def = True
                    continue
                if 'QuestionnaireItemDEFEnds' in line:
                    quest_def = False
                if quest_def:
                    if 'QuestionnaireItem_HowMuchDoYouKnowAbout' in line:
                        knowq = True
                    if 'QuestionnaireItem_HowInterestingDoYouFind' in line:
                        interestq = True
                    logging.info(line)

                if 'QuestionnaireItem_IsShown' in line:
                    quest_show = True
                if 'QuestionnaireItem_IsUnshown' in line:
                    quest_show = False
                if quest_show and 'QuestionnaireResponse' in line:
                    logging.info(doc_start_t)
                    msk = words2events['event'] >= doc_start_t
                    if knowq:
                        know = int(line[22:])
                        logging.info('KNOW %d', know)
                        if np.sum(words2events['know'][msk]) < 0:  # REL topic
                            words2events['know'][
                                (msk & words2events['isrelsent'])] = know
                        else:  # IRR topic
                            words2events['know'][
                                (msk & ~(words2events['isrelsent']))] = know
                        knowq = False
                    if interestq:
                        interest = int(line[22:])
                        logging.info('INTEREST %d', interest)
                        if np.sum(
                                words2events['interest'][msk]) < 0:  # REL topic
                            words2events['interest'][
                                (msk & words2events['isrelsent'])] = interest
                        else:  # IRR topic
                            words2events['interest'][
                                (msk & ~(words2events['isrelsent']))] = interest
                        interestq = False

        logging.info('------------------ SUMMARY ------------------')
        for t in tasks:
            logging.info(t)
        for c in sentence_counts:
            if c > -1:
                logging.info(c)

        return words2events, newevents

    def _create_epochs(self, file_name: str):
        """
        Creates epochs from the filtered data and saves
        them into the folder "epoched".
        Args:
            file_name: A string with the name of the file.

        """
        participant = file_name[:7]
        logging.info('\nEpoching %s', participant)

        raw = mne.io.Raw(path.join(self.filtered_data_dir, file_name),
                         preload=True)

        words2events, newevents = self._map_words_events(participant)

        words2events_df = pd.DataFrame(words2events)
        newevents = np.array(newevents)

        ev = {
            'jread/irrtopic/visrel': 1,
            'jread/irrtopic/semrel': 2,
            'jread/irrtopic/funcword': 3,
            'jread/irrtopic/semirr': 4,
            'jread/irrtopic/visirr': 5,
            'jread/irrtopic/semanti': 6,
            'jread/irrtopic/id': 7,
            'jread/irrtopic/unknown': 8,
            'countrel/irrtopic/visrel': 9,
            'countrel/irrtopic/semrel': 10,
            'countrel/irrtopic/funcword': 11,
            'countrel/irrtopic/semirr': 12,
            'countrel/irrtopic/visirr': 13,
            'countrel/irrtopic/semanti': 14,
            'countrel/irrtopic/id': 15,
            'countrel/irrtopic/unknown': 16,
            'jread/reltopic/visrel': 17,
            'jread/reltopic/semrel': 18,
            'jread/reltopic/funcword': 19,
            'jread/reltopic/semirr': 20,
            'jread/reltopic/visirr': 21,
            'jread/reltopic/semanti': 22,
            'jread/reltopic/id': 23,
            'jread/reltopic/unknown': 24,
            'countrel/reltopic/visrel': 25,
            'countrel/reltopic/semrel': 26,
            'countrel/reltopic/funcword': 27,
            'countrel/reltopic/semirr': 28,
            'countrel/reltopic/visirr': 29,
            'countrel/reltopic/semanti': 30,
            'countrel/reltopic/id': 31,
            'countrel/reltopic/unknown': 32,
        }

        logging.info(raw.info)

        stemmer = PorterStemmer()
        words2events_df['stem'] = words2events_df['word'].apply(
            stemmer.stem)  # Add stems

        epochs = mne.Epochs(raw, events=newevents, event_id=ev,
                            baseline=None,
                            tmin=-.2,
                            tmax=1,
                            on_missing='warn',
                            preload=True)

        logging.info('Setting sentence number')
        words2events_df.loc[:, 'sen_group'] = 0
        sen_group = 1
        for index, row in words2events_df.iterrows():
            if row['sentencenumber'] == -1:
                sen_group += 1
            words2events_df.loc[index, 'sen_group'] = sen_group

        words2events_df.loc[:, 'sen_number'] = 0
        for rel_topic in words2events_df['topicrel'].unique():
            condition = (words2events_df['topicrel'] == rel_topic) & (
                    words2events_df['sentencenumber'] != -1)
            for topic in words2events_df[condition]['topic'].unique():
                condition = (words2events_df['topicrel'] == rel_topic) & (
                        words2events_df['topic'] == topic) & (
                                    words2events_df['sentencenumber'] != -1)
                indices = words2events_df.index[condition].tolist()

                words2events_df.loc[indices, 'sen_number'] = \
                    words2events_df.loc[indices,]['sen_group'].ne(
                        words2events_df.loc[indices,][
                            'sen_group'].shift()).cumsum()

        logging.info('Saving annotations')
        annotations = words2events_df.loc[
            words2events_df['sentencenumber'] != -1]
        annotations = annotations[['event', 'word', 'topic']]
        annotations.loc[:, 'participant'] = participant
        annotations.to_csv(
            path.join(self.epoched_data_dir,
                      f'{participant}-epo-annotations.csv'),
            index=False)
        epochs.metadata = words2events_df

        # Take only justread events
        epochs = epochs['jread']
        epochs = epochs['type != "unknown"']  # Drop unknowns

        logging.info(epochs.metadata)

        epochs.save(
            path.join(self.epoched_data_dir, f'{participant}-epo.fif'),
            overwrite=True)
        np.save(
            path.join(self.epoched_data_dir, f'{participant}-words-new.npy'),
            words2events)

        logging.info('Epoching for %s is finished', participant)

    def _clean(self, file_name: str, sampling_freq: float = None):
        """
        Cleans epochs and saves them into the folder "cleaned".
        Args:
            file_name: A file name.
            sampling_freq: A sampling frequency.
        """
        participant_id = file_name[:7]
        logging.info('Cleaning %s', participant_id)
        epochs_orig = mne.read_epochs(
            path.join(self.epoched_data_dir, file_name))
        logging.info(
            'Original sampling rate: %s Hz', epochs_orig.info['sfreq'])
        if sampling_freq is not None:
            epochs = epochs_orig.resample(sampling_freq)
            logging.info(
                'Sampling rate after resampling: %s', sampling_freq)
        else:
            epochs = epochs_orig
        epochs_final = epochs.copy()
        epochs = epochs.apply_baseline((-.2, 1))
        (mini, maxi) = epochs.time_as_index([-.2, 0.7], use_rounding=True)

        # Drop bad channels
        bad_ch_names = ['Fp1', 'Fp2', 'F7', 'F8', 'FC5', 'FC6', 'T7', 'T8',
                        'TP9', 'CP5', 'CP6', 'TP10', 'P7', 'P8',
                        'FT9',
                        'O1', 'Oz', 'O2', 'FT10']
        bad_ch_indices = mne.pick_channels(epochs.info['ch_names'],
                                           bad_ch_names)
        all_ch_indices = range(0, 32)

        good_ch_indices = list(set(all_ch_indices) - set(bad_ch_indices))
        good_ch_names = np.array(epochs.info['ch_names'])[good_ch_indices]
        logging.info('Using %d chs %s', len(good_ch_names),
                     ', '.join(good_ch_names))

        # use only specified time window, convert V -> µV
        epochs_all = epochs.get_data()[:, :, mini:maxi] * 1e6
        epochs_good = epochs_all[:, good_ch_indices, :]  # pick only good chs
        n_epos = len(epochs_all)

        max_abs_epos = np.max(np.absolute(epochs_good), axis=(1, 2))
        bad_epo = np.argsort(max_abs_epos)[
                  - np.floor(n_epos * 0.2).astype(int):]  # bad epo indices
        cutoff = max_abs_epos[
            bad_epo[0]]  # cutoff based on abs max of least bad epo of bad epos
        logging.info('Cutoff %.2f', cutoff)
        logging.info(
            '%.2f %% epos marked for dropping', (100 * len(bad_epo) / n_epos))
        logging.info('Cleaned epo %d', (n_epos - len(bad_epo)))

        bad_ch_indices = []
        for i in all_ch_indices:
            if ((np.sum(np.max(np.absolute(epochs_all[:, i, :]),
                               axis=1) > cutoff) / n_epos) > 0.2
                    or (np.sum(np.var(epochs_all[:, i, :],
                                      axis=1) < 0.5) / n_epos) > 0.2):
                bad_ch_indices.append(i)
        bad_ch_names = np.array(epochs.info['ch_names'])[bad_ch_indices]
        logging.info('Drop %d chs ', len(bad_ch_names))
        logging.info(', '.join(bad_ch_names))

        epochs_final.info['bads'] = list(bad_ch_names)
        epochs_final.drop(bad_epo)
        epochs_final = epochs_final.apply_baseline((-.2, 0))

        # Add metadata:
        metadata = epochs_final.metadata
        metadata = pd.merge(metadata, self.annotations, how='left', on=['topic', 'word'])
        metadata = metadata.drop(columns=['isrelsent',
                                          'type',
                                          'sentencenumber',
                                          'sen_wid',
                                          'doc_wid',
                                          'stem',
                                          'sen_group'],
                                 inplace=False,
                                 errors='raise')

        metadata = metadata.rename(
            columns={'topicrel': 'selected_topic',
                     'annotation': 'semantic_relevance',
                     'interest': 'interestingness',
                     'know': 'pre-knowledge',
                     'sen_number': 'sentence_number'},
            errors='raise')
        metadata['participant'] = file_name.split("-")[0]
        epochs_final.metadata = metadata

        epochs_final.save(path.join(self.cleaned_data_dir, file_name),
                          overwrite=True)
        logging.info('END %s', participant_id)

    def filter(self):
        """
        Filters the EEG data by applying a band-pass filter (0.25-35Hz).
        EEG data are saved into the folder "filtered".
        """
        dirs = [d for d in sorted(listdir(self.raw_data_dir)) if
                os.path.isdir(os.path.join(self.raw_data_dir, d))]
        for dir_name in dirs:
            self._filter(file_name=dir_name)

    def create_epochs(self):
        """
        Creates epochs and saves them into the folder "epoched".
        """
        files = [f for f in sorted(listdir(self.filtered_data_dir)) if
                 'raw.fif' in f]
        for f in files:
            self._create_epochs(file_name=f)

    def clean(self, sampling_freq=None):
        """
        Cleans the epochs by applying the baseline correction and
        dropping bad epochs. Data are saved into the folder "cleaned".
        Args:
            sampling_freq: Sampling frequency of the data.
        """
        files = [f for f in sorted(listdir(self.epoched_data_dir)) if
                 'epo' in f and 'csv' not in f]
        for f in files:
            self._clean(file_name=f, sampling_freq=sampling_freq)


def load_evoked_response(dir_cleaned: str,
                         annotations: pd.DataFrame,
                         filter_flag: str = None,
                         average: bool = True) -> mne.Evoked:
    """
    Loads epoched cleaned EEG data.
    Args:
        dir_cleaned: Path to cleaned EEG.
        filter_flag: Condition for selection epochs.
        average: Average epochs or not.
        annotations: Annotations.

    Returns:
        Epoch data.
    """
    epos = []
    for i, f in enumerate(
            [f for f in sorted(listdir(dir_cleaned)) if '.fif' in f]):
        epo = mne.read_epochs(join(dir_cleaned, f))
        epo.interpolate_bads(verbose=False)
        epo.pick_types(eeg=True)
        a = epo.metadata
        a['subject'] = 'S%02d' % (i + 1)
        a['Subject'] = f.split("-")[0]
        a = pd.merge(a, annotations, how='left', on=['topic', 'word'])
        epo.metadata = a
        if filter_flag is not None:
            epo = epo[filter_flag]
        epos.append(epo)

    if average:
        result =mne.concatenate_epochs(epos).average().crop(tmin=0, tmax=1)
    else:
        result =epos

    return result


def plot_erp(work_dir, epos, queries, file_id, ch_names=['Pz'], l=None, title=None, **kwargs):
    """
    Plots ERPs.
    """
    l = l or queries
    evos = {l[0]: [], l[1]: []}

    for idx, epo in enumerate(epos, 1):
        chi = [epo.info['ch_names'].index(ch_name) for ch_name in ch_names]
        if (np.isnan(epo[queries[0]].average(picks=chi).crop(tmin=0, tmax=1).data).any() or
                np.isnan(epo[queries[1]].average(picks=chi).crop(tmin=0, tmax=1).data).any()):
            continue
        evos[l[0]].append(epo[queries[0]].average(picks=chi).crop(tmin=0, tmax=1))
        evos[l[1]].append(epo[queries[1]].average(picks=chi).crop(tmin=0, tmax=1))


    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)
    _, axs = plt.subplots(nrows=1,
                          ncols=1,
                          )
    fig = mne.viz.plot_compare_evokeds(evos,
                                       picks=ch_names,
                                       title=title,
                                       vlines=[0.25, 0.95],
                                       colors={l[1]: 'xkcd:red',
                                               l[0]: 'xkcd:green',
                                               },
                                       linestyles=['solid', 'dotted'],
                                       ylim=dict(eeg=[-5, 8]),
                                       show=False,
                                       legend='upper center', show_sensors='upper left',
                                       axes=axs
                                       )
    for item in axs.get_xticklabels():
        item.set_rotation(45)
    plt.savefig("{}/plots/erp_{}.pdf".format(work_dir, file_id), format="pdf",
                bbox_inches="tight")
    plt.close()

# The following two functions are for preparing
# the data for statistical analysis. They are not used for benchmark.
def _fill_average_evoked_responses(epochs: EpochsFIF,
                                   relevance: int,
                                   data: dict,
                                   subject: str):
    """
    Calculates averaged evoked responses.
    """
    filter_exp = f'annotation == {relevance}'
    epochs = epochs[filter_exp].pick_channels(CHANNELS)
    t1 = copy.deepcopy(epochs)
    t2 = copy.deepcopy(epochs)
    t3 = copy.deepcopy(epochs)

    t1 = t1.crop(tmin=0.250, tmax=0.34999).get_data().squeeze()
    t2 = t2.crop(tmin=0.350, tmax=0.44999).get_data().squeeze()
    t3 = t3.crop(tmin=0.500, tmax=0.70000).get_data().squeeze()

    t1 = np.mean(np.mean(t1, axis=2),
                 axis=0)  # mean over time and then mean over words
    t2 = np.mean(np.mean(t2, axis=2),
                 axis=0)  # mean over time and then mean over words
    t3 = np.mean(np.mean(t3, axis=2),
                 axis=0)  # mean over time and then mean over words

    if len(CHANNELS) != len(epochs.ch_names):
        raise ValueError('Different number of channels.')
    data['subject'].extend([subject] * len(CHANNELS))
    data['channel'].extend(epochs.ch_names)
    data['t1'].extend(t1.tolist())
    data['t2'].extend(t2.tolist())
    data['t3'].extend(t3.tolist())
    data['relevance'].extend([relevance] * len(CHANNELS))

    return data


def create_average_evoked_response(dir_cleaned: str,
                                   annotations: pd.DataFrame) -> mne.Evoked:
    """
    Calculates averaged evoked responses.
    """
    epos = []
    data = {'subject': [], 'channel': [], 't1': [], 't2': [], 't3': [],
            'relevance': []}
    for i, f in enumerate(
            [f for f in sorted(listdir(dir_cleaned)) if '.fif' in f]):
        epo = mne.read_epochs(join(dir_cleaned, f))
        epo.interpolate_bads(verbose=False)
        epo.pick_types(eeg=True)
        a = epo.metadata
        subject = f.split("-")[0]
        a['subject'] = 'S%02d' % (i + 1)
        a['Subject'] = f.split("-")[0]
        a = pd.merge(a, annotations, how='left', on=['topic', 'word'])
        epo.metadata = a

        _fill_average_evoked_responses(epochs=epo,
                                       data=data,
                                       subject=subject,
                                       relevance=1,
                                       )

        _fill_average_evoked_responses(epochs=epo,
                                       data=data,
                                       subject=subject,
                                       relevance=0,
                                       )
    result = pd .DataFrame(data)
    result["relevance"] = result['relevance'].astype(str)
    result['flagged_channel'] = result[['relevance', 'channel']].agg('_'.join, axis=1)

    result1 = result.pivot(index=['subject'], columns='flagged_channel', values='t1').reset_index()
    result2 = result.pivot(index=['subject'], columns='flagged_channel', values='t2').reset_index()
    result3 = result.pivot(index=['subject'], columns='flagged_channel', values='t3').reset_index()

    result1['time'] = 't1'
    result2['time'] = 't2'
    result3['time'] = 't3'

    result = pd.concat([result1, result2, result3])
    result.pivot(index=["subject", "time"],
                 columns=[c for c in result.columns.values if
                          c not in ['subject', 'time']])

    result = result.sort_values(by=['subject', 'time']).set_index(['subject', 'time'])
    result.to_csv('out.csv', index=True)

