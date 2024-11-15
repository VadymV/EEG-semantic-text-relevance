# Copyright 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preprocesses and prepares EEG data for benchmarking.
Run it with ``poetry run python prepare.py --project_path=path``
The project_path should point to the folder that contains the
``raw`` folder with the ``raw`` EEG data and the annotations.csv file.
Please download the data from here: ANONYMOUS.
After running this script, the following folders/files will be created in
the project_path:
    ``filtered``
    ``epoched``
    ``cleaned_data``
    ``data_prepared_for_benchmark``
    ``cleanedEEG.npy``
    ``metadataForCleanedEEG.csv`
    ``metadataForCleanedEEG.pkl``
"""

import logging

from src.data_operations.preparator import DataPreparator
from src.data_operations.preprocessor import DataPreprocessor, \
    load_evoked_response, plot_erp
from src.misc.utils import set_logging, set_seed, create_args


def run():
    plot = False  # Should the figures be produced?
    parser = create_args(seeds_args=False, benchmark_args=False)
    args = parser.parse_args()

    set_logging(args.project_path, file_name='logs_prepare')
    set_seed(1)
    logging.info('Args: %s', args)

    # Data pre-processing:
    data_preprocessor = DataPreprocessor(project_path=args.project_path)
    data_preprocessor.filter()
    data_preprocessor.create_epochs()
    data_preprocessor.clean()

    # Data preparation:
    data_preparator = DataPreparator(
        data_dir=data_preprocessor.cleaned_data_dir)
    data_preparator.save_cleaned_data()
    data_preparator.prepare_data_for_benchmark()

    if plot:
        epochs = load_evoked_response(
            dir_cleaned=data_preprocessor.cleaned_data_dir,
            annotations=data_preparator.annotations,
            average=False)
        for electrode in epochs[0].ch_names:
            epochs = load_evoked_response(
                dir_cleaned=data_preprocessor.cleaned_data_dir,
                annotations=data_preparator.annotations,
                average=False)
            plot_erp(work_dir=args.project_path,
                     epos=epochs,
                     title=f'{electrode} electrode.',
                     queries=['annotation == 1',
                              'annotation == 0'],
                     file_id=electrode,
                     ch_names=[electrode],
                     l=['Semantically relevant',
                        'Semantically irrelevant'])

        relevant = load_evoked_response(
            dir_cleaned=data_preprocessor.cleaned_data_dir,
            filter_flag='annotation == 1',
            annotations=data_preparator.annotations)
        relevant.plot_joint(picks='eeg', times=[0.3, 0.4, 0.6],
                            title=None,
                            show=False,
                            ts_args=dict(ylim=dict(eeg=[-4.5, 5]), gfp=True))

        irrelevant = load_evoked_response(
            dir_cleaned=data_preprocessor.cleaned_data_dir,
            filter_flag='annotation == 0',
            annotations=data_preparator.annotations)
        irrelevant.plot_joint(picks='eeg', times=[0.3, 0.4, 0.6],
                              title=None,
                              ts_args=dict(ylim=dict(eeg=[-4.5, 5]), gfp=True))


if __name__ == '__main__':
    run()
