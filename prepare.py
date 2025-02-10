# Copyright 2025 Vadym Gryshchuk
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
Preprocesses and prepares EEG data.

Run it with ``poetry run python prepare.py --project_path=path
--data_type=selected_data_type``

The project_path should point to the folder that contains the
``raw`` folder and the annotations.csv file.
These data can be downloaded from here: https://osf.io/xh3g5/.

If this script is executed with ``data_type=benchmark``, the
folder ``data_prepared_for_benchmark`` will be created in the project_path.
If this script is executed with ``data_type=preprocessed``, the
folder ``cleaned_data`` (preprocessed) will be created in the project_path.
"""

import logging

from src.data_operations.preparator import DataPreparator
from src.data_operations.preprocessor import DataPreprocessor, \
    load_evoked_response, plot_erp
from src.misc.utils import set_logging, set_seed, create_args


def run():
    plot = False  # Should the ERP figures be produced? Only applicable for the benchmark data
    parser = create_args(seeds_args=False,
                         benchmark_args=False,
                         data_type_args=True)
    args = parser.parse_args()

    set_logging(args.project_path, file_name='logs_prepare')
    set_seed(1)
    logging.info('Args: %s', args)

    # Data pre-processing:
    data_preprocessor = DataPreprocessor(project_path=args.project_path)
    data_preprocessor.filter()
    data_preprocessor.create_epochs()
    data_preprocessor.clean()

    if args.data_type == 'benchmark':
        # Data preparation:
        data_preparator = DataPreparator(
            data_dir=data_preprocessor.cleaned_data_dir)
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
                                ts_args=dict(ylim=dict(eeg=[-4.5, 5]),
                                             gfp=True))

            irrelevant = load_evoked_response(
                dir_cleaned=data_preprocessor.cleaned_data_dir,
                filter_flag='annotation == 0',
                annotations=data_preparator.annotations)
            irrelevant.plot_joint(picks='eeg', times=[0.3, 0.4, 0.6],
                                  title=None,
                                  ts_args=dict(ylim=dict(eeg=[-4.5, 5]),
                                               gfp=True))

        data_preprocessor.remove_filter_folder()
        data_preprocessor.remove_epoched_folder()
        data_preprocessor.remove_cleaned_folder()
    else:
        data_preprocessor.remove_filter_folder()
        data_preprocessor.remove_epoched_folder()


if __name__ == '__main__':
    run()
