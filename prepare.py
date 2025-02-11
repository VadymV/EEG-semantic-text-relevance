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

from src.releegance.data_operations.preparator import DataPreparator
from src.releegance.data_operations.preprocessor import DataPreprocessor
from src.releegance.misc.utils import set_logging, set_seed, create_args


def run():
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

    # Data preparation:
    data_preparator = DataPreparator(
        data_dir=data_preprocessor.cleaned_data_dir)

    if args.data_type == 'benchmark':
        data_preparator.prepare_data_for_benchmark()
    else:
        data_preparator.save_cleaned_data()

    data_preprocessor.remove_filter_folder()
    data_preprocessor.remove_epoched_folder()
    data_preprocessor.remove_cleaned_folder()


if __name__ == '__main__':
    run()
