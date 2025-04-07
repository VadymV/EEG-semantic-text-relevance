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
This script reproduces the benchmark results of our paper submitted to SIGIR 2025.
To run this script use ``poetry run python benchmark.py --project_path=path
--benchmark={w,s}`` where w stands for word relevance prediction and s stands
for sentence relevance prediction. The project_path should point to the
folder that contains the folder ``data_prepared_for_benchmark``.
If this folder does not exist, run the script prepare.py first or
download the data from here: https://drive.proton.me/urls/2TWQXJW2C4#9G2lbi7SuGFE
The script saves the results as pickle files.
"""
import argparse
import logging
import os
from typing import Union, Callable, List, Any, Literal, TypeVar

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.model_selection import LeaveOneGroupOut

from src.releegance.data_operations.loader_sentences import (
    MAX_SENTENCE_LENGTH,
    DatasetSentences,
)
from src.releegance.data_operations.loader_words import DatasetWords
from src.releegance.misc.utils import (
    create_args,
    set_logging,
    set_seed,
    calculate_ratio_pos_neg,
)
from src.releegance.models import Models
from src.releegance.trainer.trainer import train, test

_BATCH_SIZE = 30
T = TypeVar("T")


def fill_tracker(
    tracker: dict,
    test_idx: list,
    model: Union[torch.nn.Module, BaseEstimator],
    model_name: str,
    dataset: Union[DatasetSentences, DatasetWords],
    collator: Callable[[List[T]], Any],
    batch_size: int,
    participant: str,
    seed: int,
    strategy: Union[
        Literal["participant-dependent"], Literal["participant-independent"]
    ],
    reading_task: int,
    ratio_pos_neg: float,
):
    """
    Fills the tracker with the predictions, targets, and other info.
    Args:
        tracker: A dict for tracking info.
        test_idx: Test indices of the participant.
        model: A model.
        model_name: Model name.
        dataset: A dataset.
        collator: A collator function.
        batch_size: A batch size.
        participant: A participant.
        seed: Seed.
        strategy: Evaluation strategy.
        reading_task: A reading task as an integer.
        ratio_pos_neg: A negative sampling rate.

    Returns:
        A dict containing the information about: model, predictions, targets,
        user, seed, strategy, reading_task.
    """

    test_predictions, test_labels = test(
        model,
        dataset=dataset,
        test_idx=test_idx,
        collator=collator,
        batch_size=batch_size,
    )
    n = len(test_idx)
    tracker["model"].extend([model_name] * n)
    tracker["predictions"].extend(test_predictions.tolist())
    tracker["targets"].extend(test_labels.tolist())
    tracker["user"].extend([participant] * n)
    tracker["seed"].extend([seed] * n)
    tracker["strategy"].extend([strategy] * n)
    tracker["reading_task"].extend([reading_task] * n)
    tracker["ratio_pos_neg"].extend([ratio_pos_neg] * n)

    return tracker


def run(args: argparse.Namespace, seed: int):
    is_sentence = args.benchmark == "s"
    set_logging(args.project_path, file_name=f"logs_{args.benchmark}")
    set_seed(seed)
    logging.info("Args: %s", args)

    tracker = {
        "seed": [],
        "user": [],
        "model": [],
        "predictions": [],
        "targets": [],
        "strategy": [],
        "reading_task": [],
        "ratio_pos_neg": [],
    }

    data_dir = os.path.join(args.project_path, "data_prepared_for_benchmark")
    dataset = (
        DatasetSentences(dir_data=data_dir)
        if is_sentence
        else DatasetWords(dir_data=data_dir)
    )
    groups = dataset.participants
    logo = LeaveOneGroupOut()
    all_idx = np.arange(len(dataset))

    # Iterate over participants (1 * 15):
    for _, (outer_train_idx, test_idx) in enumerate(
        logo.split(all_idx, groups=groups), 1
    ):
        test_participant = np.unique(np.array(dataset.participants)[test_idx])
        if test_participant.size != 1:
            raise ValueError(
                f"Expected one test participant, got {test_participant.size}."
            )
        participant = dataset.unique_participants[test_participant[0]]
        logging.info("--- Test participant: %s ---", participant)

        inner_train_relative, val_relative = next(
            logo.split(np.arange(len(outer_train_idx)), groups=groups[outer_train_idx])
        )
        train_idx = outer_train_idx[inner_train_relative]
        val_idx = outer_train_idx[val_relative]

        # Get indices for reading trials (2 * 8):
        block_indices = dataset.get_indices_for_reading_tasks(test_idx)
        train_user_idx, val_user_idx, test_user_idx = [], [], []
        # Iterate over reading tasks (8) and create train/val/test indices:
        for test_block in block_indices:
            val_user_block = test_block + 1 if test_block < len(block_indices) else 1
            train_user_block = [
                b for b in block_indices if b not in [test_block] + [val_user_block]
            ]

            test_user_idx.append(block_indices[test_block])
            val_user_idx.append(block_indices[val_user_block])
            train_user_idx.append(
                [
                    i
                    for nested in [block_indices[i] for i in train_user_block]
                    for i in nested
                ]
            )

        models = Models(
            class_weight=dataset.class_weights,
            lstm_input_dim=224 if is_sentence else 32,
            transformer_sequence_length=MAX_SENTENCE_LENGTH if is_sentence else 7,
            transformer_feature_dim=224 if is_sentence else 32,
        )

        # Participant-independent strategy:
        logging.info("---- Participant-independent strategy ----")
        d = pd.concat(
            [dataset[i][2] for i in train_idx]
        )  # for the probability calibration
        # Iterate over models (5):
        for model_name in models.get_all_models():
            logging.info("---- %s ----", model_name)
            model = train(
                model=models.get_model(model_name),
                dataset=dataset,
                train_idx=train_idx,
                val_idx=val_idx,
                batch_size=_BATCH_SIZE,
                collator=models.get_collator(model_name, is_sentence),
            )
            models.set_model(model_name, model)
            # Iterate over reading tasks (8):
            for reading_task, test_subset in enumerate(test_user_idx):
                logging.info("---- Reading trial: %s ----", reading_task)
                logging.info("---- Test subset length: %s ----", len(test_subset))
                ratio_pos_neg = calculate_ratio_pos_neg(
                    positive_count=d["semantic_relevance"].sum(),
                    negative_count=len(d) - d["semantic_relevance"].sum(),
                    count=len(d),
                )
                tracker = fill_tracker(
                    tracker=tracker,
                    test_idx=test_subset,
                    model=model,
                    model_name=model_name,
                    collator=models.get_collator(model_name, is_sentence),
                    batch_size=_BATCH_SIZE,
                    strategy="participant-independent",
                    dataset=dataset,
                    participant=participant,
                    seed=seed,
                    reading_task=reading_task,
                    ratio_pos_neg=ratio_pos_neg,
                )

        # Participant-dependent strategy
        # Iterate over reading tasks (8):
        for reading_task in range(len(test_user_idx)):
            logging.info("---- Reading trial: %s ----", reading_task)
            logging.info(
                "---- Test subset length: %s ----", len(test_user_idx[reading_task])
            )
            # Iterate over models (5):
            for model_name in models.get_all_models():
                logging.info("---- %s ----", model_name)
                model = train(
                    model=models.get_model(model_name),
                    dataset=dataset,
                    train_idx=train_user_idx[reading_task],
                    val_idx=val_user_idx[reading_task],
                    batch_size=_BATCH_SIZE,
                    collator=models.get_collator(model_name, is_sentence),
                )
                d = pd.concat(
                    [dataset[i][2] for i in train_user_idx[reading_task]]
                )  # for the probability calibration
                ratio_pos_neg = calculate_ratio_pos_neg(
                    positive_count=d["semantic_relevance"].sum(),
                    negative_count=len(d) - d["semantic_relevance"].sum(),
                    count=len(d),
                )
                tracker = fill_tracker(
                    tracker=tracker,
                    test_idx=test_user_idx[reading_task],
                    model=model,
                    model_name=model_name,
                    collator=models.get_collator(model_name, is_sentence),
                    batch_size=_BATCH_SIZE,
                    strategy="participant-dependent",
                    dataset=dataset,
                    participant=participant,
                    seed=seed,
                    reading_task=reading_task,
                    ratio_pos_neg=ratio_pos_neg,
                )

        # Reset the models
        models.reset_models()

    logging.info("Saving the tracking data.")
    tracker = pd.DataFrame.from_dict(tracker)
    tracker.to_pickle(
        os.path.join(args.project_path, f"{args.benchmark}_relevance_seed{seed}.pkl")
    )


if __name__ == "__main__":
    parser = create_args()
    arguments = parser.parse_args()
    for selected_seed in range(1, arguments.seeds + 1):
        run(arguments, selected_seed)
