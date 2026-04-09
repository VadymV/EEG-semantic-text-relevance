"""
Trainer.
"""

import logging
import tempfile
from typing import Any, Dict, List, Tuple, Union, Callable, TypeVar

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from torchmetrics import MetricCollection

from src.releegance.data_operations.loader_sentences import DatasetSentences
from src.releegance.data_operations.loader_words import DatasetWords
from src.releegance.data_operations.misc import get_data

_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader
T = TypeVar("T")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8 if torch.cuda.is_available() else 0
PIN_MEMORY = True if torch.cuda.is_available() else False
PERSISTENT_WORKERS = True if torch.cuda.is_available() else False

log = logging.getLogger("torcheeg")


def test(
    model: torch.nn.Module,
    dataset: Union[DatasetSentences, DatasetWords],
    test_idx: list,
    collator: Callable[[List[T]], Any],
    batch_size: int,
) -> ():
    """
    Tests a model.

    Args:
        model: Model.
        dataset: Dataset.
        test_idx: Test indices.
        collator: A collator function.
        batch_size: A batch size.

    Returns:
        Predictions and targets.
    """
    # Create data loader:
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        shuffle=False,
        collate_fn=collator,
    )

    logging.info("Testing model ...")
    if isinstance(model, torch.nn.Module):
        trainer = ClassifierTrainer(model, accelerator=DEVICE)
        test_predictions = torch.cat(trainer.predict(test_loader))
        _, test_labels = get_data(test_loader, only_labels=True)
    else:
        test_eeg, test_labels = get_data(test_loader)
        test_predictions = model.predict_proba(test_eeg)[:, 1]

    return test_predictions, test_labels


def train(
    model: torch.nn.Module,
    dataset: Union[DatasetSentences, DatasetWords],
    train_idx: list,
    val_idx: list,
    collator: Callable[[List[T]], Any],
    batch_size: int,
    lr: float = 1e-3,
    patience: int = 1,
    weight_decay: float = 1e-4,
) -> Tuple[torch.nn.Module, List[float], List[float]]:
    """
    Trains a model.

    Args:
        collator: A collator function.
        model: Model.
        dataset: Dataset.
        train_idx: Train indices.
        val_idx: Validation indices.
        batch_size: A batch size.
        lr: Learning rate.
        patience: Early stopping patience.
        weight_decay: L2 regularisation for the Adam optimizer.

    Returns:
        A tuple of (model, train_auc_history, val_auc_history).
    """

    # Create data loaders:
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        shuffle=False,
        collate_fn=collator,
    )

    if isinstance(model, torch.nn.Module):
        trainer = ClassifierTrainer(model, lr=lr, weight_decay=weight_decay, accelerator=DEVICE)
        train_auc_history, val_auc_history = trainer.fit(train_loader, val_loader, patience=patience)
        model = trainer.model
    else:
        train_eeg, train_labels = get_data(train_loader)
        model.fit(train_eeg, train_labels)
        train_auc_history, val_auc_history = [], []

    return model, train_auc_history, val_auc_history


def classification_metrics(metric_list: List[str], num_classes: int):
    # Copied from https://github.com/torcheeg/torcheeg/blob/v1.1.2/torcheeg/trainers/classifier.py
    allowed_metrics = [
        "precision",
        "recall",
        "f1score",
        "accuracy",
        "matthews",
        "auroc",
        "kappa",
    ]

    for metric in metric_list:
        if metric not in allowed_metrics:
            raise ValueError(
                f"{metric} is not allowed. Please choose 'precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc', 'kappa'."
            )
    metric_dict = {
        "accuracy": torchmetrics.Accuracy(task="binary"),
        "precision": torchmetrics.Precision(task="binary"),
        "recall": torchmetrics.Recall(task="binary"),
        "f1score": torchmetrics.F1Score(task="binary"),
        "matthews": torchmetrics.MatthewsCorrCoef(task="binary"),
        "auroc": torchmetrics.AUROC(task="binary"),
        "kappa": torchmetrics.CohenKappa(task="binary"),
    }
    metrics = [metric_dict[name] for name in metric_list]
    return MetricCollection(metrics)


class ClassifierTrainer(pl.LightningModule):
    # Copied from here https://github.com/torcheeg/torcheeg/blob/v1.1.2/torcheeg/trainers/classifier.py
    # and modified
    r"""
    A generic trainer class for EEG classification.

    .. code-block:: python

        trainer = ClassifierTrainer(model)
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

    Args:
        model (nn.Module): The classification model, and the dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
        num_classes (int, optional): The number of categories in the dataset. If :obj:`None`, the number of categories will be inferred from the attribute :obj:`num_classes` of the model. (defualt: :obj:`None`)
        lr (float): The learning rate. (default: :obj:`0.001`)
        weight_decay (float): The weight decay. (default: :obj:`0.0`)
        devices (int): The number of devices to use. (default: :obj:`1`)
        accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
        metrics (list of str): The metrics to use. Available options are: 'precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc', and 'kappa'. (default: :obj:`["accuracy"]`)

    .. automethod:: fit
    .. automethod:: test
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        devices: int = 1,
        accelerator: str = "cpu",
        metrics: List[str] = ["auroc"],
    ):

        super().__init__()
        self.model = model

        self.num_classes = 1
        self.lr = lr
        self.weight_decay = weight_decay

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.ce_fn = nn.BCEWithLogitsLoss()

        self.init_metrics(metrics, self.num_classes)

    def init_metrics(self, metrics: List[str], num_classes: int) -> None:
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_metrics = classification_metrics(metrics, num_classes)
        self.val_metrics = classification_metrics(metrics, num_classes)
        self.test_metrics = classification_metrics(metrics, num_classes)

        self.train_auc_history: List[float] = []
        self.val_auc_history: List[float] = []

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        patience: int = 1,
    ) -> None:
        r"""
        Args:
            train_loader (DataLoader): Iterable DataLoader for traversing the training data batch.
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch.
            max_epochs (int): Maximum number of epochs to train the model. (default: :obj:`100`)
            patience (int): Early stopping patience. (default: :obj:`1`)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            early_stopping = EarlyStopping(
                monitor="val_loss", mode="min", patience=patience, verbose=True
            )
            checkpoint = ModelCheckpoint(
                dirpath=tmpdir, monitor="val_loss", mode="min", save_top_k=1
            )
            trainer = pl.Trainer(
                devices=self.devices,
                accelerator=self.accelerator,
                max_epochs=max_epochs,
                num_sanity_val_steps=0,
                gradient_clip_val=1.0,
                logger=False,
                default_root_dir=tmpdir,
                callbacks=[early_stopping, checkpoint],
            )
            trainer.fit(self, train_loader, val_loader)
            if checkpoint.best_model_path:
                best_state = torch.load(
                    checkpoint.best_model_path, weights_only=True
                )["state_dict"]
                self.load_state_dict(best_state)
                logging.info(
                    "Loaded best checkpoint (val_loss=%.4f)",
                    checkpoint.best_model_score,
                )
        return self.train_auc_history, self.val_auc_history

    def test(self, test_loader: DataLoader, *args, **kwargs) -> _EVALUATE_OUTPUT:
        r"""
        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        """
        trainer = pl.Trainer(
            devices=self.devices,
            accelerator=self.accelerator,
            num_sanity_val_steps=0,
            logger=False,
            *args,
            **kwargs,
        )
        return trainer.test(self, test_loader)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        # log to prog_bar
        self.log(
            "train_loss",
            self.train_loss(loss),
            prog_bar=True,
            on_epoch=False,
            logger=False,
            on_step=True,
        )

        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(
                f"train_{self.metrics[i]}",
                metric_value(torch.sigmoid(y_hat), y),
                prog_bar=True,
                on_epoch=False,
                logger=False,
                on_step=True,
            )

        return loss

    def on_train_epoch_end(self) -> None:
        self.log(
            "train_loss",
            self.train_loss.compute(),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            logger=True,
        )
        for i, metric_value in enumerate(self.train_metrics.values()):
            value = metric_value.compute()
            self.log(
                f"train_{self.metrics[i]}",
                value,
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                logger=True,
            )
            if self.metrics[i] == "auroc":
                self.train_auc_history.append(value.item())

        # print the metrics
        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str += f"{key}: {value:.3f} "
        log.info(str + "\n")

        # reset the metrics
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.val_loss.update(loss)
        self.val_metrics.update(torch.sigmoid(y_hat), y)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log(
            "val_loss",
            self.val_loss.compute(),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            logger=True,
        )
        for i, metric_value in enumerate(self.val_metrics.values()):
            value = metric_value.compute()
            self.log(
                f"val_{self.metrics[i]}",
                value,
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                logger=True,
            )
            if self.metrics[i] == "auroc":
                self.val_auc_history.append(value.item())

        # print the metrics
        str = "\n[Val] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val_"):
                str += f"{key}: {value:.3f} "
        log.info(str + "\n")

        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.test_loss.update(loss)
        self.test_metrics.update(torch.sigmoid(y_hat), y)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log(
            "test_loss",
            self.test_loss.compute(),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            logger=True,
        )
        for i, metric_value in enumerate(self.test_metrics.values()):
            self.log(
                f"test_{self.metrics[i]}",
                metric_value.compute(),
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                logger=True,
            )

        # print the metrics
        str = "\n[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                str += f"{key}: {value:.3f} "
        log.info(str + "\n")

        self.test_loss.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = torch.optim.Adam(
            trainable_parameters, lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def predict_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def predict(self, test_loader: DataLoader, *args, **kwargs):
        trainer = pl.Trainer(
            devices=self.devices,
            accelerator=self.accelerator,
            num_sanity_val_steps=0,
            *args,
            **kwargs,
        )
        return trainer.predict(self, test_loader, return_predictions=True)
