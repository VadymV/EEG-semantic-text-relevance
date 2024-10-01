"""
Definition of all models.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from src.data_operations.loader_sentences import \
    CollatorLSTMSentence, CollatorEEGNetSentence, CollatorTransformerSentence, \
    CollatorSentence
from src.data_operations.loader_words import CollatorLSTMWord, \
    CollatorEEGNetWord, CollatorTransformerWord, CollatorWords
from src.models.eegnet import EEGNet
from src.models.lstm import LSTM
from src.models.uercm import UERCM


class Models:
    """
    A class for holding all models.

    Attributes:
        lstm: An LSTM model.
        eegnet: An EEGNet model.
        lda: A linear discriminant analysis model.
        lr: A logistic regression model.
        uercm: An UERCM model.
    """

    def __init__(self, class_weight: dict, lstm_input_dim: int,
                 transformer_sequence_length: int,
                 transformer_feature_dim: int):
        self.lstm = LSTM(input_dim=lstm_input_dim, hid_channels=32, num_classes=1)
        self.eegnet = EEGNet(chunk_size=151, num_electrodes=32, num_classes=1)
        self.lda = LinearDiscriminantAnalysis()
        self.lr = LogisticRegression()
        self.uercm = UERCM(max_len=transformer_sequence_length,
                           feat_dim=transformer_feature_dim)
        self.class_weight = class_weight
        self.lstm_input_dim = lstm_input_dim
        self.transformer_sequence_length = transformer_sequence_length
        self.transformer_feature_dim = transformer_feature_dim

    def reset_models(self):
        self.lstm = LSTM(input_dim=self.lstm_input_dim, hid_channels=32, num_classes=1)
        self.eegnet = EEGNet(chunk_size=151, num_electrodes=32, num_classes=1)
        self.lda = LinearDiscriminantAnalysis()
        self.lr = LogisticRegression()
        self.uercm = UERCM(max_len=self.transformer_sequence_length,
                           feat_dim=self.transformer_feature_dim)

    def get_all_models(self) -> list:
        return ["lstm", "eegnet", "lda", "lr", "uercm"]

    def get_model(self, model_name: str):
        if model_name == "lstm":
            return self.lstm
        elif model_name == "eegnet":
            return self.eegnet
        elif model_name == "lda":
            return self.lda
        elif model_name == "lr":
            return self.lr
        elif model_name == "uercm":
            return self.uercm

    def set_model(self, model_name: str, model):
        if model_name == "lstm":
            self.lstm = model
        elif model_name == "eegnet":
            self.eegnet = model
        elif model_name == "lda":
            self.lda = model
        elif model_name == "lr":
            self.lr = model
        elif model_name == "uercm":
            self.uercm = model



    def get_model_name(self, model: object) -> str:
        if isinstance(model, LSTM):
            return "lstm"
        elif isinstance(model, EEGNet):
            return "eegnet"
        elif isinstance(model, LinearDiscriminantAnalysis):
            return "lda"
        elif isinstance(model, LogisticRegression):
            return "lr"
        elif isinstance(model, UERCM):
            return "uercm"

    def get_collator(self, model_name: str, is_sentence: bool):
        if model_name == "lstm" and is_sentence:
            return CollatorLSTMSentence()
        elif model_name == "lstm" and not is_sentence:
            return CollatorLSTMWord()
        elif model_name == "eegnet" and is_sentence:
            return CollatorEEGNetSentence()
        elif model_name == "eegnet" and not is_sentence:
            return CollatorEEGNetWord()
        elif model_name == "uercm" and is_sentence:
            return CollatorTransformerSentence()
        elif model_name == "uercm" and not is_sentence:
            return CollatorTransformerWord()
        elif model_name == "lda" and is_sentence:
            return CollatorSentence()
        elif model_name == "lda" and not is_sentence:
            return CollatorWords()
        elif model_name == "lr" and is_sentence:
            return CollatorSentence()
        elif model_name == "lr" and not is_sentence:
            return CollatorWords()
