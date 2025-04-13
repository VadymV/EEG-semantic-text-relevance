# Code to reproduce the benchmark results of our paper *An EEG dataset of word-level brain responses for semantic text relevance*

**The paper is accepted to ACM SIGIR 2025**

**Short description of the dataset**: Electroencephalography (EEG) enables non-invasive, real-time tracking of brain activity during language processing. Existing EEG datasets focus either on natural reading or controlled psycholinguistic settings. To address this gap, we introduce a novel dataset of 23,270 time-locked (0.7s) word-level EEG recordings, where participants read text that was either semantically relevant or irrelevant to self-selected topics. We benchmark two evaluation protocols — participant-independent and participant-dependent — on word and sentence relevance prediction tasks, using five models. Our dataset and code are openly available, supporting advancements in language relevance research, psycholinguistics, and brain-computer interface (BCI) systems for real-time semantic detection.

**The raw EEG data and the datasheet are available in our dataset repository: https://osf.io/xh3g5/.**

---

## Configure the environment
``poetry`` is used for dependency management (It is planned to replace *poetry* with *uv*).
See how to install ``poetry`` [here][2].

After ``poetry`` is installed, run ``poetry install`` in the folder where the ``README.md`` file is located.


## Getting the preprocessed and prepared data
There are 2 options:
1. Download the ``data_prepared_for_benchmark`` from [here][3] and extract the files.
This option is the fastest, as the data are already preprocessed and prepared for benchmarking.
2. Download the *raw* data and annotations.csv from [here][5] and 
run the script ``poetry run python prepare.py --project_path=path --data_type=benchmark``,
where ``project_path`` points to the folder that contains the *raw* data and annotations.csv.
After running this script, the folder ``data_prepared_for_benchmark``
with the prepared data for benchmarking will be created in the project_path.

## Run word relevance classification task

The ``project_path`` must point to the folder that contains the prepared data.
```py
poetry run python benchmark.py --project_path=path --benchmark=w
```

## Run sentence relevance classification task

The ``project_path`` must point to the folder that contains the prepared data.
```py
poetry run python benchmark.py --project_path=path --benchmark=s
```

## Generate prediction scores
Scores are saved to a ``logs_results.log`` file and outputted in a terminal window

```py
poetry run python generate_results.py --project_path=path
```

## Generate figures
```py
poetry run python generate_figures.py --project_path=path
```

## Benchmark results:

![Benchmark results](results.PNG)

## Details on classification models
We trained all our models without performing a hyperparameter optimisation and using in most cases the default parameters. 
This was intended as we wanted to provide the baseline benchmark results. 
If the default parameter was not used, we justify our selection of the value for that parameter. 
For example, the parameter **num\_classes** was set to 1 for the EEGNet, LSTM, and UERCM models, 
since a binary cross-entropy loss was used.

### Logistic regression and linear discriminant analysis
We have used the implementation of logistic regression provided by the scikit-learn library,
version 1.4.2 (https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html).
The default parameters were used and were not changed in all benchmark experiments.

### EEGNet
The implementation of the EEGNet model provided by the torcheeg library,
version 1.1.0 (https://torcheeg.readthedocs.io/en/v1.1.0/generated/torcheeg.models.EEGNet.html), is used.
The parameter **num_electrodes** was set to 32 and represents all the electrodes available in our data.

### LSTM
The implementation of the LSTM model provided by the PyTorch library, 
version 2.3.0 (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html), is used.

The parameter **input_dim** represents the dimensionality of the input EEG data.
This parameter can be set to 224 or 32:
- The value of 224 is used when performing the *sentence relevance* classification task. 
The value of 224 corresponds to the vector representation of each word-level EEG data in a sentence, 
where the word-level EEG data are represented as a matrix with 32 rows and 7 columns. 
We have 32 rows, as we have 32 electrodes. Each row represents an electrode, 
and each column represents the averaged EEG signal within the 0.1s range over a time span of 0.25 to 0.95 s.  
- The value of 32 is used when performing the *word relevance** classification task. 
The value of 32 corresponds to the number of electrodes.

The parameter **hid_channels** represents the number of features in the hidden state and was set to 32. 
We select 32 following simple reasoning: a single feature for each electrode. 

### UERCM
The implementation of the UERCM model is provided by the creators of the model (https://github.com/YeZiyi1998/UERCM).

The parameter **feat_dim** represents the dimensionality of the input EEG data. 
This parameter can be set to 224 or 32. The reason for selecting these values is the same as for the LSTM model.

The parameter **max_len** represents the input sequence length and can be set to 39 or 7:
- The value of 39 was used only during the *sentence relevance* classification task 
and corresponds to the longest sentence in all documents (i.e., the sentence that has the highest number of words). 
We select the value 39 to ensure that each sentence has the same length and 
can be put into a single *batch* containing many sentences. 
The sentences that have less than 39 words are padded with zeros. 
We ensure that padded data are not considered when training the model. 
- The value of 7 was used only during the *word relevance* classification task and 
corresponds to 7 values produced by averaging EEG recordings for a single word over a time span of 0.25 to 0.95 seconds.
Here, the first value represents the averaged EEG signal within the range of 0.25-0.35 s, 
the second value represents the averaged EEG signal within the range of 0.35-0.45 s, etc.

The parameter **d_model** represents the number of expected features in the encoder input and was set to 32. 
The reason for selecting 32 is the same as for the LSTM model setting the parameter **hid_channels** to 32. 

The parameter **num_layer** was set to 2, as used by [Pappagari et al.][4] for document classification 
using a small Transformer architecture.

---

If you use our dataset, please cite:
```
@unpublished{Gryshchuk2025_EEG-dataset,
   author = {Vadym Gryshchuk and Michiel Spapé and Maria Maistro and Christina Lioma and Tuukka Ruotsalo},
   title = {An EEG dataset of word-level brain responses for semantic text relevance},
   year = {2025},
   note = {Accepted to ACM SIGIR 2025}
}
```

  [1]: https://huggingface.co/datasets/Quoron/EEG-semantic-text-relevance
  [2]: https://python-poetry.org/docs/#installation
  [3]: https://drive.proton.me/urls/2TWQXJW2C4#9G2lbi7SuGFE
  [4]: https://arxiv.org/abs/1910.10781
  [5]: https://osf.io/xh3g5/

Issues:
- Currently, **src/** is the part of the installed package. After paper acceptance, it will not be a part of the package
(i.e., the pyproject.toml will contain ``packages = [{include = "releegance", from = "src"}]``).

References:

- R. Pappagari, P. Zelasko, J. Villalba, Y. Carmiel and N. Dehak, "Hierarchical Transformers for Long Document Classification," 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), Singapore, 2019, pp. 838-844, doi: 10.1109/ASRU46091.2019.9003958.
