import os

import mne
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.releegance.data_operations.loader_words import DatasetWords
from src.releegance.data_operations.preparator import DataPreparator
from src.releegance.data_operations.preprocessor import DataPreprocessor, \
    load_evoked_response, plot_erp
from src.releegance.misc.utils import create_args, set_seed


def erps_electrodes(project_path):
    data_preprocessor = DataPreprocessor(project_path=project_path)
    data_preparator = DataPreparator(
        data_dir=data_preprocessor.cleaned_data_dir)
    epochs = load_evoked_response(
        dir_cleaned=data_preprocessor.cleaned_data_dir,
        annotations=data_preparator.annotations,
        average=False)
    for electrode in epochs[0].ch_names:
        epochs = load_evoked_response(
            dir_cleaned=data_preprocessor.cleaned_data_dir,
            annotations=data_preparator.annotations,
            average=False)
        plot_erp(work_dir=project_path,
                 epos=epochs,
                 title=f'{electrode} electrode.',
                 queries=['annotation == 1',
                          'annotation == 0'],
                 file_id=electrode,
                 ch_names=[electrode],
                 l=['Semantically relevant words',
                    'Semantically irrelevant words'])


def erps_word_level_sentence_level(project_path):
    data_preprocessor = DataPreprocessor(project_path=project_path)
    data_preparator = DataPreparator(
        data_dir=data_preprocessor.cleaned_data_dir)

    epochs = load_evoked_response(
        dir_cleaned=data_preprocessor.cleaned_data_dir,
        annotations=data_preparator.annotations,
        average=False)
    plot_erp(work_dir=project_path,
             epos=epochs,
             ch_names=['Pz'],
             title='Pz electrode',
             queries=['selected_topic == topic', 'selected_topic != topic'],
             file_id="semantic_relevance_sentences",
             l=['Semantically relevant sentences',
                'Semantically irrelevant sentences'])

    plot_erp(work_dir=project_path,
             epos=epochs,
             ch_names=['Pz'],
             title='Pz electrode',
             queries=['semantic_relevance == 1', 'semantic_relevance == 0'],
             file_id="semantic_relevance_words",
             l=['Semantically relevant words', 'Semantically irrelevant words'])


def erps_differences(project_path):
    data_preprocessor = DataPreprocessor(project_path=project_path)
    data_preparator = DataPreparator(
        data_dir=data_preprocessor.cleaned_data_dir)

    channels = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']

    relevant_words = load_evoked_response(
        dir_cleaned=data_preprocessor.cleaned_data_dir,
        filter_flag='annotation == 1',
        annotations=data_preparator.annotations).pick_channels(channels)
    relevant_sentences = load_evoked_response(
        dir_cleaned=data_preprocessor.cleaned_data_dir,
        filter_flag='selected_topic == topic',
        annotations=data_preparator.annotations).pick_channels(channels)

    irrelevant_words = load_evoked_response(
        dir_cleaned=data_preprocessor.cleaned_data_dir,
        filter_flag='annotation == 0',
        annotations=data_preparator.annotations).pick_channels(channels)
    irrelevant_sentences = load_evoked_response(
        dir_cleaned=data_preprocessor.cleaned_data_dir,
        filter_flag='selected_topic != topic',
        annotations=data_preparator.annotations).pick_channels(channels)

    combined_words = mne.combine_evoked([relevant_words, irrelevant_words], weights=[1, -1])
    combined_sentences = mne.combine_evoked([relevant_sentences, irrelevant_sentences], weights=[1, -1])

    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    fig, axs = plt.subplots(figsize=(10, 3), nrows=1,
                            ncols=4,
                            gridspec_kw={'width_ratios': [6, 6, 6, 1]}
                            )
    plt.savefig("{}/figures/erp_diff_words.pdf".format(project_path), format="pdf",
                bbox_inches="tight")
    im = combined_words.plot_topomap(ch_type="eeg",
                               times=[0.3, 0.4, 0.6],
                               axes=axs,
                               colorbar=True)
    plt.savefig("{}/figures/erp_diff_words.pdf".format(project_path),
                format="pdf",
                bbox_inches="tight")
    fig, axs = plt.subplots(figsize=(10, 3), nrows=1,
                            ncols=4,
                            gridspec_kw={'width_ratios': [6, 6, 6, 1]}
                            )
    im = combined_sentences.plot_topomap(ch_type="eeg",
                                     times=[0.3, 0.4, 0.6],
                                     axes=axs,
                                     colorbar=True)
    plt.savefig("{}/figures/erp_diff_sentences.pdf".format(project_path),
                format="pdf",
                bbox_inches="tight")


def preknowledge(data):
    data = data['text'][
        ['topic', 'pre-knowledge', 'interestingness',
         'participant']].drop_duplicates()
    data['pre-knowledge'] = data['pre-knowledge'].abs()
    data['interestingness'] = data['interestingness'].abs()
    priors = np.empty((30, 15))
    for id_x, x in enumerate(data['participant'].unique()):
        d = data[data['participant'] == x]
        for id_y, y in enumerate(data['topic'].unique()):
            try:
                priors[id_y, id_x] = d[d['topic'] == y]['pre-knowledge']
            except:
                priors[id_y, id_x] = 0
    fig, ax = plt.subplots()
    cmap = sns.cubehelix_palette(light=1, gamma=.6, n_colors=50, rot=-0.4,
                                 as_cmap=True)
    ax.imshow(priors, cmap=cmap)
    for (i, j), z in np.ndenumerate(priors):
        ax.text(j, i, f' {int(z)} ' if z > 0 else ' - ', ha='center',
                va='center', size=8,
                linespacing=4)

    plt.yticks(np.arange(len(data['topic'].unique())), data['topic'].unique())
    plt.xticks(np.arange(len(data['participant'].unique())),
               data['participant'].unique())
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.ylabel('Topic', fontsize=14)
    plt.xlabel('Participant', fontsize=14, labelpad=10)
    plt.savefig("figures\pre-knowledge.pdf", format="pdf", bbox_inches="tight")


def interestigness(data):
    data = data['text'][
        ['topic', 'pre-knowledge', 'interestingness',
         'participant']].drop_duplicates()
    data['pre-knowledge'] = data['pre-knowledge'].abs()
    data['interestingness'] = data['interestingness'].abs()
    priors = np.empty((30, 15))
    for id_x, x in enumerate(data['participant'].unique()):
        d = data[data['participant'] == x]
        for id_y, y in enumerate(data['topic'].unique()):
            try:
                priors[id_y, id_x] = d[d['topic'] == y]['interestingness']
            except:
                priors[id_y, id_x] = 0
    fig, ax = plt.subplots()
    cmap = sns.cubehelix_palette(light=1, gamma=.6, n_colors=50, rot=-0.4,
                                 as_cmap=True)
    ax.imshow(priors, cmap=cmap)
    for (i, j), z in np.ndenumerate(priors):
        ax.text(j, i, f' {int(z)} ' if z > 0 else ' - ', ha='center',
                va='center', size=8,
                linespacing=4)

    ax.xaxis.labelpad = 20
    plt.yticks(np.arange(len(data['topic'].unique())), data['topic'].unique())
    plt.xticks(np.arange(len(data['participant'].unique())),
               data['participant'].unique())
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.ylabel('Topic', fontsize=14)
    plt.xlabel('Participant', fontsize=14, labelpad=10)
    plt.savefig("figures\interestingness.pdf", format="pdf",
                bbox_inches="tight")


def overlap_all_words(data):
    sns.set(font_scale=1)
    sns.set_theme(style="white")
    data = data['text'][
        ['word', 'topic', 'semantic_relevance']].drop_duplicates()
    shared_words = np.empty((30, 30))
    for id_x, x in enumerate(data['topic'].unique()):
        x_words = data[data['topic'] == x]['word']
        for id_y, y in enumerate(data['topic'].unique()):
            y_words = data[data['topic'] == y]['word']
            u = set.intersection(set(x_words), set(y_words))
            number_shared_words = 0
            for word in list(x_words) + list(y_words):
                if word in u:
                    number_shared_words += 1
            print(f"Topic {x} and topic {y} share {len(u)} unique words.")
            fraction = number_shared_words / len(list(x_words) + list(y_words))
            shared_words[id_x, id_y] = int(fraction * 100)

    mask = np.triu(np.ones_like(shared_words, dtype=bool))
    cmap = sns.cubehelix_palette(light=1, gamma=.6, n_colors=50, rot=-0.4,
                                 as_cmap=True)
    _ = sns.heatmap(shared_words, mask=mask, center=0, annot=True,
                    linewidths=1,
                    square=True, cmap=cmap, cbar_kws={"shrink": .5},
                    annot_kws={"size": 6}, cbar=False)

    x_labels = data['topic'].unique().tolist()
    y_labels = data['topic'].unique().tolist()
    x_labels.pop(len(x_labels) - 1)
    y_labels.pop(0)
    x_labels.insert(len(x_labels), "")
    y_labels.insert(0, "")
    plt.xticks(np.arange(len(x_labels)), x_labels)
    plt.yticks(np.arange(len(y_labels)), y_labels)
    plt.xticks(rotation=90, fontsize=6, ha="left")
    plt.yticks(rotation=0, fontsize=6, va="top")
    plt.savefig("figures\overlap_all_words.pdf", format="pdf",
                bbox_inches="tight")


def overlap_semantic_words(data):
    sns.set(font_scale=1)
    sns.set_theme(style="white")
    data = data['text'][
        ['word', 'topic', 'semantic_relevance']].drop_duplicates()
    data = data[data['semantic_relevance'] == 1]
    shared_words = np.empty((30, 30))
    for id_x, x in enumerate(data['topic'].unique()):
        x_words = data[data['topic'] == x]['word']
        for id_y, y in enumerate(data['topic'].unique()):
            y_words = data[data['topic'] == y]['word']
            u = set.intersection(set(x_words), set(y_words))
            number_shared_words = 0
            for word in list(x_words) + list(y_words):
                if word in u:
                    number_shared_words += 1
            print(f"Topic {x} and topic {y} share {len(u)} unique words.")
            fraction = number_shared_words / len(list(x_words) + list(y_words))
            shared_words[id_x, id_y] = int(fraction * 100)

    mask = np.triu(np.ones_like(shared_words, dtype=bool))
    cmap = sns.cubehelix_palette(light=1, gamma=.6, n_colors=50, rot=-0.4,
                                 as_cmap=True)
    _ = sns.heatmap(shared_words, mask=mask, center=0, annot=True, linewidths=1,
                    square=True, cmap=cmap, cbar_kws={"shrink": .6},
                    annot_kws={"size": 6}, cbar=False)
    x_labels = data['topic'].unique().tolist()
    y_labels = data['topic'].unique().tolist()
    x_labels.pop(len(x_labels) - 1)
    y_labels.pop(0)
    x_labels.insert(len(x_labels), "")
    y_labels.insert(0, "")
    plt.xticks(np.arange(len(x_labels)), x_labels)
    plt.yticks(np.arange(len(y_labels)), y_labels)
    plt.xticks(rotation=90, fontsize=6, ha="left")
    plt.yticks(rotation=0, fontsize=6, va="top")
    plt.savefig("figures\overlap_sem_words.pdf", format="pdf",
                bbox_inches="tight")


def semantic_words(data):
    data['text']['fraction_sem_rel_per_topic'] = \
        data['text'].groupby("topic")["semantic_relevance"].transform(
            "mean")
    data = data['text'][
        ['topic', 'fraction_sem_rel_per_topic']].drop_duplicates()
    data['fraction_sem_rel_per_topic'].mean()
    sns.set(font_scale=2.2)
    a4_dims = (15, 4)
    fig, ax = plt.subplots(figsize=a4_dims)
    ax = sns.barplot(x=data['topic'], y=data['fraction_sem_rel_per_topic'])
    ax.set_xlabel(xlabel="Topic")
    ax.set_ylabel(ylabel="")
    fig.tight_layout()
    plt.xticks(rotation=90)
    plt.axhline(y=data['fraction_sem_rel_per_topic'].mean(), color='r',
                linestyle='-')
    plt.savefig("figures\sem_words.pdf", format="pdf", bbox_inches="tight")
    plt.close()


def distribution_word_lengths(data):
    data['text']['w_length'] = data['text']['word'].str.len()
    sns.set(font_scale=3.2)
    a4_dims = (15, 10)
    fig, ax = plt.subplots(figsize=a4_dims)
    bins = np.arange(1, max(data['text']['w_length']) + 1)
    ax = sns.histplot(data=data['text'], x="w_length", discrete=True,
                      log_scale=(False, True), bins=bins)
    ax.set_xlabel(xlabel="Word length")
    ax.set_ylabel(ylabel="Count of words")
    ax.set_xticks(bins)
    ax.set_xticklabels(bins)
    fig.tight_layout()
    plt.savefig("figures\dist_word_lengths.pdf", format="pdf",
                bbox_inches="tight")
    plt.close()


def distribution_topics(data):
    dist_data = data['text'][
        ['selected_topic', 'participant']].drop_duplicates()
    dist_data['count'] = dist_data.groupby(['selected_topic'])[
        'participant'].transform('count')
    dist_data = dist_data[
        ['selected_topic', 'count']].drop_duplicates().sort_values('count',
                                                                   ascending=False).reset_index(
        drop=True)

    sns.set(font_scale=3.2)
    a4_dims = (15, 10)
    fig, ax = plt.subplots(figsize=a4_dims)
    ax = sns.barplot(data=dist_data, x="selected_topic", y='count',
                     color='steelblue')
    ax.set_xlabel(xlabel="Topic")
    ax.set_ylabel(ylabel="Number of times a topic \n was selected")
    plt.xticks(rotation=90)
    fig.tight_layout()
    plt.savefig("figures\dist_topic.pdf", format="pdf", bbox_inches="tight")
    plt.close()


def distribution_topic_words(data):
    # Group by participant and topic and count the number of words
    count_data = data['text'][['topic', 'participant', 'word']]
    count_data['count'] = count_data.groupby(['topic', 'participant'])[
        'word'].transform('count')
    count_data = count_data[['topic', 'participant', 'count']].drop_duplicates()
    count_data = count_data.groupby(['topic'], as_index=False)['count'].mean()
    count_data = count_data.sort_values('count', ascending=False).reset_index(
        drop=True)

    sns.set(font_scale=3.2)
    a4_dims = (15, 10)
    fig, ax = plt.subplots(figsize=a4_dims)
    ax = sns.barplot(data=count_data, x='topic', y='count', color='steelblue')
    ax.set_xlabel(xlabel='Topic')
    ax.set_ylabel(ylabel='Number of words')
    plt.xticks(rotation=90)
    fig.tight_layout()
    plt.savefig('figures\dist_topic_words.pdf', format='pdf',
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = create_args()
    arguments = parser.parse_args()
    set_seed(1)
    dataset = DatasetWords(
        dir_data=os.path.join(arguments.project_path,
                              'data_prepared_for_benchmark'))
    erps_word_level_sentence_level(arguments.project_path)
    erps_differences(arguments.project_path)
    interestigness(dataset.data)
    preknowledge(dataset.data)
    distribution_topic_words(dataset.data)
    distribution_topics(dataset.data)
    distribution_word_lengths(dataset.data)
    semantic_words(dataset.data)
    overlap_semantic_words(dataset.data)
    overlap_all_words(dataset.data)
    erps_electrodes(arguments.project_path)
