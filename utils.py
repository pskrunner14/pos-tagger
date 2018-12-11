from collections import defaultdict
from collections import Counter

import nltk
import sys
import numpy as np

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


def load_dataset():
    nltk.download('brown')
    nltk.download('universal_tagset')

    # list of sentences that are list of tuples like (<word>, <tag>)
    data = nltk.corpus.brown.tagged_sents(tagset='universal')
    all_tags = ['#PAD#', '#EOS#', '#UNK#', 'ADV', 'NOUN', 'ADP', 'PRON',
                'DET', '.', 'PRT', 'VERB', 'X', 'NUM', 'CONJ', 'ADJ']

    # convert all words to lower case
    data = np.array([[(word.lower(), tag) for word, tag in sentence]
                     for sentence in data])

    print(f'Brown Corpus dims: {data.shape}')

    train_data, test_data = train_test_split(
        data, test_size=0.25, random_state=42)

    # word frequency counter
    word_counts = Counter()
    for sentence in data:
        words, _ = zip(*sentence)
        word_counts.update(words)

    # take the 10000 most common words from the counter to build vocab `all_words`
    all_words = ['#PAD#', '#EOS#', '#UNK#'] + \
        list(list(zip(*word_counts.most_common(10000)))[0])

    #  measure what fraction of data corpus words are in the dictionary i.e. vocab coverage
    print("Vocab coverage is {:.5f}%".format(
        float(sum(word_counts[w] for w in all_words)) / sum(word_counts.values()) * 100))

    # create `word_to_id` and `tag_to_id` dictionaries for later conversion

    word_to_id = defaultdict(
        lambda: 1, {word: i for i, word in enumerate(all_words)})  # lambda: 1 since id of `#UNK#` is 1
    tag_to_id = {tag: i for i, tag in enumerate(all_tags)}

    return train_data, test_data, all_words, word_to_id, all_tags, tag_to_id


def to_matrix(lines, token_to_id, max_len=None, pad=0, dtype='int32', time_major=False):
    """Converts a list of names into rnn-digestable matrix with paddings added after the end"""

    max_len = max_len or max(map(len, lines))
    matrix = np.empty([len(lines), max_len], dtype)
    matrix.fill(pad)

    for i in range(len(lines)):
        line_ix = list(map(token_to_id.__getitem__, lines[i]))[:max_len]
        matrix[i, :len(line_ix)] = line_ix

    return matrix.T if time_major else matrix


def generate_batches(sentences, all_tags, word_to_id, tag_to_id, batch_size=64, max_len=None, pad=0):
    assert isinstance(
        sentences, np.ndarray), "Make sure sentences is a numpy array"

    while True:
        indices = np.random.permutation(np.arange(len(sentences)))
        for start in range(0, len(indices) - 1, batch_size):
            batch_indices = indices[start: start + batch_size]
            batch_words, batch_tags = [], []
            for sent in sentences[batch_indices]:
                words, tags = zip(*sent)
                batch_words.append(words)
                batch_tags.append(tags)

            batch_words = to_matrix(batch_words, word_to_id, max_len, pad)
            batch_tags = to_matrix(batch_tags, tag_to_id, max_len, pad)

            batch_tags_1hot = to_categorical(batch_tags, len(
                all_tags)).reshape(batch_tags.shape + (-1, ))
            yield batch_words, batch_tags_1hot
