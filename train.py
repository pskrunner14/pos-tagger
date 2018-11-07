import sys
import click
import logging
import coloredlogs

import numpy as np
import keras
import keras.layers as L

from utils import load_dataset, to_matrix, generate_batches

batch_size = 64

train_data, test_data, all_words, word_to_id, all_tags, tag_to_id = load_dataset()


def compute_test_accuracy(model):
    test_words, test_tags = zip(*[zip(*sentence) for sentence in test_data])
    test_words, test_tags = to_matrix(
        test_words, word_to_id), to_matrix(test_tags, tag_to_id)

    # predict tag probabilities of shape [batch,time,n_tags]
    predicted_tag_probabilities = model.predict(test_words, verbose=1)
    predicted_tags = predicted_tag_probabilities.argmax(axis=-1)

    # compute accurary excluding padding
    numerator = np.sum(np.logical_and(
        (predicted_tags == test_tags), (test_words != 0)))
    denominator = np.sum(test_words != 0)
    return float(numerator)/denominator


def create_model():
    # Define a model that utilizes bidirectional GRU/LSTM celss
    model = keras.models.Sequential()

    model.add(L.InputLayer([None], dtype='int32'))
    model.add(L.Embedding(len(all_words), 64))

    model.add(L.Bidirectional(L.GRU(64, return_sequences=True)))
    model.add(L.Bidirectional(L.GRU(64, return_sequences=True)))
    model.add(L.Bidirectional(L.LSTM(64, return_sequences=True)))

    # add top layer that predicts tag probabilities
    model.add(L.TimeDistributed(L.Dense(len(all_tags), activation='softmax')))
    return model


class EvaluateAccuracy(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        sys.stdout.flush()
        print("\nMeasuring validation accuracy...")
        acc = compute_test_accuracy(self.model)
        print("\nValidation accuracy: %.5f\n" % acc)
        sys.stdout.flush()


model = create_model()
model.compile('adam', 'categorical_crossentropy')
model.fit_generator(generate_batches(train_data, all_tags, word_to_id, tag_to_id), len(train_data)/batch_size,
                    callbacks=[EvaluateAccuracy()], epochs=5,)

acc = compute_test_accuracy(model)
print("\nFinal accuracy: %.5f" % acc)
