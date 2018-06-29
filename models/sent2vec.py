"""Module defining the `Sent2Vec` model class."""

import numpy as np
import tensorflow as tf

from models.base_models import WordVectorModel


def n_gram_weights(n, sequence_lengths, max_length):
  """Given n, get the n-gram weights for a list of lengths.

  Example:
    n = 3
    sequence_lengths = [4,5,3]
    max_length = 6

    >>> [[1 1 1 0 0 0] + [0 1 1 1 0 0],
         [1 1 1 0 0 0] + [0 1 1 1 0 0] + [0 0 1 1 1 0],
         [1 1 1 0 0 0]]
  """
  roll_vec = [np.pad([1.] * n, [0, max_length], 'constant')]

  def f(l):
    return tf.reduce_sum(
      tf.map_fn(lambda i: tf.manip.roll(roll_vec, i), tf.range(l)), axis=0)

  return tf.reduce_sum(tf.map_fn(f, sequence_lengths), axis=0)


class Sent2Vec(WordVectorModel):
  """Sent2Vec model (https://arxiv.org/pdf/1703.02507).

  Hyperparameters:
    minn: minimum n-gram size,
    maxn: maximum n-gram size,
    sos_token: whether to append a start-of-sentence token.

  For additional hyperparameters, refer to parent class.
  """

  def build_encoder(self, input_batch):
    # The parent class builds the word embedding matrix.
    super().build_encoder(input_batch)

    # Concatenate the start-of-sentence token if specified.
    if self.hparams['sos_token']:
      input_batch = self.build_sequence_with_sos(input_batch)

    reduction_axis = int(not self.hparams['time_major'])
    sequence_lengths = tf.reduce_sum(
      tf.sign(inputs), reduction_indices=reduction_axis)

    # Summing the n-grams is equivalent to finding a scale for the vectors.
    weights = tf.zeros(
      [self.hparams['batch_size'], self.hparams['max_sequence_length']])
    for n in range(self.hparams['minn'], self.hparams['maxn'] + 1):
      weights += n_gram_weights(
        n, sequence_lengths, self.hparams['max_sequence_length'])

    # Return the weighted sums of the vectors.
    input_batch = tf.nn.embedding_lookup(self.embeddings, input_batch)
    input_batch = tf.transpose(input_batch, perm=[2, 0, 1])
    return tf.reduce_sum(weights * input_batch, axis=reduction_axis + 1)
