"""Module defining the `OneHot` model class."""

import numpy as np
import tensorflow as tf

from models.base_models import BaseModel
from models.utils import mask_embeddings


class OneHot(BaseModel):
  """Sentence encoder based on the words' one-hot vectors.

  Hyperparameters:
    mask_sequences: whether to exclude padding tokens from calculations.
  """

  def build_encoder(self, input_batch):
    one_hot_depth = tf.cast(tf.reduce_max(input_batch) + 1, tf.int32)
    one_hot_batch = tf.one_hot(input_batch, one_hot_depth)

    if self.hparams['mask_sequences']:
      one_hot_batch = mask_embeddings(input_batch, one_hot_batch)

    return tf.reduce_sum(one_hot_batch, axis=1)

  def to_sequence(self, sentences):
    unique_words = {}
    num_unique_words = 1  # compensate for padding
    max_seq_length = 0

    sequences = []
    for sentence in sentences:

      word_sequence = []
      for word in sentence:
        if word not in unique_words:
          unique_words[word] = num_unique_words
          num_unique_words += 1
        word_sequence.append(unique_words[word])

      sequences.append(word_sequence)
      max_seq_length = max(len(word_sequence), max_seq_length)

    # Pad to the biggest sequence length
    for i in range(len(sequences)):
      while len(sequences[i]) < max_seq_length:
        sequences[i].append(0)

    return np.array(sequences)

  def encode_sentences(self, sentences):
    sess = tf.get_default_session()
    sequences = self.to_sequence(sentences)
    return sess.run(self.encode, feed_dict={self.input_placeholder: sequences})
