"""Module defining the `SkipThoughts` model class."""

import re
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq

from models.base_models import EncoderDecoderModel


class SkipThoughts(EncoderDecoderModel):
  """Skip-thoughts model (https://arxiv.org/abs/1506.06726).

  Hyperparameters:
    batch_size: ,
    output_size: ,
    learning_rate: ,
    max_grad_norm: .

  For additional hyperparameters, refer to parent class.
  """

  def __init__(self, **kwargs):
    """Constructor."""
    super().__init__(**kwargs)

    # Training mode.
    if self.input_iterator is not None:
      _, (fw_labels, bw_labels) = self.unpack_iterator()

      fw_logits = self.build_decoder(self.encode, fw_labels)
      bw_logits = self.build_decoder(self.encode, bw_labels)

      fw_loss = self.build_loss(fw_logits, fw_labels)
      bw_loss = self.build_loss(bw_logits, bw_labels)
      self.loss = fw_loss + bw_loss

      # Optimizer with gradient clipping
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(
        tf.gradients(self.loss, tvars), self.hparams['max_grad_norm'])
      optimizer = tf.train.AdamOptimizer(self.hparams['learning_rate'])
      self.train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=self.global_step)

  def unpack_iterator(self):
    iit = self.input_iterator
    return iit[1], (iit[0], iit[2])

  def build_decoder(self, thought, labels):
    """Internally used to build a decoder RNN."""
    labels = self.build_sequence_with_sos(labels)
    decoder_in = tf.nn.embedding_lookup(self.embedding_matrix, labels)
    output_size = self.hparams['output_size']

    if self._cuda:
      decoder_out = tf.contrib.cudnn_rnn.CudnnGRU(
        1, output_size, direction='unidirectional')(decoder_in)[0]

    else:
      rnn_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(output_size)
      max_seq_lengths = tf.constant(
        [self.hparams['max_sequence_length']] * self.hparams['batch_size'])
      helper = seq2seq.TrainingHelper(
        decoder_in, max_seq_lengths, time_major=self.hparams['time_major'])
      decoder = seq2seq.BasicDecoder(rnn_cell, helper, thought)
      decoder_out = seq2seq.dynamic_decode(
        decoder, output_time_major=self.hparams['time_major'])[0].rnn_output

    return decoder_out
