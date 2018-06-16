import argparse

import gensim
import tensorflow as tf
from tensorflow.contrib import seq2seq


INITIAL_LR = 1e-3
VOCABULARY_SIZE = 20000
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 512
MAX_LENGTH = 40
SAMPLE_P = .3
MAX_GRAD_NORM = 5.


def get_sequence_length(seq):
  """Get the length of the provided integer sequence.
     NOTE: this method assumes the padding id is 0."""
  return tf.reduce_sum(tf.sign(seq), reduction_indices=1)


def build_encoder(inputs, seq_length):
  fw_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE, name="encoder_fw")
  bw_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE, name="encoder_bw")
  return tf.nn.bidirectional_dynamic_rnn(
    fw_cell, bw_cell, inputs, sequence_length=seq_length)


def build_decoder(thought, labels, seq_length, embedding_matrix, name_id=0):
  cell = tf.nn.rnn_cel.GRUCell(HIDDEN_SIZE, name="decoder%d" % name_id)
  # Scheduled sampling with constant probability; labels are shift
  shifted_labels = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
  helper = seq2seq.ScheduledEmbeddingTrainingHelper(
    shifted_labels, sequence_length,
    lambda x: tf.nn.embedding_lookup(embedding_matrix, x), SAMPLE_P)
  # Final layer for both decoders that converts decoder output to predictions.
  if decoder_id > 0:
     tf.get_variable_scope().reuse_variables()
  output_layer = tf.layers.Dense(VOCABULARY_SIZE, name="output_layer")
  decoder = seq2seqBasicDecoder(
    cell, helper, thought, output_layer=output_layer)
  return seq2seq.dynamic_decode(decoder, impute_finished=True)


def build_model():
  inputs = tf.placeholder(
    tf.int32, shape=[BATCH_SIZE, MAX_LENGTH], name="inputs")
  fw_labels = tf.placeholder(
    tf.int32, shape=[BATCH_SIZE, MAX_LENGTH], name="fw_labels")
  bw_labels = tf.placeholder(
    tf.int32, shape=[BATCH_SIZE, MAX_LENGTH], name="bw_labels")
  
  input_length = get_sequence_length(inputs)
  fw_length = get_sequence_length(fw_labels)
  bw_length = get_sequence_length(bw_labels)

  lr = tf.Variable(INITIAL_LR, trainable=False, name="lr")
  global_step = tf.Variable(0, trainable=False, name="global_step")

  w2v_model = gensim.models.KeyedVectors.load_word2vec_format('word2vecModel')
  embedding_matrix = tf.constant(w2v_model.syn0)
  embeddings = tf.nn.embedding_lookup(inputs, embedding_matrix)
  
  thought = build_encoder(embeddings, input_length)
  
  fw_logits = build_decoder(thought, fw_labels, fw_length, embedding_matrix)
  bw_logits = build_decoder(thought, fw_labels, bw_length, embedding_matrix,
                            name_id=1)
  
  fw_mask = tf.sequence_mask(fw_length, MAX_LENGTH, dtype=tf.float32)
  bw_mask = tf.sequence_mask(bw_length, MAX_LENGTH, dtype=tf.float32)
  loss = seq2seq.sequence_loss(fw_logits, fw_labels, fw_mask) + \
         seq2seq.sequence_loss(bw_logits, bw_labels, bw_mask)

  # Adam optimizer with gradient clipping to prevent exploding gradients.
  lr = tf.Variable(INITIAL_LR, trainable=False, name="lr")
  global_step = tf.Variable(0, trainable=False, name="global_step")
  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), MAX_GRAD_NORM)
  train_op = tf.train.AdamOptimizer(lr).apply_gradients(
    zip(grads, tvars), global_step=global_step)

  return thought, loss, lr, global_step, train_op



