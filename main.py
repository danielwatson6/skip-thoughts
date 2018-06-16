import argparse
from os import listdir

import tensorflow as tf
from tensorflow.contrib import seq2seq
from gensim.models import KeyedVectors


parser = argparse.ArgumentParser()

parser.add_argument('--initial_lr', type=float, default=1e-3,
  help="Initial learning rate.")
parser.add_argument('--vocabulary_size', type=int, default=20000,
  help="Keep only the n most common words of the training data."))
parser.add_argument('--batch_size', type=int, default=16,
  help="Stochastic gradient descent minibatch size.")
parser.add_argument('--hidden_size', type=int, default=512,
  help="Number of hidden units for the encoder and decoder GRUs.")
parser.add_argument('--max_length', type=int, default=40,
  help="Truncate input and output sentences to maximum length n.")
parser.add_argument('--sample_prob', type=float, default=.3,
  help="Decoder probability to sample from its predictions duing training.")
parser.add_argument('--max_grad_norm', type=float, default=5.,
  help="Clip gradients to the specified maximum norm.")
parser.add_argument('--embeddings_path', type=str, default="./word2vecModel",
  help="Path to the pre-trained word embeddings model.")
parser.add_argument('--dataset_path', type=str, default="./books",
  help="Path to the BookCorpus dataset files.")
parser.add_argument('--train_embeddings', type=bool, default=False,
  help="Set to backpropagate over the word embedding matrix.")
parser.add_argument('--encode', type=str, default='',
  help="Set to a file path to obtain sentence vectors for the given lines.")
parser.add_argument('--eos_token', type=bool, default=False,
  help="Set to use the end-of-string token when running on inference.")

FLAGS = parser.parse_args()


def get_sequence_length(seq):
  """Get the length of the provided integer sequence.
     NOTE: this method assumes the padding id is 0."""
  return tf.reduce_sum(tf.sign(seq), reduction_indices=1)


def build_encoder(inputs, seq_length):
  """When called, adds the encoder layer to the computational graph."""
  fw_cell = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size, name="encoder_fw")
  bw_cell = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size, name="encoder_bw")
  return tf.nn.bidirectional_dynamic_rnn(
    fw_cell, bw_cell, inputs, sequence_length=seq_length)


def build_decoder(thought, labels, seq_length, embedding_matrix, name_id=0):
  """When called, adds a decoder layer to the computational graph."""
  cell = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size, name="decoder%d" % name_id)
  
  # For convenience-- this gets passed as an argument later.
  def get_embeddings(query):
    return tf.nn.embedding_lookup(embedding_matrix, query)

  # Scheduled sampling with constant probability. Labels are shifted to the
  # right by adding a start-of-string token (id: 2).
  shifted_labels = tf.concat(
    [tf.tile(2, FLAGS.batch_size), labels[::-1]], axis=1)
  helper = seq2seq.ScheduledEmbeddingTrainingHelper(
    shifted_labels, sequence_length, get_embeddings, FLAGS.sample_prob)
  # Final layer for both decoders that converts decoder output to predictions.
  if decoder_id > 0:
     tf.get_variable_scope().reuse_variables()
  output_layer = tf.layers.Dense(FLAGS.vocabulary_size, name="output_layer")
  decoder = seq2seqBasicDecoder(
    cell, helper, thought, output_layer=output_layer)
  return seq2seq.dynamic_decode(decoder, impute_finished=True)


def build_model():
  """When called, builds the whole computational graph.."""
  inputs = tf.placeholder(
    tf.int32, shape=[FLAGS.batch_size, FLAGS.max_length], name="inputs")
  fw_labels = tf.placeholder(
    tf.int32, shape=[FLAGS.batch_size, FLAGS.max_length], name="fw_labels")
  bw_labels = tf.placeholder(
    tf.int32, shape=[FLAGS.batch_size, FLAGS.max_length], name="bw_labels")
  
  input_length = get_sequence_length(inputs)
  fw_length = get_sequence_length(fw_labels)
  bw_length = get_sequence_length(bw_labels)

  lr = tf.Variable(FLAGS.initial_lr, trainable=False, name="lr")
  global_step = tf.Variable(0, trainable=False, name="global_step")

  w2v_model = KeyedVectors.load_word2vec_format(FLAGS.embeddings_path)
  embedding_matrix = tf.Variable(
    w2v_model.syn0, trainable=FLAGS.train_embeddings, name="embedding_matrix")
  embeddings = tf.nn.embedding_lookup(embedding_matrix, inputs)
  
  thought = build_encoder(embeddings, input_length)

  fw_logits = build_decoder(thought, fw_labels, fw_length, embedding_matrix)
  bw_logits = build_decoder(thought, fw_labels, bw_length, embedding_matrix,
                            name_id=1)
  
  # Mask the loss to avoid taking padding tokens into account.
  fw_mask = tf.sequence_mask(fw_length, FLAGS.max_length, dtype=tf.float32)
  bw_mask = tf.sequence_mask(bw_length, FLAGS.max_length, dtype=tf.float32)
  loss = seq2seq.sequence_loss(fw_logits, fw_labels, fw_mask) + \
         seq2seq.sequence_loss(bw_logits, bw_labels, bw_mask)

  # Adam optimizer with gradient clipping to prevent exploding gradients.
  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(
    tf.gradients(loss, tvars), FLAGS.max_grad_norm)
  train_op = tf.train.AdamOptimizer(lr).apply_gradients(
    zip(grads, tvars), global_step=global_step)

  return thought, loss, lr, global_step, train_op


def sequences(fp):
  """Yield the integer id sentences from a file object."""
  sentence_buffer = []
  # Compensate for go token and the end-of-string token if requested.
  append_eos = not FLAGS.encode or FLAGS.eos_token
  max_length = FLAGS.max_length - 1
  if append_eos:
    max_length -= 1
  for line in fp:
    words = line.split()
    for word in words:
      sentence_buffer.append(...)
      if len(sentence_buffer) == max_length or word == '.':
        if append_eos:
          sentence_buffer.append(3)
        # Pad the sentence as a final step.
        while len(sentence_buffer) < FLAGS.max_length:
          sentence_buffer.append(0)
        yield sentence_buffer
        sentence_buffer = []


def batches(sequences):
  for 


if __name__ == '__main__':
  build_model()

  # Inference
  if FLAGS.encode:
    # TODO: Restore model
    
    with open(FLAGS.encode) as f:
      s
  
  # Training
  else:
    for filename in listdir(FLAGS.dataset_path):
      seqs = list(sequences())

