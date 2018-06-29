"""Script for training a skip-thoughts or autoencoder model."""

import argparse
import itertools
import os
import time

import tensorflow as tf
from gensim.models import KeyedVectors

from models import SkipThoughts, VAE


parser = argparse.ArgumentParser()


def argparse_bool(s):
  """Type for argparse making falsy values behave as false."""
  return s.lower() in ['true', 't', 'yes', '1']


# Hyperparameter args

parser.add_argument(
  '--learning_rate', type=float, default=1e-3,
  help="Initial learning rate.")

parser.add_argument(
  '--vocabulary_size', type=int, default=100000,
  help="Keep only the n most common words of the training data.")

parser.add_argument(
  '--batch_size', type=int, default=128,
  help="Stochastic gradient descent minibatch size.")

parser.add_argument(
  '--output_size', type=int, default=512,
  help="Number of hidden units for the encoder and decoder GRUs.")

parser.add_argument(
  '--max_sequence_length', type=int, default=40,
  help="Truncate input and output sentences to maximum length n.")

parser.add_argument(
  '--encoder_depth', type=int, default=1,
  help="Number of neural layers in the encoder.")

parser.add_argument(
  '--max_grad_norm', type=float, default=5.,
  help="Clip gradients to the specified maximum norm.")

parser.add_argument(
  '--concat', type=argparse_bool, default=False,
  help="Set to true to concatenate rather than add the biRNN outputs. "
       "Note this doubles the dimension of the output vectors.")

parser.add_argument(
  '--softmax_samples', type=int, default=64,
  help="Set to n > 0 to use sampled softmax with n candidate samples.")

parser.add_argument(
  '--train_word_embeddings', type=argparse_bool, default=False,
  help="Set to backpropagate over the word embeddings.")

parser.add_argument(
  '--train_special_embeddings', type=argparse_bool, default=False,
  help="Set to backpropagate over the special token embeddings.")

parser.add_argument(
  '--eos_token', type=argparse_bool, default=True,
  help="Set to use the end-of-string token when running on inference.")

parser.add_argument(
  '--epochs', type=int, default=1,
  help="Number of epochs to train the model for.")

# Performance args

parser.add_argument(
  '--time_major', type=argparse_bool, default=True,
  help="Set to feed time-major batches to the RNNs.")

parser.add_argument(
  '--cuda', type=argparse_bool, default=False,
  help="Set to False to forcefully disable the use of Cudnn ops.")

parser.add_argument(
  '--benchmark', type=int, default=0,
  help="Set to n > 0 to estimate running time by executing n steps.")

# Configuration args

parser.add_argument(
  '--model', type=str, default="skip_thoughts",
  help="Set to 'skip_thoughts', 'ae', or 'vae'.")

parser.add_argument(
  '--embeddings_path', type=str, default="word2vecModel",
  help="Path to the pre-trained word embeddings model.")

parser.add_argument(
  '--input', type=str, default="data/books_tf",
  help="Path to the directory containing the dataset TFRecord files.")

parser.add_argument(
  '--model_name', type=str, default="default",
  help="Will save/restore model in ./output/[model_name].")

parser.add_argument(
  '--num_steps_per_save', type=int, default=1000,
  help="Save the model's trainable variables every n steps.")


FLAGS = parser.parse_args()


def parse_and_pad(seq):
  """Parse a `tf.SequenceExample` instance."""
  sequence_features = {
    "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
  }
  _, sequence_parsed = tf.parse_single_sequence_example(
    serialized=seq, sequence_features=sequence_features)
  # Pad the sequence
  t = sequence_parsed["tokens"]
  if FLAGS.eos_token:
    t = tf.pad(t, [[0, 1]], constant_values=3)
  return tf.pad(t, [[0, FLAGS.max_sequence_length - tf.shape(t)[0]]])


def skip_thoughts_iterator(filenames):
  """Build the input pipeline for training a skip-thoughts model."""
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(parse_and_pad).repeat(FLAGS.epochs)

  dataset = dataset.apply(tf.contrib.data.sliding_window_batch(
    window_size=FLAGS.batch_size, stride=1))
  dataset = dataset.batch(FLAGS.batch_size).map(lambda x: x[:3])

  if FLAGS.time_major:
    dataset = dataset.map(lambda x: tf.transpose(x, perm=[0, 2, 1]))

  dataset = dataset.prefetch(1)
  return dataset.make_one_shot_iterator().get_next()


def ae_iterator(filenames):
  """Build the input pipeline for training an autoencoder model."""
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(parse_and_pad).repeat(FLAGS.epochs)

  dataset = dataset.batch(FLAGS.batch_size)

  if FLAGS.time_major:
    dataset = dataset.map(lambda x: tf.transpose(x))

  dataset = dataset.prefetch(1)
  return dataset.make_one_shot_iterator().get_next()


if __name__ == '__main__':
  print("Loading word vector model...")
  start = time.time()

  w2v_model = KeyedVectors.load(FLAGS.embeddings_path, mmap='r')

  duration = time.time() - start
  print("Done ({:0.4f}s).".format(duration))

  # Set up configuration options.

  cuda = FLAGS.cuda
  gpu_available = tf.test.is_gpu_available(cuda_only=True)
  if cuda and not (FLAGS.time_major or gpu_available):
    print("WARNING: disabling CUDA ops. GPU must be available and time "
          "major mode must be enabled.")
    cuda = False

  if FLAGS.model == 'skip_thoughts':
    Model = SkipThoughts
    iterator_fn = skip_thoughts_iterator
  else:
    Model = VAE
    iterator_fn = ae_iterator

  # Build the graph model.

  print("Building computational graph...")
  start = time.time()

  graph = tf.Graph()
  with graph.as_default():

    filenames = [os.path.join(FLAGS.input, f) for f in os.listdir(FLAGS.input)]
    iterator = iterator_fn(filenames)

    hparams = {
      'vocabulary_size': FLAGS.vocabulary_size,
      'batch_size': FLAGS.batch_size,
      'output_size': FLAGS.output_size,
      'max_sequence_length': FLAGS.max_sequence_length,
      'encoder_depth': FLAGS.encoder_depth,
      'learning_rate': FLAGS.learning_rate,
      'max_grad_norm': FLAGS.max_grad_norm,
      'concat': FLAGS.concat,
      'softmax_samples': FLAGS.softmax_samples,
      'train_special_embeddings': FLAGS.train_special_embeddings,
      'train_word_embeddings': FLAGS.train_word_embeddings,
      'time_major': FLAGS.time_major,
    }
    m = Model(
      model_name=FLAGS.model_name, w2v_model=w2v_model, hparams=hparams,
      input_iterator=iterator, cuda=cuda)

  duration = time.time() - start
  print("Done ({:0.4f}s).".format(duration))

  with tf.Session(graph=graph) as sess:
    m.start()

    # Used for benchmarking running time.
    i = 0
    min_duration = float('inf')  # infinity
    average_duration = 0

    # Training loop.
    while True:
      start = time.time()
      loss_, _ = sess.run([m.loss, m.train_op])
      duration = time.time() - start
      current_step = sess.run(m.global_step)

      # Only benchmark running time if requested.
      if FLAGS.benchmark:
        i += 1
        min_duration = min(duration, min_duration)
        average_duration = (duration + (i - 1) * average_duration) / i
        if i >= FLAGS.benchmark:
          print("Running time benchmarks for", FLAGS.benchmark, "steps:")
          print("  Average:", average_duration)
          print("  Minimum:", min_duration)
          exit()
        else:
          continue

      print(
        "Step", current_step,
        "(loss={:0.4f}, time={:0.4f}s)".format(loss_, duration))

      if current_step % FLAGS.num_steps_per_save == 0:
        m.save()
