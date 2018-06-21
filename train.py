import argparse
import itertools
import os
import time

import tensorflow as tf
from gensim.models import KeyedVectors

from skip_thoughts import SkipThoughts


parser = argparse.ArgumentParser()

# Hyperparameter args
parser.add_argument('--initial_lr', type=float, default=1e-3,
  help="Initial learning rate.")
parser.add_argument('--vocabulary_size', type=int, default=20000,
  help="Keep only the n most common words of the training data.")
parser.add_argument('--batch_size', type=int, default=16,
  help="Stochastic gradient descent minibatch size.")
parser.add_argument('--output_size', type=int, default=512,
  help="Number of hidden units for the encoder and decoder GRUs.")
parser.add_argument('--max_length', type=int, default=40,
  help="Truncate input and output sentences to maximum length n.")
parser.add_argument('--sample_prob', type=float, default=0.,
  help="Decoder probability to sample from its predictions duing training.")
parser.add_argument('--max_grad_norm', type=float, default=5.,
  help="Clip gradients to the specified maximum norm.")
parser.add_argument('--concat', type=bool, default=False,
  help="Set to true to concatenate rather than add the biRNN outputs. "
       "Note this doubles the dimension of the output vectors.")
parser.add_argument('--train_word_embeddings', type=bool, default=False,
  help="Set to backpropagate over the word embeddings.")
parser.add_argument('--train_special_embeddings', type=bool, default=False,
  help="Set to backpropagate over the special token embeddings.")
parser.add_argument('--eos_token', type=bool, default=False,
  help="Set to use the end-of-string token when running on inference.")

# Configuration args
parser.add_argument('--embeddings_path', type=str, default="word2vecModel",
  help="Path to the pre-trained word embeddings model.")
parser.add_argument('--input', type=str, default="books_tf",
  help="Path to the directory containing the dataset TFRecord files.")
parser.add_argument('--model_name', type=str, default="default",
  help="Will save/restore model in ./output/[model_name].")
parser.add_argument('--num_steps_per_save', type=int, default=4500,
  help="Save the model's trainable variables every n steps.")

FLAGS = parser.parse_args()


def parse_and_pad(seq):
  # Extract features from `tf.SequenceExample`
  sequence_features = {
    "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
  }
  _, sequence_parsed = tf.parse_single_sequence_example(
    serialized=seq, sequence_features=sequence_features)
  # Pad the sequence
  t = sequence_parsed["tokens"]
  return tf.pad(t, [[0, FLAGS.max_length - tf.shape(t)[0]]])


def train_iterator(filenames):
  """Build the input pipeline for training.."""
  
  def _single_iterator(skip):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_and_pad)  # TODO: add option for parallel calls
    return dataset.apply(
      tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size))
  
  # Get the contiguous batches of triples
  bw_labels = _single_iterator(0)
  inputs = _single_iterator(1)
  fw_labels = _single_iterator(2)
  
  dataset = tf.data.Dataset.zip((bw_labels, inputs, fw_labels))
  dataset = dataset.prefetch(1)
  return dataset.make_one_shot_iterator().get_next()


if __name__ == '__main__':
  print("Loading word vector model...")
  start = time.time()
  
  w2v_model = KeyedVectors.load(FLAGS.embeddings_path, mmap='r')
  
  duration = time.time() - start
  print("Done ({:0.4f}s).".format(duration))
  
  print("Building computational graph...")
  start = time.time()
  
  graph = tf.Graph()
  with graph.as_default():
    
    filenames = [os.path.join(FLAGS.input, f) for f in os.listdir(FLAGS.input)]
    iterator = train_iterator(filenames)
    
    # TODO: add hyperparameters from argparse
    m = SkipThoughts(w2v_model, train=iterator)
  
  duration = time.time() - start
  print("Done ({:0.4f}s).".format(duration))
  
  
  with tf.Session(graph=graph) as sess:
    saver = tf.train.Saver()
    output_dir = os.path.join('output', FLAGS.model_name)
    
    if not m.restore(output_dir):
      print("Initializing model...")
      start = time.time()
      sess.run(tf.global_variables_initializer())
      duration = time.time() - start
      print(
        "Initialized model at", output_dir,
        "({:0.4f}s).".format(duration))
      
      # Avoid crashes due to directory not existing.
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
      start = time.time()
      loss_, _ = sess.run([m.loss, m.train_op])
      duration = time.time() - start
      current_step = sess.run(m.global_step)
      print(
        "Step", current_step,
        "(loss={:0.4f}, time={:0.4f}s)".format(loss_, duration))

      if current_step % FLAGS.num_steps_per_save == 0:
        print("Saving model...")
        saver.save(
          sess,
          os.path.join('output', FLAGS.model_name, 'checkpoint.ckpt'),
          global_step=current_step)
