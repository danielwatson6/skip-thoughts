"""Script for evaluating a trained sentence embedding model."""

# Putting this first allows to run on a headless device.
import matplotlib
matplotlib.use('Agg')

import argparse
from itertools import starmap
import os
import re
import sys

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

import models


parser = argparse.ArgumentParser()

parser.add_argument(
  '--model', type=str, default='one_hot',
  help="Set to any of the ./models/* files.")

parser.add_argument(
  '--model_name', type=str, default="default",
  help="Will save/restore model in ./output/[model_name].")

parser.add_argument(
  '--embeddings_path', type=str, default="word2vecModel",
  help="Path to the pre-trained word embeddings model.")

parser.add_argument(
  '--interactive', type=bool, default=False,
  help="Set to prompt for and encode sentences.")

FLAGS = parser.parse_args()


def cosine_similarity(vec1, vec2):
  """Get the cosine similarity of the two given vectors."""
  return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def accuracy(model_scores, gold_scores):
  """Get the number of correct guesses / number of total guesses."""
  disc_scores = map(lambda x: int(x > .5), model_scores)
  diffs = starmap(lambda x, y: int(x - y == 0), zip(disc_scores, gold_scores))
  return sum(diffs) / len(model_scores)


def mean_squared_similarity(model_scores, gold_scores):
  """Get 1 - mean squared error between the system and the gold."""
  diffs = starmap(lambda x, y: (x - y) ** 2, zip(model_scores, gold_scores))
  return 1. - sum(diffs) / len(model_scores)


def plot_sick(model_scores, gold_scores):
  """Save a correlation plot between the model and gold similarity scores."""
  f, ax = plt.subplots(figsize=(6, 6))
  plt.plot(model_scores, gold_scores, 'ko', markersize=1)
  ax.set(xlim=(0, 1), ylim=(0, 1))
  ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")  # diagonal line
  f.savefig(os.path.join(save_dir, "sick_correlation.pdf"))


def evaluate_sick(model):
  """Evaluate the model against the SICK dataset."""
  model_scores = []
  gold_scores = []
  with open('data/sick/SICK.txt') as f:
    for i, line in enumerate(f):
      # The first line of the dataset has the column names, not examples.
      if i == 0:
        continue

      line = line.split('\t')
      vec1, vec2 = model.encode_sentences(line[1:3])
      model_scores.append(cosine_similarity(vec1, vec2))
      # Squeeze [1-5] |-> [0-1].
      gold_scores.append(.25 * (float(line[4]) - 1))

  print(
    "SICK", "PCC", pearsonr(model_scores, gold_scores)[0], sep='\t')
  print(
    "SICK", "SCC", spearmanr(model_scores, gold_scores)[0], sep='\t')
  return model_scores, gold_scores


def evaluate_msr(model):
  """Evaluate the model against the MSR dataset."""
  model_scores = []
  gold_scores = []
  for filename in ['msr_train.txt', 'msr_test.txt']:
    with open(os.path.join('data', 'msr', filename)) as f:
      for i, line in enumerate(f):
        # The first line of the dataset has the column names, not examples.
        if i == 0:
          continue

        line = line.split('\t')
        gold_scores.append(int(line[0]))
        vec1, vec2 = model.encode_sentences(line[3:5])
        model_scores.append(cosine_similarity(vec1, vec2))

  print(
    "MSR", "ACC", accuracy(model_scores, gold_scores), sep='\t')
  print(
    "MSR", "MSS", mean_squared_similarity(model_scores, gold_scores), sep='\t')
  return model_scores, gold_scores


MODELS = {
  'one_hot': models.OneHot,
  'cbow': models.CBOW,
  'sent2vec': models.Sent2Vec,
  'skip_thoughts': models.SkipThoughts,
  'vae': models.VAE,
}


if FLAGS.model not in MODELS:
  raise ValueError("`{}` is not a valid model.".format(FLAGS.model))

kwargs = {'model_name': FLAGS.model_name}

# Load word2vec files if the model uses embeddings and hasn't been saved.
save_dir = os.path.join('output', FLAGS.model_name)
model_saved = os.path.exists(save_dir) and \
              os.path.exists(os.path.join(save_dir, 'checkpoint.ckpt'))
uses_embeddings = issubclass(
  MODELS[FLAGS.model], models.base_models.WordVectorModel)

if uses_embeddings and not model_saved:
  print("Loading word vector model...")
  kwargs['w2v_model'] = KeyedVectors.load(FLAGS.embeddings_path, mmap='r')


print("Building", FLAGS.model, "model `{}`".format(FLAGS.model_name))
graph = tf.Graph()
with graph.as_default():
  kwargs['input_placeholder'] = tf.placeholder(tf.int64, shape=[2, None])
  model = MODELS[FLAGS.model](**kwargs)


with tf.Session(graph=graph) as sess:
  if not model.start():
    # Model did not exist. Save any associated weights.
    model.save(verbose=False)

  if FLAGS.interactive:
    while True:
      s1 = input("Type a sentence: ")
      s2 = input("Type a sentence: ")
      # print(sess.run(model.encode, feed_dict={
      #   model.input_placeholder: [[15172 + 4, 0], [15172 + 4, 0]] }))

      vec1, vec2 = model.encode_sentences([s1, s2])
      print("Cosine similarity:", cosine_similarity(vec1, vec2))

  else:
    # Evaluation on the SICK dataset
    print("Encoding SICK sentences...")
    model_scores, gold_scores = evaluate_sick(model)
    plot_sick(model_scores, gold_scores)

    # Evaluation on the PPDB dataset
    print("Encoding MSR sentences...")
    model_scores, gold_scores = evaluate_msr(model)
