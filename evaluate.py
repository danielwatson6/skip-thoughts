"""Script for evaluating a trained skip-thoughts model."""

import os
import re
import sys

import numpy as np
from numpy.linalg import norm
import tensorflow as tf
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

from skip_thoughts import SkipThoughts


def get_features(vec1, vec2):
  """Following Kiros et al., get the cos similarity and difference norm."""
  return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2)), norm(vec1 - vec2)


word2vec_model = KeyedVectors.load('word2vecModel', mmap='r')
graph = tf.Graph()
with graph.as_default():
  model = SkipThoughts(word2vec_model,
    vocabulary_size=100000, batch_size=2, output_size=512)


with tf.Session(graph=graph):
  model.restore(sys.argv[1])
  
  # Evaluation on the SICK dataset
  features = []
  gold_scores = []
  with open('data/ppdb/sick/SICK.txt') as f:
    for i, line in enumerate(f):
      if i == 0:
        continue
      line = line.split('\t')
      vec1, vec2 = model.encode(line[1:3])
      features.append(get_features(vec1, vec2))
      # Squeeze [1-5] |-> [0-1]
      gold_scores.append(.25 * (float(line[4]) - 1))
  # Train / test split
  train_inputs = np.array(features[:5000])
  train_labels = np.array(gold_scores[:5000])
  test_inputs = np.array(features[5000:])
  test_labels = np.array(gold_scores[5000:])
  # Plot distributions
  f, ax = plt.subplots(figsize=(6, 6))
  plt.plot(train_inputs[:,0], test_labels, 'bo')
  ax.set(xlim=(0, 1), ylim=(0, 1))
  ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")  # diagonal line
  f.savefig(os.path.join(sys.argv[1], "distro.pdf"), bbox_inches='tight')
