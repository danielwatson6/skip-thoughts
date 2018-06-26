"""Script for evaluating a trained sentence embedding model."""

# Putting this first allows to run on a headless device.
import matplotlib
matplotlib.use("Agg")

import os
import re
import sys

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from tqdm import tqdm

from skip_thoughts import SkipThoughts


def similarity(vec1, vec2):
  """Cosine similarity."""
  return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


word2vec_model = KeyedVectors.load('word2vecModel', mmap='r')
graph = tf.Graph()
with graph.as_default():
  model = SkipThoughts(word2vec_model,
    vocabulary_size=100000, batch_size=2, output_size=512, cuda=True)


with tf.Session(graph=graph):
  model.restore(sys.argv[1])
  
  # Evaluation on the SICK dataset
  print("Encoding sentences...")
  model_scores = []
  gold_scores = []
  with open('data/sick/SICK.txt') as f:
    for i, line in tqdm(enumerate(f)):
      if i == 0:
        continue
      
      line = line.split('\t')
      vec1, vec2 = model.encode(line[1:3])
      model_scores.append(similarity(vec1, vec2))
      # Squeeze [1-5] |-> [0-1]
      gold_scores.append(.25 * (float(line[4]) - 1))

  # Plot distributions
  f, ax = plt.subplots(figsize=(6, 6))
  plt.plot(model_scores, gold_scores, 'bo')
  minx = min(model_scores)
  ax.set(xlim=(minx, 1), ylim=(0, 1))
  ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")  # diagonal line
  f.savefig(os.path.join(sys.argv[1], "scatter_plot.pdf"), bbox_inches='tight')
  
  # Print correlation coefficients and MSE
  print("Pearson correlation:", pearsonr(model_scores, gold_scores)[0])
  print("Spearman correlation:", spearmanr(model_scores, gold_scores)[0])
