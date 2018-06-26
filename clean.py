"""Skip-thought dataset preprocessor.

Every file in the input directory will be parsed to a TFRecord file. This file
will contain a cleaned version of the data (sentences are extracted; only
alphanumerics and apostrophes are kept), where the words are represented by
integer ids according to a given word2vec model."""

import argparse
import os
import re
import string

import tensorflow as tf
from gensim.models import KeyedVectors
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument('--vocabulary_size', type=int, default=20000,
  help="Keep only the n most common words of the training data.")
parser.add_argument('--max_length', type=int, default=40,
  help="Truncate input and output sentences to maximum length n.")
parser.add_argument('--input', type=str, default="data/books",
  help="Path to the directory containing the text files.")
parser.add_argument('--output', type=str, default="data/books_tf",
  help="Path to the directory that will contain the TFRecord files.")
parser.add_argument('--embeddings_path', type=str, default="./word2vecModel",
  help="Path to the pre-trained word embeddings model.")

FLAGS = parser.parse_args()


# TODO: optimize this to not read the entire file at once.
def sentences(s):
  """Convert a string of text to a list of cleaned sentences."""
  result = []
  for sentence in s.split('.'):
    sentence = re.sub(r"[^A-Za-z0-9 ']", " ", sentence)
    sentence = re.sub(r"[ ]+", " ", sentence).strip()
    result.append(sentence)
  return result


def sequence(s, w2v_model):
  """Get a `tf.SequenceExample` id sequence from a sentence string."""
  words = s.split()
  seq = tf.train.SequenceExample()
  fl_tokens = seq.feature_lists.feature_list["tokens"]
  # Compensate for start-of-string and end-of-string tokens.
  for word in words[:FLAGS.max_length - 2]:
    id_to_append = 1  # unknown word (id: 1)
    if word in w2v_model:
      # Add 4 to compensate for the special seq2seq tokens.
      word_id = w2v_model.vocab[word].index + 4
      if word_id < FLAGS.vocabulary_size:
        id_to_append = word_id
    fl_tokens.feature.add().int64_list.value.append(id_to_append)
  return seq


if __name__ == '__main__':
  if not os.path.exists(FLAGS.output):
    os.makedirs(FLAGS.output)
  
  print("Loading word vector model...")
  w2v_model = KeyedVectors.load(FLAGS.embeddings_path, mmap='r')
  
  print("Cleaning data...")
  for filename in tqdm(os.listdir(FLAGS.input)):
    
    with open(os.path.join(FLAGS.input, filename)) as f:
      contents = sentences(f.read())
    
    with open(os.path.join(FLAGS.output, filename), 'w') as f:
      writer = tf.python_io.TFRecordWriter(f.name)
      for seq in map(lambda s: sequence(s, w2v_model), contents):
        writer.write(seq.SerializeToString())
      writer.close()
