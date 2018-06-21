"""Module defining the skip-thoughts model."""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq


class SkipThoughts:
  """Skip-thoughts model.
  
  Instantiating this class builds the computational graph Note that this model
  expects input sequences with the following reserved special token ids:
    0: padding token,
    1: out-of-vocabulary token,
    2: start-of-string token,
    3: end-of-string token.
  
  Input sequences are expected to already be padded and have the end-of-string
  token added (necessary for labels, optional for inputs; regard it as a
  hyperparameter to the model). The start-of-string token is added
  automatically as part of the computational graph during training."""
  
  def __init__(self, w2v_model, train=None, vocabulary_size=20000,
    batch_size=16, output_size=512, embedding_size=300,
    max_sequence_length=40, lr=1e-3, sample_prob=0., max_grad_norm=10.,
    train_special_embeddings=False, train_word_embeddings=False, concat=False):
    """Build the computational graph.
    
    Args:
      w2v_model: a word2vec model instance.
    
    Keyword args:
      train: either None (default), or `iterator.get_next()` where `iterator`
             is an instance of `tf.data.Iterator` that yields 3-tuples in the
             order bw_label, input, fw_label.
      
      TODO: complete documentation
    """
    
    self.learning_rate = tf.get_variable(
      "learning_rate", shape=[], trainable=False,
      initializer=tf.initializers.constant(learning_rate))
    
    self.sample_prob = tf.get_variable(
      "sample_prob", shape=[], trainable=False,
      initializer=tf.initializers.constant(sample_prob))
    
    self.global_step = tf.get_variable(
      "global_step", shape=[], trainable=False,
      initializer=tf.intializers.zeros())
    
    self.w2v_model = w2v_model
    self._max_sequence_length = max_sequence_length
    self._max_grad_norm = max_grad_norm
    
    # Embedding matrices
    
    special_embeddings = tf.get_variable(
      "special_embeddings", shape=[4, embedding_size],
      initializer=tf.random_uniform_initializer(-sqrt3, sqrt3),
      trainable=train_special_embeddings)
    
    word_embeddings = tf.get_variable(
      "word_embeddings", shape=[vocabulary_size, embedding_size],
      initializer=tf.constant_initializer(w2v_model.syn0[:vocabulary_size]),
      trainable=train_word_embeddings)
    
    self._embeddings = tf.concat([special_embeddings, word_embeddings], 0)
    
    # RNN cells
    RNNCell = tf.contrib.rnn.GRUBlockCell
    self._fw_cell = RNNCell(self.output_size, name="fw_cell")
    self._bw_cell = RNNCell(self.output_size, name="bw_cell")
    self._dec_cell = RNNCell(self.output_size, name="dec_cell")
    
    # Softmax layer
    self.output_layer = tf.layers.Dense(vocabulary_size, name="output_layer")
    
    # Training
    if train is not None:
      
      # Unpack iterator ops
      bw_labels, train_inputs, fw_labels = train
      
      # Encoder
      thought = self._thought(train_inputs)
      
      # Forward and backward decoders
      fw_logits = self._decoder(thought, fw_labels, self._fw_cell)
      bw_logits = self._decoder(thought, bw_labels, self._bw_cell)
      
      # Loss
      fw_mask = tf.cast(tf.sign(fw_labels), tf.float32)
      bw_mask = tf.cast(tf.sign(bw_labels), tf.float32)
      self.loss = seq2seq.sequence_loss(fw_logits, fw_labels, fw_mask) + \
                  seq2seq.sequence_loss(bw_logits, bw_labels, bw_mask)
      
      # Optimizer with gradient clipping
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(
        tf.gradients(self.loss, tvars), self._max_grad_norm)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=self.global_step)
    
    # Inference
    else:
      self._inputs = tf.placeholder(
        tf.int64, shape=[None, max_sequence_length], name="inference_inputs")
      self._get_thought = self._thought(self._inputs)
  
  
  def _get_embeddings(self, query):
    """Internally used for embedding lookup."""
    return tf.nn.embedding_lookup(self._embeddings, query)
  
  
  def _thought(self, inputs):
    """Internally used to run the model agnostic to input feeding method."""
    sequence_length = tf.reduce_sum(tf.sign(inputs), reduction_indices=1)
    
    rnn_output = tf.nn.bidirectional_dynamic_rnn(
      self._fw_cell, self._bw_cell, self._get_embeddings(inputs),
      sequence_length=sequence_length, dtype=tf.float32)
    
    if self.concat:
      return tf.concat(rnn_output[1], 0)
    return sum(rnn_output[1])
  
  
  def _decoder(thought, labels, rnn_cell):
    """Internally used to build a decoder RNN."""

    # Scheduled sampling with constant probability. Labels are shifted to the
    # right by adding a start-of-string token.
    sos_tokens = tf.tile([[2]], [self.batch_size, 1])
    shifted_labels = tf.concat([sos_tokens, labels[::-1]], 1)
    
    decoder_in = self._get_embeddings(shifted_labels)
    max_seq_lengths = tf.tile([self._max_sequence_length], [self.batch_size])
    helper = seq2seq.ScheduledEmbeddingTrainingHelper(
      decoder_in, max_seq_lengths, self._get_embeddings, self.sample_prob)
    
    # Final layer for both decoders that converts decoder output to 
    decoder = seq2seq.BasicDecoder(
      rnn_cell, helper, thought, output_layer=self.output_layer)
    return seq2seq.dynamic_decode(decoder)[0].rnn_output
  
  
  def _sequence(sentence):
    """Interally used to convert strings to integer sequences."""
    words = sentence.split()
    seq = []
    # Compensate for start-of-string and end-of-string tokens.
    for word in words[:FLAGS.max_length - 2]:
      id_to_append = 1  # unknown word (id: 1)
      if word in w2v_model:
        # Add 4 to compensate for the special seq2seq tokens.
        word_id = self.w2v_model.vocab[word].index + 4
        if word_id < FLAGS.vocabulary_size:
          id_to_append = word_id
      seq.append(id_to_append)
    return seq
  
  
  def encode(self, sentences):
    """Runs the encoder op on a list of sentences (sentence strings)."""
    sess = tf.get_default_session()
    sequences = list(map(self._sequence, sentences))
    return sess.run(self._get_thought, feed_dict={self._inputs: sequences})
