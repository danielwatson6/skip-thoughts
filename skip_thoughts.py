"""Module defining the skip-thoughts model."""

import time

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
    batch_size=16, output_size=512, max_sequence_length=40, learning_rate=1e-3,
    max_grad_norm=10., concat=False, optimizer=None, softmax_samples=0,
    train_special_embeddings=False, train_word_embeddings=False,
    time_major=False, cuda=False):
    """Build the computational graph.
    
    Args:
      w2v_model: a word2vec model instance.
    
    Keyword args:
      train: either None (default), or `iterator.get_next()` where `iterator`
             is an instance of `tf.data.Iterator` that yields 3-tuples in the
             order bw_label, input, fw_label.
      
      TODO: complete documentation
    """
    
    # Internally used attributes
    self._w2v_model = w2v_model
    self._vocabulary_size = vocabulary_size
    self._batch_size = batch_size
    self._output_size = output_size
    self._max_sequence_length = max_sequence_length
    self._max_grad_norm = max_grad_norm
    self._concat = concat
    self._softmax_samples = softmax_samples
    self._time_major = time_major
    self._cuda = cuda
    self._embedding_size = w2v_model.vector_size
    
    self.learning_rate = tf.get_variable(
      "learning_rate", shape=[], trainable=False,
      initializer=tf.initializers.constant(learning_rate))
    
    self.global_step = tf.get_variable(
      "global_step", shape=[], trainable=False,
      initializer=tf.initializers.zeros())
    
    # Embedding matrices. The special embeddings are initialized with a mean 0
    # and variance 1 random uniform distribution.
    special_embeddings = tf.get_variable(
      "special_embeddings", shape=[4, self._embedding_size],
      initializer=tf.initializers.random_uniform(-np.sqrt(3), np.sqrt(3)),
      trainable=train_special_embeddings)
    
    word_embeddings = tf.get_variable(
      "word_embeddings", shape=[vocabulary_size, self._embedding_size],
      initializer=tf.initializers.constant(w2v_model.syn0[:vocabulary_size]),
      trainable=train_word_embeddings)
    
    self._embeddings = tf.concat([special_embeddings, word_embeddings], 0)
    
    # Softmax layer
    self._output_layer = tf.layers.Dense(vocabulary_size, name="output_layer")
    # Call this to be able to obtain the layer's weights at any given moment.
    self._output_layer.build(output_size)
    
    # Training
    if train is not None:
      
      # Unpack iterator ops
      bw_labels, train_inputs, fw_labels = tf.unstack(train, num=3)
      
      # Encoder
      self._get_thought = self._thought(train_inputs)
      
      # Forward and backward decoders
      fw_logits = self._decoder(self._get_thought, fw_labels)
      bw_logits = self._decoder(self._get_thought, bw_labels)
      
      # Loss
      self.loss = self._loss(fw_logits, fw_labels) + \
                  self._loss(bw_logits, bw_labels)
      
      # Optimizer with gradient clipping
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(
        tf.gradients(self.loss, tvars), self._max_grad_norm)
      if optimizer is None:
        optimizer = tf.train.AdamOptimizer
      self.train_op = optimizer(self.learning_rate).apply_gradients(
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
    encoder_in = self._get_embeddings(inputs)
    
    if self._cuda:
      rnn = tf.contrib.cudnn_rnn.CudnnGRU(
        1, self._output_size, direction='bidirectional')
      rnn_output = tf.unstack(rnn(encoder_in)[1][0])
    
    else:
      fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self._output_size)
      bw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self._output_size)
      sequence_length = tf.reduce_sum(
        tf.sign(inputs), reduction_indices=int(not self._time_major))
      rnn_output = tf.nn.bidirectional_dynamic_rnn(
        fw_cell, bw_cell, encoder_in, sequence_length=sequence_length,
        dtype=tf.float32, time_major=self._time_major)[1]

    if self._concat:
      return tf.concat(rnn_output, 0)
    return sum(rnn_output)
 

  def _decoder(self, thought, labels):
    """Internally used to build a decoder RNN."""

    # Labels are shifted to the right by adding a start-of-string token.
    if self._time_major:
      sos_tokens = tf.constant([[2] * self._batch_size], dtype=tf.int64)
      shifted_labels = tf.concat([sos_tokens, labels[:-1]], 0)
    else:
      sos_tokens = tf.constant([[2]] * self._batch_size, dtype=tf.int64)
      shifted_labels = tf.concat([sos_tokens, labels[:,:-1]], 1)
    
    decoder_in = self._get_embeddings(shifted_labels)

    if self._cuda:
      decoder_out = tf.contrib.cudnn_rnn.CudnnGRU(
        1, self._output_size, direction='unidirectional')(decoder_in)[0]
    
    else:
      rnn_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self._output_size)
      max_seq_lengths = tf.constant(
        [self._max_sequence_length] * self._batch_size)
      helper = seq2seq.TrainingHelper(
        decoder_in, max_seq_lengths, time_major=self._time_major)
      decoder = seq2seq.BasicDecoder(rnn_cell, helper, thought)
      decoder_out = seq2seq.dynamic_decode(
        decoder, output_time_major=self._time_major)[0].rnn_output
    
    return decoder_out
  
  
  def _loss(self, rnn_outputs, labels):
    """Get a properly masked loss for the given logits and labels."""
    if not self._softmax_samples:
      mask = tf.cast(tf.sign(labels), tf.float32)
      logits = self._output_layer(rnn_outputs)
      return seq2seq.sequence_loss(logits, labels, mask)
    
    # Sampled softmax for improved performance.
    rnn_outputs = tf.reshape(rnn_outputs, [-1, self._output_size])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(tf.sign(labels), tf.float32)
    
    weights, biases = self._output_layer.trainable_weights
    losses = tf.nn.sampled_softmax_loss(
      tf.transpose(weights), biases, tf.expand_dims(labels, axis=1),
      rnn_outputs, self._softmax_samples, self._vocabulary_size)
    
    return tf.reduce_mean(mask * losses)  # Hadamard product
  
  
  def _sequence(self, sentence):
    """Interally used to convert strings to integer sequences."""
    words = sentence.split()
    seq = []
    # Compensate for start-of-string and end-of-string tokens.
    for word in words[:FLAGS.max_length - 2]:
      id_to_append = 1  # unknown word (id: 1)
      if word in w2v_model:
        # Add 4 to compensate for the special seq2seq tokens.
        word_id = self._w2v_model.vocab[word].index + 4
        if word_id < FLAGS.vocabulary_size:
          id_to_append = word_id
      seq.append(id_to_append)
    return seq
  
  
  def restore(self, save_dir, verbose=True):
    """Attempt to restore the model's weights from the given directory.
    
    Returns True or False depending on success."""
    sess = tf.get_default_session()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(save_dir)
    if not ckpt:
      if verbose:
        print("Failed to restore model at {}.".format(save_dir))
      return False
    if verbose:
      print("Restoring model...")
    start = time.time()
    saver.restore(sess, ckpt.model_checkpoint_path)
    duration = time.time() - start
    if verbose:
      print(
        "Restored model at step", sess.run(self.global_step),
        "({:0.4f}s).".format(duration))
    return True
  
  
  def encode(self, sentences):
    """Run the encoder op on a list of sentences (sentence strings)."""
    sess = tf.get_default_session()
    sequences = list(map(self._sequence, sentences))
    return sess.run(self._get_thought, feed_dict={self._inputs: sequences})

