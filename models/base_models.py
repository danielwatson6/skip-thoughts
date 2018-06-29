"""Module containing all abstractions used by the sentence embedding models."""

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq


class BaseModel:
  """Abstract base class for sentence embedding models.

  Instantiating any child class will build a computational graph.
  """

  def __init__(self, model_name='default', hparams=None, input_iterator=None,
               input_placeholder=None):
    """Constructor.

    Initializes the following attributes:
      hparams: ,
      global_step: ,
      encode: .

    Keyword arguments:
      model_name: will read/write in the directory "output/[model_name]",
      hparams: dictionary containing the initial hyperparameters. If the file
        "output/[model_name]/hparams.json" exists, this can be left empty and
        the model will use those hyperparameters instead,
      input_iterator: a TensorFlow operation that yields inputs; typically the
        returned op from `iterator.get_next()`, where `iterator` is an instance
        of `tf.data.Iterator`,
      input_placeholder: a TensorFlow placeholder to be used when no input
        iterator is provided.
    """
    self.hparams = hparams
    self.input_iterator = input_iterator
    self.input_placeholder = input_placeholder

    # Used internally.
    self._model_name = model_name

    self.global_step = tf.get_variable(
      "global_step", shape=[], trainable=False, dtype=tf.int64,
      initializer=tf.initializers.zeros())

    # Load hyperparameters from JSON file if it exists, otherwise save them.
    output_path = os.path.join('output', model_name)
    hparams_path = os.path.join(output_path, 'hparams.json')
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    if not os.path.exists(hparams_path):
      with open(hparams_path, 'w') as f:
        json.dump(self.hparams, f)
    else:
      with open(hparams_path) as f:
        self.hparams = json.load(f)
    try:
      self.parse_hparams()
    except KeyError:
      raise ValueError("The model is missing hyperparameter(s).")

    # Use input pipeline if available, otherwise use a placeholder.
    if input_iterator is not None:
      self.encode = self.build_encoder(input_iterator)
    elif input_placeholder is not None:
      self.encode = self.build_encoder(input_placeholder)
    else:
      raise ValueError("At least one of `input_iterator`, `input_placeholder` "
                       "must be supplied to the constructor.")

  def parse_hparams(self):
    """Parse the hyperparameters passed to the model."""
    pass

  def build_encoder(self, input_batch):
    """Get the sentence vectors for the given input integer id sequences."""
    raise NotImplementedError

  def to_sequence(self, sentence):
    """Convert a sentence string to an integer sequence."""
    raise NotImplementedError

  def save(self, verbose=True):
    """Attempt to save the model's weights."""
    sess = tf.get_default_session()
    if verbose:
      print("Saving model...")

    tf.train.Saver(max_to_keep=1).save(
      sess, os.path.join('output', self._model_name, 'checkpoint.ckpt'),
      global_step=self.global_step)

  def start(self, verbose=True):
    """Restore the model's weights, otherwise initialize them.

    Returns a boolean, True if the model was restored successfully.
    """
    sess = tf.get_default_session()
    output_path = os.path.join('output', self._model_name)
    if verbose:
      print("Restoring model...")

    ckpt = tf.train.get_checkpoint_state(output_path)
    if not ckpt:
      if verbose:
        print("Failed to restore model at {}.".format(output_path))
      # Initialize the model's weights.
      sess.run(tf.global_variables_initializer())
      return False

    # Restore the model's weights.
    tf.train.Saver(max_to_keep=1).restore(sess, ckpt.model_checkpoint_path)
    if verbose:
      print(
        "Restored model at step", sess.run(self.global_step))
    return True

  def encode_sentences(self, sentences):
    """Run the encoder op on a list of sentences (sentence strings)."""
    raise NotImplementedError


class WordVectorModel(BaseModel):
  """Abstract base class for models using pre-trained word representations.

  Note that any such model expects input sequences with the following reserved
  special token ids:
    0: padding token,
    1: out-of-vocabulary token,
    2: start-of-sentence token,
    3: end-of-sentence token.
  """

  def __init__(self, w2v_model=None, **kwargs):
    """Constructor.

    Initializes the following attributes:
      embeddings: a word embedding matrix.

    Keyword arguments:
      w2v_model: a gensim word2vec model instance. The embeddings will be saved
        in the model's parameters, so this can be left empty when restoring.

    For additional keyword arguments, refer to parent class.
    """
    self._w2v_model = w2v_model

    super().__init__(**kwargs)

  def parse_hparams(self):
    """Parse the hyperparameters passed to the model.

    Hyperparameters:
      vocabulary_size: ,
      batch_size: ,
      max_sequence_length: ,
      train_special_embeddings: ,
      train_word_embeddings: ,
      eos_token: ,
      time_major: .
    """
    # Add 4 to compensate for all the reserved special tokens.
    self.hparams['vocabulary_size'] += 4

  def build_encoder(self, inputs):
    """Build the word embedding matrix, but expect override."""
    # Special embeddings initialized with random uniform (mean 0, variance 1).
    special_embeddings = tf.get_variable(
      "special_embeddings", shape=[4, self._w2v_model.vector_size],
      initializer=tf.initializers.random_uniform(-np.sqrt(3), np.sqrt(3)),
      trainable=self.hparams['train_special_embeddings'])

    word_voc = self.hparams['vocabulary_size'] - 4
    word_embeddings = tf.get_variable(
      "word_embeddings", shape=[word_voc, self._w2v_model.vector_size],
      initializer=tf.initializers.constant(self._w2v_model.syn0[:word_voc]),
      trainable=self.hparams['train_word_embeddings'])

    self.embedding_matrix = tf.concat([special_embeddings, word_embeddings], 0)
    self.embeddings = tf.nn.embedding_lookup(self.embedding_matrix, inputs)

  def to_sequence(self, sentence):
    words = sentence.split()
    seq = []

    # Compensate for start-of-sentence and end-of-sentence tokens.
    for word in words[:self.hparams['max_sequence_length'] - 2]:
      id_to_append = 1  # unknown word (id: 1)
      if word in self._w2v_model:
        # Shift the vocabulary indices by 4 to preserve special tokens.
        word_id = self._w2v_model.vocab[word].index + 4
        if word_id < self.hparams['vocabulary_size']:
          id_to_append = word_id
      seq.append(id_to_append)

    # Append the end-of-sentence token (id: 3)
    if self.hparams['eos_token']:
      seq.append(3)

    # Pad the sequence (id: 0)
    while len(seq) < self.hparams['max_sequence_length']:
      seq.append(0)
    return seq

  def build_sequence_with_sos(self, sequences):
    """Build new sequences beginning with start-of-sentence tokens.

    The input must be a 2D tensor with dimensions for batch size and max
    sequence length (or in reverse, if time major).
    """
    batch_size = self.hparams['batch_size']
    if self.hparams['time_major']:
      sos_tokens = tf.constant([[2] * batch_size], dtype=tf.int64)
      return tf.concat([sos_tokens, sequences[:-1]], 0)

    sos_tokens = tf.constant([[2]] * batch_size, dtype=tf.int64)
    return tf.concat([sos_tokens, sequences[:, :-1]], 1)

  def encode_sentences(self, sentences):
    """Run the encoder op on a list of sentences (sentence strings)."""
    sess = tf.get_default_session()
    sequences = np.array(list(map(self.to_sequence, sentences)))
    if self.hparams['time_major']:
      sequences = sequences.T
    return sess.run(self.encode, feed_dict={self.input_placeholder: sequences})


class EncoderDecoderModel(WordVectorModel):
  """Abstract base class for encoder-decoder models that require training.

  Note that to ditinguish between training and inference, this class relies
  on `input_iterator` being passed. In this case, the input iterator will
  include labels, so an extra overridable method is provided to separate them.

  Hyperparameters:
    output_size: ,
    encoder_depth: ,
    softmax_samples: ,
    concat: .

  For additional hyperparameters, refer to parent class.
  """

  def __init__(self, cuda=False, **kwargs):
    """Constructor.

    Keyword arguments:
      cuda: whether to use CUDA optimized RNN ops. Only supported for devices
        with a CUDA-capable GPU.

    For additional keyword arguments, refer to parent class.
    """
    self._cuda = cuda
    super().__init__(**kwargs)

    # Final feedforward layer that reshapes embeddings to vocabulary size.
    self.output_layer = tf.layers.Dense(self.hparams['vocabulary_size'])
    # Call this to be able to obtain the layer's weights at any given moment.
    self.output_layer.build(self.hparams['output_size'])

  def unpack_iterator(self):
    """Separate the input iterator into (input, label(s))."""
    return (self.input_iterator,)

  def build_encoder(self, inputs):
    if self.input_iterator is not None:
      inputs = self.unpack_iterator()[0]
    super().build_encoder(inputs)

    output_size = self.hparams['output_size']

    if self._cuda:
      rnn = tf.contrib.cudnn_rnn.CudnnGRU(
        self.hparams['encoder_depth'], output_size, direction='bidirectional')
      rnn_output = tf.unstack(rnn(self.embeddings)[1][0])

    else:
      rnn_output = self.embeddings
      sequence_length = tf.reduce_sum(
        tf.sign(inputs),
        reduction_indices=int(not self.hparams['time_major']))

      for _ in range(self.hparams['encoder_depth']):
        fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(output_size)
        bw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(output_size)
        rnn_output = tf.nn.bidirectional_dynamic_rnn(
          fw_cell, bw_cell, rnn_output, sequence_length=sequence_length,
          dtype=tf.float32, time_major=self.hparams['time_major'])[1]

    if self.hparams['concat']:
      return tf.concat(rnn_output, 0)
    return sum(rnn_output)

  def build_decoder(self, encoded_sentence):
    """Build a decoder RNN using the encoded sentence as an initial state."""
    raise NotImplementedError

  def build_loss(self, rnn_outputs, labels):
    """Get a properly masked loss for the given logits and labels.

    If the `sotfmax_samples` hyperparameter is set to a positive value,
    this builds the appropriate sampled softmax loss.
    """
    if not self.hparams['softmax_samples']:
      mask = tf.cast(tf.sign(labels), tf.float32)
      logits = self.output_layer(rnn_outputs)
      return seq2seq.sequence_loss(logits, labels, mask)

    # Sampled softmax for improved performance.
    rnn_outputs = tf.reshape(rnn_outputs, [-1, self.hparams['output_size']])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(tf.sign(labels), tf.float32)

    weights, biases = self.output_layer.trainable_weights
    losses = tf.nn.sampled_softmax_loss(
      tf.transpose(weights), biases, tf.expand_dims(labels, axis=1),
      rnn_outputs, self.hparams['softmax_samples'],
      self.hparams['voabulary_size'])

    return tf.reduce_mean(mask * losses)  # Hadamard product
