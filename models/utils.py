"""Miscellaneous utilities for building models."""

import tensorflow as tf


def mask_embeddings(inputs_tensor, embeds_tensor):
  """Gien a 3D tensor, zero out the embeddings corresponding to padding."""
  sequence_mask = tf.cast(tf.sign(inputs_tensor), tf.float32)
  # TODO: is there a more efficient way to weight out the vectors?
  transposed_embeds = tf.transpose(embeds_tensor, perm=[2, 0, 1])
  return tf.transpose(sequence_mask * transposed_embeds, perm=[1, 2, 0])
