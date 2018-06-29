"""Module defining the `CBOW` model class."""

import tensorflow as tf

from models.base_models import WordVectorModel
from models.utils import mask_embeddings


class CBOW(WordVectorModel):
  """Continuous bag of words model (https://arxiv.org/pdf/1301.3781).

  Hyperparameters:
    mask_sequences: whether to exclude padding tokens from calculations.

  For additional hyperparameters, refer to parent class.
  """

  def build_encoder(self, input_batch):
    # The parent class builds the word embedding matrix.
    super().build_encoder(input_batch)

    embeddings = self.embeddings
    reduction_axis = int(not self.hparams['time_major'])

    if self.hparams['mask_sequences']:
      embeddings = mask_embeddings(input_batch, embeddings)

    return tf.reduce_sum(embeddings, axis=reduction_axis)
