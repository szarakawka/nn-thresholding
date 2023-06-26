import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging


# Taken from: https://github.com/tensorflow/tensorflow/issues/6646
def embedding_lookup_sparse(params, sp_mat,
                            name=None,
                            combiner=None,
                            max_norm=None):
    """Computes embeddings for the given ids and weights.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_mat (i.e. there are no rows with empty features, if so,
  put 0.0 in sp_mat entry), and that all the indices of sp_mat are in
  canonical row-major order.

  It also assumes that all id values lie in the range [0, p0), where p0
  is the sum of the size of params along dimension 0.

  Args:
    params: A single tensor representing the complete embedding tensor,
      or a list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.  Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    sp_mat: N x M SparseTensor of zero or non-zero weights,
      where N is typically batch size and M is the embedding table size.
    name: Optional name for the op.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
      and "sum" are supported.
      "sum" computes the weighted sum of the embedding results for each row.
      "mean" is the weighted sum divided by the total weight.
      "sqrtn" is the weighted sum divided by the square root of the sum of the
      squares of the weights.
    max_norm: If not None, each embedding is normalized to have l2 norm equal
      to max_norm before combining.

  Returns:
    A dense tensor representing the combined embeddings for the sparse ids.
    For each row in the dense tensor represented by sp_mat, the op looks up
    the embeddings for all (non-zero) ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if

      shape(combined params) = [p0, p1, ..., pm]

    and

      shape(sp_mat) = [d0, d1, ..., dn]

    then

      shape(output) = [d0, d1, ..., dn-1, p1, ..., pm].

    For instance, if params is a 10x20 matrix, and sp_mat is

      [0, 0]: 1.0
      [0, 1]: 3.0
      [1, 0]: 0.0
      [2, 3]: 1.0

    with `combiner`="mean", then the output will be a 3x20 matrix where

      output[0, :] = (params[0, :] * 1.0 + params[1, :] * 3.0) / (1.0 + 3.0)
      output[1, :] = params[0, :] * 0.0 / div_protect
      output[2, :] = params[3, :] * 1.0 / 1.0

  Raises:
    TypeError: If sp_mat is not a SparseTensor.
    ValueError: If combiner is not one of {"mean", "sqrtn", "sum"}.
  """
    if combiner is None:
        logging.warn("The default value of combiner will change from \"mean\" "
                     "to \"sqrtn\" after 2016/11/01.")
        combiner = "mean"
    if combiner not in ("mean", "sqrtn", "sum"):
        raise ValueError("combiner must be one of 'mean', 'sqrtn' or 'sum'")
    if isinstance(params, variables.PartitionedVariable):
        params = list(params)  # Iterate to get the underlying Variables.
    if not isinstance(params, list):
        params = [params]
    if not isinstance(sp_mat, sparse_tensor.SparseTensor):
        raise TypeError("sp_mat must be SparseTensor")

    with ops.name_scope(name, "embedding_lookup_sparse",
                        params + [sp_mat]) as name:
        segment_ids = sp_mat.indices[:, 0]
        if segment_ids.dtype != dtypes.int32:
            segment_ids = math_ops.cast(segment_ids, dtypes.int32)

        ids = sp_mat.indices[:, 1]

        embeddings = tf.nn.embedding_lookup(
            params, ids, max_norm=max_norm)

        weights = sp_mat.values
        if weights.dtype != embeddings.dtype:
            weights = math_ops.cast(weights, embeddings.dtype)

        # Reshape weights to allow broadcast
        ones = array_ops.fill(
            array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0), 1)
        bcast_weights_shape = array_ops.concat_v2(
            [array_ops.shape(weights), ones], 0)

        orig_weights_shape = weights.get_shape()
        weights = array_ops.reshape(weights, bcast_weights_shape)

        # Set the weight shape, since after reshaping to bcast_weights_shape,
        # the shape becomes None.
        if embeddings.get_shape().ndims is not None:
            weights.set_shape(orig_weights_shape.concatenate(
                [1 for _ in range(embeddings.get_shape().ndims - 1)]))

        embeddings *= weights

        div_protect = 1e-32  # would not work for float16 or float8
        if combiner == "sum":
            embeddings = math_ops.segment_sum(embeddings, segment_ids, name=name)
        elif combiner == "mean":
            embeddings = math_ops.segment_sum(embeddings, segment_ids)
            weight_sum = math_ops.segment_sum(weights, segment_ids)
            embeddings = math_ops.div(embeddings, weight_sum + div_protect, name=name)
        elif combiner == "sqrtn":
            embeddings = math_ops.segment_sum(embeddings, segment_ids)
            weights_squared = math_ops.pow(weights, 2)
            weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
            weight_sum_sqrt = math_ops.sqrt(weight_sum)
            embeddings = math_ops.div(embeddings, weight_sum_sqrt + div_protect, name=name)
        else:
            assert False, "Unrecognized combiner"

        return embeddings


class SparseEmbeddingLookup(layers.Layer):

    def __init__(self, input_size=None, embed_size=None, initial_embeddings=None, trainable_embeddings=True,
                 name=None):
        """

        :param input_size:
        :param embed_size:
        :param initial_embeddings: of size [input_size, embed_size]
        """
        super().__init__(trainable=trainable_embeddings, name=name, dtype=tf.float32)

        if initial_embeddings is not None:
            # embeddings = initial_embeddings
            self.input_size = initial_embeddings.shape[0]
            self.embed_size = initial_embeddings.shape[1]
            if input_size is not None and input_size != self.input_size:
                raise AssertionError('initial_embeddings.shape[0] must be equal to '
                                     'input_size (or leave input_size unspecified)')
            if embed_size is not None and embed_size != self.embed_size:
                raise AssertionError('initial_embeddings.shape[1] must be equal to embed_size '
                                     '(or leave input_size unspecified)')
        elif input_size is not None and embed_size is not None:
            self.input_size = input_size
            self.embed_size = embed_size
            emb_init = tf.random_uniform_initializer()
            initial_embeddings = emb_init(shape=(input_size, embed_size), dtype=tf.float32)
        else:
            raise AssertionError('input_size and embed_size OR initial_embeddings params must be passed')
        # self.embeddings = self.add_weight(name='embeddings', initializer=initial_embeddings,
        #                                   trainable=trainable_embeddings)
        self.embeddings = tf.Variable(initial_value=initial_embeddings, dtype=tf.float32,
                                      trainable=trainable_embeddings)

    def call(self, inputs, **kwargs):
        """ inputs is a SparseVector input"""
        return embedding_lookup_sparse(self.embeddings, inputs, name='effective_embedding', combiner='sum')

    def get_config(self):
        config = super().get_config()
        config.update({'input_size': self.input_size})
        config.update({'embed_size': self.embed_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
