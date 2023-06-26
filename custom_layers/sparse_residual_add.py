import tensorflow as tf
from tensorflow.keras import layers


class SparseResidualAdd(layers.Layer):
    """
        Called on sparse inputs of size [batch_size, n_input_feats], adds (with broadcast) a dense vector
        [1, n_input_feats] producing a dense output [batch_size, n_input_feats]

    """

    def __init__(self, dense_vector_to_add, **kwargs):
        """
        :param dense_vector_to_add: nd.array [1, n_input_feats], constant, not trainable
        """
        super().__init__(**kwargs)
        self.dense_vector = tf.constant(dense_vector_to_add, dtype=tf.float32, name='dense_vector',
                                        shape=(1, len(dense_vector_to_add)))
        self.add_layer = layers.Add(name='add_layer')

    def call(self, inputs, **kwargs):
        """ inputs is a list of two elements [sparse_inputs, dense_inputs], both of shape [batch_size, n_input_feats]"""
        dense_sum = tf.sparse.add(inputs[0], inputs[1])
        return self.add_layer([dense_sum, self.dense_vector])       # Add supports broadcasting

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)
