# This code is based on repo (but heavily modified):
# https://github.com/eggie5/NCE-loss

import tensorflow as tf


class NCE(tf.keras.layers.Layer):
    """
    Keras layer implementing NCE loss with learned unigram candidate sampler. Built on tf functions realizing those.
    """
    def __init__(self, num_classes, neg_samples=100, unique_negs=True, **kwargs):

        self.num_classes = num_classes
        self.neg_samples = neg_samples
        self.unique = unique_negs
        self.W = None
        self.b = None

        super(NCE, self).__init__(**kwargs)

    # keras Layer interface
    def build(self, input_shape):

        self.W = self.add_weight(
            name="approx_softmax_weights",
            shape=(self.num_classes, input_shape[0][1].value),
            initializer="glorot_normal",
        )

        self.b = self.add_weight(
            name="approx_softmax_biases", shape=(self.num_classes,), initializer="zeros"
        )

        # keras
        super(NCE, self).build(input_shape)

    # keras Layer interface
    def call(self, inputs, **kwargs):
        layer_inputs, targets = inputs

        sv_tuple = tf.nn.learned_unigram_candidate_sampler(targets, num_true=1, num_sampled=self.neg_samples,
                                                           unique=self.unique, range_max=self.num_classes)

        loss_per_example = tf.nn.nce_loss(self.W, self.b, targets, layer_inputs, self.neg_samples, self.num_classes,
                                          sampled_values=sv_tuple)

        loss = tf.reduce_sum(loss_per_example)

        self.add_loss(loss)         # !!!! crucially important !!!!

        logits = tf.matmul(layer_inputs, tf.transpose(self.W))
        logits = tf.nn.bias_add(logits, self.b)
        outputs = tf.nn.sigmoid(logits)

        return outputs

    # keras Layer interface
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.num_classes
        return tuple(output_shape)
