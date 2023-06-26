import tensorflow as tf
import tensorflow.contrib.keras as keras
import unittest
import numpy as np
from dataio.lshtc3_io2 import LSHTC3IOHelper
from scipy.sparse import csr_matrix
import time


def locally_connected_layer(input_tensor, kernel_initializer, bias_initializer, kernel_size=3, nb_filters=1):
    """
    
    :param input_tensor: tensor of shape [batchsize, n_features]
    :param kernel_initializer: shape [kernel_size, n_features], kernel_size should be even (3, 5, 7)
    :param bias_initializer: shape [1, n_features]
    :param kernel_size: kernel_size should be even (3, 5, 7)
    :param nb_filters: not used currently
    :return: output tensor of shape [batchsize, n_features]
    """

    with tf.name_scope('loc_connected'):

        # padd input
        zeros_to_padd = int((kernel_size - 1)/2)
        padded_input = tf.pad(input_tensor, [[0, 0], [zeros_to_padd, zeros_to_padd]], name='padding')

        # slice input vector and reshape it
        with tf.name_scope('input_slicing_reshaping'):
            x_aux = []
            for n in range(kernel_size-1):
                x_aux.append(tf.expand_dims(padded_input[:, n:-kernel_size+1+n], axis=1))
            x_aux.append(tf.expand_dims(padded_input[:, kernel_size-1:], axis=1))
            x_sliced = tf.concat(x_aux, axis=1)

        # define a variable w
        w = tf.Variable(initial_value=kernel_initializer, name='w')
        b = tf.Variable(initial_value=bias_initializer, name='b')

        #  reduce_sum (multiply w and sliced reshaped input), add bias
        output = tf.reduce_sum(tf.multiply(x_sliced, w), axis=1) + b

    return output


def randomly_connected_layer(input_tensor, kernel_initializer, bias_initializer, kernel_size=3,
                             shuffle_order=None, nb_filters=1):
    """
    Randomness is not inside the layer, but has to be created outside (through shuffle_order)
    
    :param input_tensor: tensor of shape [batchsize, n_features]
    :param kernel_initializer: shape [kernel_size, n_features], kernel_size should be even (3, 5, 7)
    :param bias_initializer: shape [1, n_features]
    :param kernel_size: kernel_size should be even (3, 5, 7)
    :param shuffle_order: [[in_order], [out_order]]
    :param nb_filters: not used currently
    :return: output tensor of shape [batchsize, n_features]
    """

    with tf.name_scope('randomly_connected'):
        shuffle_tensor = tf.constant(shuffle_order, name='shuffle_order_constant')
        shuffled_input = tf.gather(input_tensor, shuffle_tensor[0], axis=1)
        y = locally_connected_layer(
            input_tensor=shuffled_input,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_size=kernel_size,
            nb_filters=nb_filters)

        y_shuffled = tf.gather(y, shuffle_tensor[1], axis=1)

    return y_shuffled


class RandomlyConnectedLayer(keras.layers.Layer):

    def __init__(self, output_dim, kernel_size=3, kernel_initializer=None, bias_initializer=None,
                 shuffle_order=None, seed=None, **kwargs):
        """

        :param output_dim: has to be equal to input_shape[1] (input n_features)
        :param kernel_size:
        :param kernel_initializer:
        :param bias_initializer:
        :param shuffle_order:
        :param kwargs:
        """

        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.shuffle_order = shuffle_order
        self.w = None
        self.b = None
        self.shuffle_tensor = None
        self.seed = seed
        super(RandomlyConnectedLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variables for this layer.
        n_features = input_shape[1]
        assert n_features == self.output_dim

        if self.shuffle_order is None:
            self.shuffle_order = np.tile(np.arange(n_features.value), (2, 1))
            for irow in range(2):
                np.random.shuffle(self.shuffle_order[irow, :])
            self.shuffle_order.astype(np.int32)
        with tf.name_scope('randomly_connected'):
            self.shuffle_tensor = tf.constant(self.shuffle_order, name='shuffle_order_constant')

        if self.kernel_initializer is None:
            self.kernel_initializer = keras.initializers.glorot_normal(seed=self.seed)
        else:
            self.kernel_initializer = keras.initializers.Constant(value=self.kernel_initializer)

        if self.bias_initializer is None:
            self.bias_initializer = keras.initializers.Constant(value=np.zeros((1, n_features)))
        else:
            self.bias_initializer = keras.initializers.Constant(value=self.bias_initializer)

        self.w = self.add_weight(name='w', shape=(self.kernel_size, n_features), initializer=self.kernel_initializer,
                                 trainable=True)
        self.b = self.add_weight(name='b', shape=(1, n_features), initializer=self.bias_initializer, trainable=True)
        super(RandomlyConnectedLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):

        with tf.name_scope('randomly_connected'):
            shuffled_input = tf.gather(x, self.shuffle_tensor[0], axis=1)

            # padd input
            zeros_to_padd = int((self.kernel_size - 1)/2)
            padded_input = tf.pad(shuffled_input, [[0, 0], [zeros_to_padd, zeros_to_padd]], name='padding')

            # slice input vector and reshape it
            with tf.name_scope('input_slicing_reshaping'):
                x_aux = []
                for n in range(self.kernel_size - 1):
                    x_aux.append(tf.expand_dims(padded_input[:, n:-self.kernel_size+1+n], axis=1))
                x_aux.append(tf.expand_dims(padded_input[:, self.kernel_size-1:], axis=1))
                x_sliced = tf.concat(x_aux, axis=1)

            #  reduce_sum (multiply w and sliced reshaped input), add bias
            y = tf.reduce_sum(tf.multiply(x_sliced, self.w), axis=1) + self.b

            y_shuffled = tf.gather(y, self.shuffle_tensor[1], axis=1)

        return y_shuffled

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


# class TestLocallyConnectedLayerInitialization(unittest.TestCase):
#     def test_only_weights_central(self):
#         n = 6
#         x = np.expand_dims(np.arange(n, dtype='float32'), axis=0)
#         w_initial = np.vstack((np.zeros(n), np.ones(n), np.zeros(n))).astype('float32')
#         b_initial = np.zeros(n).astype('float32')
#
#         x_pl = tf.placeholder(dtype=x.dtype, shape=x.shape)
#         y = locally_connected_layer(
#             x_pl,
#             kernel_initializer=w_initial,
#             bias_initializer=b_initial)
#
#         init_op = tf.global_variables_initializer()
#
#         with tf.Session() as sess:
#             sess.run(init_op)
#             y_val = sess.run(y, feed_dict={x_pl: x})
#             print('y_val = {}'.format(y_val))
#
#         y_val_expected = x
#         self.assertTrue(np.array_equal(y_val, y_val_expected))
#
#     def test_only_weights_central_with_bias(self):
#         n = 6
#         x = np.expand_dims(np.arange(n, dtype='float32'), axis=0)
#         w_initial = np.vstack((np.zeros(n), np.ones(n), np.zeros(n))).astype('float32')
#         bias_initial = -np.ones(n).astype('float32')
#
#         x_pl = tf.placeholder(dtype=x.dtype, shape=x.shape)
#         y = locally_connected_layer(
#             x_pl,
#             kernel_initializer=w_initial,
#             bias_initializer=bias_initial)
#
#         init_op = tf.global_variables_initializer()
#
#         with tf.Session() as sess:
#             sess.run(init_op)
#             y_val = sess.run(y, feed_dict={x_pl: x})
#             print('y_val = {}'.format(y_val))
#
#         y_val_expected = x-1
#         print('y_val_expected = {}'.format(y_val_expected))
#         self.assertTrue(np.array_equal(y_val, y_val_expected))
#
#     def test_same_weights_with_bias(self):
#         n = 6
#         x = np.expand_dims(np.arange(n, dtype='float32'), axis=0)
#         w_initial = np.vstack((1*np.ones(n), 2*np.ones(n), 3*np.ones(n))).astype('float32')
#         bias_initial = -np.arange(n).astype('float32')
#
#         x_pl = tf.placeholder(dtype=x.dtype, shape=x.shape)
#         y = locally_connected_layer(
#             x_pl,
#             kernel_initializer=w_initial,
#             bias_initializer=bias_initial)
#
#         init_op = tf.global_variables_initializer()
#
#         with tf.Session() as sess:
#             sess.run(init_op)
#             y_val = sess.run(y, feed_dict={x_pl: x})
#             print('y_val = {}'.format(y_val))
#
#         y_val_expected = np.expand_dims(np.array([3, 7, 12, 17, 22, 9], dtype='float32'), axis=0)
#         print('y_val_expected = {}'.format(y_val_expected))
#         self.assertTrue(np.array_equal(y_val, y_val_expected))
#
#     def test_different_weights_at_different_positions(self):
#         n = 6
#         x = np.expand_dims(np.arange(n, dtype='float32'), axis=0)
#         w_initial = np.tile(np.arange(1, n+1), (3, 1)).astype('float32')
#         bias_initial = np.zeros(n).astype('float32')
#
#         x_pl = tf.placeholder(dtype=x.dtype, shape=x.shape)
#         y = locally_connected_layer(
#             x_pl,
#             kernel_initializer=w_initial,
#             bias_initializer=bias_initial)
#
#         init_op = tf.global_variables_initializer()
#
#         with tf.Session() as sess:
#             sess.run(init_op)
#             y_val = sess.run(y, feed_dict={x_pl: x})
#             print('y_val = {}'.format(y_val))
#
#         y_val_expected = np.expand_dims(np.array([1, 6, 18, 36, 60, 54], dtype='float32'), axis=0)
#         self.assertTrue(np.array_equal(y_val, y_val_expected))
#
#     def testBatchSizeGreaterThan1(self):
#         n = 6
#         batchsize = 5
#         x = np.tile(np.arange(n, dtype='float32'), (batchsize, 1))
#         x += (np.arange(batchsize, dtype='float32')[np.newaxis]).transpose()
#         w_initial = np.vstack((np.zeros(n), np.ones(n), np.zeros(n))).astype('float32')
#         b_initial = np.zeros(n).astype('float32')
#
#         x_pl = tf.placeholder(dtype=x.dtype, shape=x.shape)
#         y = locally_connected_layer(
#             x_pl,
#             kernel_initializer=w_initial,
#             bias_initializer=b_initial)
#
#         init_op = tf.global_variables_initializer()
#
#         with tf.Session() as sess:
#             sess.run(init_op)
#             y_val = sess.run(y, feed_dict={x_pl: x})
#             print('y_val = {}'.format(y_val))
#
#         y_val_expected = x
#         self.assertTrue(np.array_equal(y_val, y_val_expected))
#
#     def test_large_input(self):
#         n = 36505
#         x = np.expand_dims(np.arange(n, dtype='float32'), axis=0)
#         w_initial = np.vstack((np.zeros(n, dtype='float32'), np.ones(n, dtype='float32'), np.zeros(n, dtype='float32')))
#         b_initial = np.zeros(n, dtype='float32')
#
#         x_pl = tf.placeholder(dtype=x.dtype, shape=x.shape)
#         y = locally_connected_layer(
#             x_pl,
#             kernel_initializer=w_initial,
#             bias_initializer=b_initial)
#
#         init_op = tf.global_variables_initializer()
#
#         with tf.Session() as sess:
#             sess.run(init_op)
#             y_val = sess.run(y, feed_dict={x_pl: x})
#
#         y_val_expected = x
#         self.assertTrue(np.array_equal(y_val, y_val_expected))
#
#     def test_lshtc3_input(self):
#         h = LSHTC3IOHelper('/home/kadr/codes/kadrnn/nn_thresholding/nn_th_config.ini')
#         scores = h.load_scores_train_train()
#
#         x = scores[0, :].toarray().astype('float32')
#         n = x.shape[1]
#         w_initial = np.vstack((np.zeros(n, dtype='float32'), np.ones(n, dtype='float32'), np.zeros(n, dtype='float32')))
#         b_initial = np.zeros(n, dtype='float32')
#
#         x_pl = tf.placeholder(dtype=x.dtype, shape=x.shape)
#         y = locally_connected_layer(
#             x_pl,
#             kernel_initializer=w_initial,
#             bias_initializer=b_initial)
#
#         init_op = tf.global_variables_initializer()
#
#         with tf.Session() as sess:
#             sess.run(init_op)
#             y_val = sess.run(y, feed_dict={x_pl: x})
#             print('x = {}'.format(csr_matrix(x)))
#             print('y_val = {}'.format(csr_matrix(y_val)))
#
#         y_val_expected = x
#         self.assertTrue(np.array_equal(y_val, y_val_expected))
#
#
# class TestLocallyConnectedLayerTraining(unittest.TestCase):
#     def test_learnIdentity(self):
#
#         # data generation
#         n_train_samples = 1000
#         n_features = 10
#         x_data = np.random.rand(n_train_samples, n_features).astype('float32')
#         y_data = x_data
#
#         # training hyperparams
#         num_epochs = 250
#         batch_size = 32
#         seed = 1234
#
#         # Tell TensorFlow that the model will be built into the default Graph.
#         with tf.Graph().as_default():
#             with tf.name_scope('input'):
#                 # Input data
#                 inputs_initializer = tf.placeholder(
#                     dtype=x_data.dtype,
#                     shape=x_data.shape)
#                 labels_initializer = tf.placeholder(
#                     dtype=y_data.dtype,
#                     shape=y_data.shape)
#                 input_source = tf.Variable(
#                     inputs_initializer, trainable=False, collections=[])
#                 labels_source = tf.Variable(
#                     labels_initializer, trainable=False, collections=[])
#
#                 input_sliced, labels_sliced = tf.train.slice_input_producer(
#                     [input_source, labels_source], num_epochs=num_epochs)
#                 inputs, labels = tf.train.batch(
#                     [input_sliced, labels_sliced], batch_size=batch_size)
#
#             # Build a Graph that computes predictions from the inference model.
#
#             # initializers for model parameters:
#             kernel_size = 3
#             w_initializer = tf.truncated_normal(
#                 shape=[kernel_size, n_features],
#                 dtype=x_data.dtype,
#                 seed=seed)
#             b_initializer = tf.zeros(
#                 shape=[1, n_features],
#                 dtype=x_data.dtype)
#
#             output = locally_connected_layer(
#                 inputs,
#                 kernel_initializer=w_initializer,
#                 bias_initializer=b_initializer)
#
#             # Add to the Graph the Ops for loss calculation.
#             loss = tf.losses.mean_squared_error(labels=labels, predictions=output)
#
#             # Add to the Graph the Ops that calculate and apply gradients.
#             train_op = tf.train.AdamOptimizer().minimize(loss)
#
#             expected_w = tf.constant(
#                 value=np.vstack((np.zeros(n_features), np.ones(n_features), np.zeros(n_features))).astype('float32'),
#                 shape=[3, n_features],
#                 dtype='float32',
#                 name='expected_w')
#
#             current_w = tf.get_default_graph().get_tensor_by_name('loc_connected/w:0')
#             evaluate_op = tf.losses.mean_squared_error(tf.reshape(expected_w, [-1]), tf.reshape(current_w, [-1]))
#
#             # Build the summary operation based on the TF collection of Summaries.
#             summary_op = tf.summary.merge_all()
#
#             # Create the op for initializing variables.
#             init_op = tf.group(tf.global_variables_initializer(),
#                                tf.local_variables_initializer())
#
#             # Create a session for running Ops on the Graph.
#             sess = tf.Session()
#
#             # Run the Op to initialize the variables.
#             sess.run(init_op)
#             sess.run(input_source.initializer,
#                      feed_dict={inputs_initializer: x_data})
#             sess.run(labels_source.initializer,
#                      feed_dict={labels_initializer: y_data})
#
#             # Instantiate a SummaryWriter to output summaries and the Graph.
#             summary_writer = tf.summary.FileWriter('/tmp/tf', sess.graph)
#
#             # Start input enqueue threads.
#             coord = tf.train.Coordinator()
#             threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#             # And then after everything is built, start the training loop.
#             step = 0
#             try:
#                 while not coord.should_stop():
#                     start_time = time.time()
#
#                     # Run one step of the model.
#                     _, loss_value = sess.run([train_op, loss])
#
#                     duration = time.time() - start_time
#
#                     # Write the summaries and print an overview fairly often.
#                     if step % 100 == 0:
#                         diff = sess.run(evaluate_op)
#                         # Print status to stdout.
#                         print('Step %d: loss = %.2f, w_diff = %.4f (%.3f sec)' % (step, loss_value, diff, duration))
#                         # Update the events file.
#                         summary_str = sess.run(summary_op)
#                         summary_writer.add_summary(summary_str, step)
#                         step += 1
#
#                     step += 1
#             except tf.errors.OutOfRangeError:
#                 print('Done training for {} epochs, {} steps.'.format(num_epochs, step))
#             finally:
#                 # When done, ask the threads to stop.
#                 coord.request_stop()
#
#             # Wait for threads to finish.
#             coord.join(threads)
#
#             var = sess.graph.get_tensor_by_name('loc_connected/w:0')
#             w_values, mse_diff_val = sess.run([var, evaluate_op])
#             sess.close()
#
#         print('W_values:\n{}'.format(w_values))
#         rel_error = mse_diff_val/np.mean(w_values)
#         print('ralative difference between expected and actual learnt weights = {}'.format(rel_error))
#         self.assertTrue(rel_error < 0.03)
#
#
# class TestRandomlyConnectedLayerInitialization(unittest.TestCase):
#     def test_only_weights_central(self):
#         n = 6
#         # x = np.expand_dims(np.arange(n, dtype='float32'), axis=0)
#         x = np.random.rand(4, 6).astype('float32')
#         w_initial = np.vstack((np.zeros(n), np.ones(n), np.zeros(n))).astype('float32')
#         b_initial = np.zeros(n).astype('float32')
#         shuffle_order = np.array([[3, 2, 5, 1, 0, 4], [5, 1, 3, 2, 0, 4]], dtype='int32')
#
#         x_pl = tf.placeholder(dtype=x.dtype, shape=x.shape)
#         y = randomly_connected_layer(
#             input_tensor=x_pl,
#             kernel_initializer=w_initial,
#             bias_initializer=b_initial,
#             shuffle_order=shuffle_order)
#
#         init_op = tf.global_variables_initializer()
#
#         with tf.Session() as sess:
#             sess.run(init_op)
#             y_val = sess.run(y, feed_dict={x_pl: x})
#             print('y_val = {}'.format(y_val))
#
#         total_shuffle = np.arange(n)[shuffle_order[0]][shuffle_order[1]]
#         y_val_expected = x[:, total_shuffle]
#         self.assertTrue(np.array_equal(y_val, y_val_expected))
#
#     def test_weights_central_with_bias(self):
#         n = 6
#         # x = np.expand_dims(np.arange(n, dtype='float32'), axis=0)
#         x = np.random.rand(4, 6).astype('float32')
#         w_initial = np.vstack((np.zeros(n), np.ones(n), np.zeros(n))).astype('float32')
#         b_initial = -np.ones(n).astype('float32')
#         shuffle_order = np.array([[3, 2, 5, 1, 0, 4], [5, 1, 3, 2, 0, 4]], dtype='int32')
#
#         x_pl = tf.placeholder(dtype=x.dtype, shape=x.shape)
#         y = randomly_connected_layer(
#             input_tensor=x_pl,
#             kernel_initializer=w_initial,
#             bias_initializer=b_initial,
#             shuffle_order=shuffle_order)
#
#         init_op = tf.global_variables_initializer()
#
#         with tf.Session() as sess:
#             sess.run(init_op)
#             y_val = sess.run(y, feed_dict={x_pl: x})
#             print('y_val = {}'.format(y_val))
#
#         total_shuffle = np.arange(n)[shuffle_order[0]][shuffle_order[1]]
#         y_val_expected = x[:, total_shuffle] - 1.
#         self.assertTrue(np.array_equal(y_val, y_val_expected))
#
#     def test_random_order(self):
#         n = 6
#         x = np.arange(2*n, dtype='float32').reshape([2, -1])
#         w_initial = np.vstack((np.zeros(n), np.arange(n), np.zeros(n))).astype('float32')
#         b_initial = np.zeros(n).astype('float32')
#         shuffle_order = np.array([[3, 2, 5, 1, 0, 4], [5, 1, 3, 2, 0, 4]], dtype='int32')
#
#         x_pl = tf.placeholder(dtype=x.dtype, shape=x.shape)
#         y = randomly_connected_layer(
#             input_tensor=x_pl,
#             kernel_initializer=w_initial,
#             bias_initializer=b_initial,
#             shuffle_order=shuffle_order)
#
#         init_op = tf.global_variables_initializer()
#
#         with tf.Session() as sess:
#             sess.run(init_op)
#             y_val = sess.run(y, feed_dict={x_pl: x})
#             print('y_val = {}'.format(y_val))
#
#         y_val_expected = np.array([[20, 2, 3, 10, 0, 0], [50, 8, 21, 22, 0, 24]], dtype='float32')
#         self.assertTrue(np.array_equal(y_val, y_val_expected))
#
#     def test_random_order_with_bias(self):
#         n = 6
#         x = np.arange(2*n, dtype='float32').reshape([2, -1])
#         w_initial = np.vstack((np.zeros(n), np.arange(n), np.zeros(n))).astype('float32')
#         b_initial = -np.arange(n).astype('float32')
#         shuffle_order = np.array([[3, 2, 5, 1, 0, 4], [5, 1, 3, 2, 0, 4]], dtype='int32')
#
#         x_pl = tf.placeholder(dtype=x.dtype, shape=x.shape)
#         y = randomly_connected_layer(
#             input_tensor=x_pl,
#             kernel_initializer=w_initial,
#             bias_initializer=b_initial,
#             shuffle_order=shuffle_order)
#
#         init_op = tf.global_variables_initializer()
#
#         with tf.Session() as sess:
#             sess.run(init_op)
#             y_val = sess.run(y, feed_dict={x_pl: x})
#             print('y_val = {}'.format(y_val))
#
#         y_val_expected = np.array([[15, 1, 0, 8, 0, -4], [45, 7, 18, 20, 0, 20]], dtype='float32')
#         self.assertTrue(np.array_equal(y_val, y_val_expected))


# class TestRandomlyConnectedLayerTraining(unittest.TestCase):
#     def test_learnIdentity(self):
#
#         # data generation
#         n_train_samples = 1000
#         n_features = 10
#         x_data = np.random.rand(n_train_samples, n_features).astype('float32')
#         shuffle_order = np.vstack((np.random.permutation(n_features), np.random.permutation(n_features))).astype('int32')
#         total_shuffle = np.arange(n_features)[shuffle_order[0]][shuffle_order[1]]
#         y_data = x_data[:, total_shuffle]
#
#         # training hyperparams
#         num_epochs = 250
#         batch_size = 32
#         seed = 1234
#
#         # Tell TensorFlow that the model will be built into the default Graph.
#         with tf.Graph().as_default():
#             with tf.name_scope('input'):
#                 # Input data
#                 inputs_initializer = tf.placeholder(
#                     dtype=x_data.dtype,
#                     shape=x_data.shape)
#                 labels_initializer = tf.placeholder(
#                     dtype=y_data.dtype,
#                     shape=y_data.shape)
#                 input_source = tf.Variable(
#                     inputs_initializer, trainable=False, collections=[])
#                 labels_source = tf.Variable(
#                     labels_initializer, trainable=False, collections=[])
#
#                 input_sliced, labels_sliced = tf.train.slice_input_producer(
#                     [input_source, labels_source], num_epochs=num_epochs)
#                 inputs, labels = tf.train.batch(
#                     [input_sliced, labels_sliced], batch_size=batch_size)
#
#             # Build a Graph that computes predictions from the inference model.
#
#             # initializers for model parameters:
#             kernel_size = 3
#             w_initializer = tf.truncated_normal(
#                 shape=[kernel_size, n_features],
#                 dtype=x_data.dtype,
#                 seed=seed)
#             b_initializer = tf.zeros(
#                 shape=[1, n_features],
#                 dtype=x_data.dtype)
#
#             output = randomly_connected_layer(
#                 inputs,
#                 kernel_initializer=w_initializer,
#                 bias_initializer=b_initializer,
#                 shuffle_order=shuffle_order)
#
#             # Add to the Graph the Ops for loss calculation.
#             loss = tf.losses.mean_squared_error(labels=labels, predictions=output)
#
#             # Add to the Graph the Ops that calculate and apply gradients.
#             train_op = tf.train.AdamOptimizer().minimize(loss)
#
#             expected_w = tf.constant(
#                 value=np.vstack((np.zeros(n_features), np.ones(n_features), np.zeros(n_features))).astype('float32'),
#                 shape=[3, n_features],
#                 dtype='float32',
#                 name='expected_w')
#
#             current_w = tf.get_default_graph().get_tensor_by_name('randomly_connected/loc_connected/w:0')
#             evaluate_op = tf.losses.mean_squared_error(tf.reshape(expected_w, [-1]), tf.reshape(current_w, [-1]))
#
#             # Build the summary operation based on the TF collection of Summaries.
#             summary_op = tf.summary.merge_all()
#
#             # Create the op for initializing variables.
#             init_op = tf.group(tf.global_variables_initializer(),
#                                tf.local_variables_initializer())
#
#             # Create a session for running Ops on the Graph.
#             sess = tf.Session()
#
#             # Run the Op to initialize the variables.
#             sess.run(init_op)
#             sess.run(input_source.initializer,
#                      feed_dict={inputs_initializer: x_data})
#             sess.run(labels_source.initializer,
#                      feed_dict={labels_initializer: y_data})
#
#             # Instantiate a SummaryWriter to output summaries and the Graph.
#             summary_writer = tf.summary.FileWriter('/tmp/tf', sess.graph)
#
#             # Start input enqueue threads.
#             coord = tf.train.Coordinator()
#             threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#             # And then after everything is built, start the training loop.
#             step = 0
#             try:
#                 while not coord.should_stop():
#                     start_time = time.time()
#
#                     # Run one step of the model.
#                     _, loss_value = sess.run([train_op, loss])
#
#                     duration = time.time() - start_time
#
#                     # Write the summaries and print an overview fairly often.
#                     if step % 100 == 0:
#                         diff = sess.run(evaluate_op)
#                         # Print status to stdout.
#                         print('Step %d: loss = %.2f, w_diff = %.4f (%.3f sec)' % (step, loss_value, diff, duration))
#                         # Update the events file.
#                         summary_str = sess.run(summary_op)
#                         summary_writer.add_summary(summary_str, step)
#                         step += 1
#
#                     step += 1
#             except tf.errors.OutOfRangeError:
#                 print('Done training for {} epochs, {} steps.'.format(num_epochs, step))
#             finally:
#                 # When done, ask the threads to stop.
#                 coord.request_stop()
#
#             # Wait for threads to finish.
#             coord.join(threads)
#
#             var_w = sess.graph.get_tensor_by_name('randomly_connected/loc_connected/w:0')
#             var_b = sess.graph.get_tensor_by_name('randomly_connected/loc_connected/b:0')
#             w_values, b_values, mse_diff_val = sess.run([var_w, var_b, evaluate_op])
#             sess.close()
#
#         print('W_values:\n{}'.format(w_values))
#         print('b_values:\n{}'.format(b_values))
#
#         # now test this model
#         x = np.expand_dims(np.arange(10, dtype='float32'), axis=0)
#         w_initial = w_values
#         b_initial = b_values
#
#         g = tf.Graph()
#         with g.as_default():
#             x_pl = tf.placeholder(dtype=x.dtype, shape=x.shape)
#             y = randomly_connected_layer(
#                 input_tensor=x_pl,
#                 kernel_initializer=w_initial,
#                 bias_initializer=b_initial,
#                 shuffle_order=shuffle_order)
#             init_op = tf.global_variables_initializer()
#
#         with tf.Session(graph=g) as sess2:
#             sess2.run(init_op)
#             y_val = sess2.run(y, feed_dict={x_pl: x})
#             print('y_val = {}'.format(y_val))
#
#         y_val_expected = np.expand_dims(total_shuffle, axis=0)
#
#         print('y_val_expected = {}'.format(y_val_expected))
#         mse_y_diff = np.mean(np.square(y_val - y_val_expected))
#         print('MSE y = {}'.format(mse_y_diff))
#         self.assertTrue(mse_y_diff < 0.01)


class TestRandomlyConnectedKerasLayerInitialization(unittest.TestCase):
    def test_only_weights_central(self):
        n = 6
        # x = np.expand_dims(np.arange(n, dtype='float32'), axis=0)
        x = np.random.rand(4, n).astype('float32')
        w_initial = np.vstack((np.zeros(n), np.ones(n), np.zeros(n))).astype('float32')
        b_initial = np.zeros(n).astype('float32')
        shuffle_order = np.array([[3, 2, 5, 1, 0, 4], [5, 1, 3, 2, 0, 4]], dtype='int32')

        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(n,)))
        model.add(RandomlyConnectedLayer(output_dim=n, kernel_size=3, kernel_initializer=w_initial,
                                         bias_initializer=b_initial, shuffle_order=shuffle_order))

        y_val = model.predict(x)

        total_shuffle = np.arange(n)[shuffle_order[0]][shuffle_order[1]]
        y_val_expected = x[:, total_shuffle]
        self.assertTrue(np.array_equal(y_val, y_val_expected))

    def test_weights_central_with_bias(self):
        n = 6
        # x = np.expand_dims(np.arange(n, dtype='float32'), axis=0)
        x = np.random.rand(4, 6).astype('float32')
        w_initial = np.vstack((np.zeros(n), np.ones(n), np.zeros(n))).astype('float32')
        b_initial = -np.ones(n).astype('float32')
        shuffle_order = np.array([[3, 2, 5, 1, 0, 4], [5, 1, 3, 2, 0, 4]], dtype='int32')

        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(n,)))
        model.add(RandomlyConnectedLayer(output_dim=n, kernel_size=3, kernel_initializer=w_initial,
                                         bias_initializer=b_initial, shuffle_order=shuffle_order))

        y_val = model.predict(x)

        total_shuffle = np.arange(n)[shuffle_order[0]][shuffle_order[1]]
        y_val_expected = x[:, total_shuffle] - 1.
        self.assertTrue(np.array_equal(y_val, y_val_expected))

    def test_random_order(self):
        n = 6
        x = np.arange(2*n, dtype='float32').reshape([2, -1])
        w_initial = np.vstack((np.zeros(n), np.arange(n), np.zeros(n))).astype('float32')
        b_initial = np.zeros(n).astype('float32')
        shuffle_order = np.array([[3, 2, 5, 1, 0, 4], [5, 1, 3, 2, 0, 4]], dtype='int32')

        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(n,)))
        model.add(RandomlyConnectedLayer(output_dim=n, kernel_size=3, kernel_initializer=w_initial,
                                         bias_initializer=b_initial, shuffle_order=shuffle_order))

        y_val = model.predict(x)

        y_val_expected = np.array([[20, 2, 3, 10, 0, 0], [50, 8, 21, 22, 0, 24]], dtype='float32')
        self.assertTrue(np.array_equal(y_val, y_val_expected))

    def test_random_order_with_bias(self):
        n = 6
        x = np.arange(2*n, dtype='float32').reshape([2, -1])
        w_initial = np.vstack((np.zeros(n), np.arange(n), np.zeros(n))).astype('float32')
        b_initial = -np.arange(n).astype('float32')
        shuffle_order = np.array([[3, 2, 5, 1, 0, 4], [5, 1, 3, 2, 0, 4]], dtype='int32')

        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(n,)))
        model.add(RandomlyConnectedLayer(output_dim=n, kernel_size=3, kernel_initializer=w_initial,
                                         bias_initializer=b_initial, shuffle_order=shuffle_order))

        y_val = model.predict(x)

        y_val_expected = np.array([[15, 1, 0, 8, 0, -4], [45, 7, 18, 20, 0, 20]], dtype='float32')
        self.assertTrue(np.array_equal(y_val, y_val_expected))

    def test_some_random_order(self):
        n = 6
        x = np.arange(2*n, dtype='float32').reshape([2, -1])
        w_initial = np.vstack((np.zeros(n), np.ones(n), np.zeros(n))).astype('float32')

        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(n,)))
        myLayer = RandomlyConnectedLayer(output_dim=n, kernel_size=3, kernel_initializer=w_initial)
        model.add(myLayer)

        y_val = model.predict(x)

        shuffle_order = myLayer.shuffle_order
        total_shuffle = np.arange(n)[shuffle_order[0]][shuffle_order[1]]

        y_val_expected = x[:, total_shuffle]
        self.assertTrue(np.array_equal(y_val, y_val_expected))


# class TestRandomlyConnectedKerasLayerTraining(unittest.TestCase):
#     def test_learnIdentity(self):
#         n_train_samples = 1000
#         n_features = 10
#         x_data = np.random.rand(n_train_samples, n_features).astype('float32')
#         shuffle_order = np.vstack((np.random.permutation(n_features), np.random.permutation(n_features))).astype('int32')
#         total_shuffle = np.arange(n_features)[shuffle_order[0]][shuffle_order[1]]
#         y_data = x_data[:, total_shuffle]
#
#         # training hyperparams
#         num_epochs = 300
#         batch_size = 64
#         seed = 1234
#
#         model = keras.models.Sequential()
#         model.add(keras.layers.InputLayer(input_shape=(n_features,)))
#         model.add(RandomlyConnectedLayer(output_dim=n_features, kernel_size=3, shuffle_order=shuffle_order, seed=seed))
#
#         model.compile(optimizer='adam', loss='mse')
#
#         class MyCallback(keras.callbacks.Callback):
#             def __init__(self, w_expected):
#                 self.w_expected = w_expected
#                 super(MyCallback, self).__init__()
#
#             def on_epoch_end(self, epoch, logs=None):
#                 if epoch % 10 == 0:
#                     model = self.model
#                     w_actual = model.get_weights()[0]
#                     mse = np.mean((w_actual - self.w_expected) ** 2)
#                     print('Epoch {}: w_mse = {}'.format(epoch, mse))
#
#         w_expected = np.vstack((np.zeros(n_features), np.ones(n_features), np.zeros(n_features))).astype('float32')
#         clbs = [MyCallback(w_expected=w_expected)]
#         model.fit(x_data, y_data, batch_size=batch_size, epochs=num_epochs, callbacks=clbs)
#
#         y_val = model.predict(x_data)
#         data_mse = np.mean((y_val - x_data[:, total_shuffle])**2)
#         print(data_mse)
#         self.assertTrue(data_mse < 1e-5)


if __name__ == '__main__':
    # # suite = unittest.TestLoader().loadTestsFromTestCase(TestLocallyConnectedLayerInitialization)
    # # suite = unittest.TestLoader().loadTestsFromTestCase(TestLocallyConnectedLayerTraining)
    # # suite = unittest.TestLoader().loadTestsFromTestCase(TestRandomlyConnectedLayerInitialization)
    # # suite = unittest.TestLoader().loadTestsFromTestCase(TestRandomlyConnectedLayerTraining)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRandomlyConnectedKerasLayerInitialization)
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestRandomlyConnectedKerasLayerTraining)
    unittest.TextTestRunner().run(suite)
