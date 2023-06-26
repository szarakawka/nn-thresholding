import math
import time

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from utils import etc


class BatchIdx:
    def __init__(self, n_examples, batch_size):
        self.n_examples = n_examples
        self.batch_size = batch_size
        self.n_batches = self.n_examples // batch_size
        if self.n_examples % batch_size != 0:
            self.n_batches += 1

    def start_idx(self, i_batch):
        if i_batch >= self.n_batches:
            raise ValueError(
                'Too big i_batch ({}), should be smaller than number of batches ({} in this case.).'.format(
                    i_batch, self.n_batches))
        else:
            return i_batch*self.batch_size

    def end_idx(self, i_batch):
        if i_batch >= self.n_batches:
            raise ValueError(
                'Too big i_batch ({}), should be smaller than number of batches ({} in this case.).'.format(
                    i_batch, self.n_batches))
        elif i_batch == (self.n_batches - 1):
            return self.n_examples
        else:
            return (i_batch+1)*self.batch_size


class BatchedData:
    def __init__(self, data, batch_size):
        self.data = data
        self.batched_idx = BatchIdx(data.shape[0], batch_size)
        self.current_batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch == self.batched_idx.n_batches:
            raise StopIteration
        else:
            start_idx = self.batched_idx.start_idx(self.current_batch)
            end_idx = self.batched_idx.end_idx(self.current_batch)
            self.current_batch += 1
            return self.data[start_idx:end_idx]

    def __getitem__(self, item):
        start_idx = self.batched_idx.start_idx(item)
        end_idx = self.batched_idx.end_idx(item)
        return self.data[start_idx:end_idx]

    def __len__(self):
        return self.batched_idx.n_batches


class Data:
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def get_part(self, startidx, stopidx=None):
        stopidx = self.X.shape[0] if stopidx is None else stopidx
        if self.y is not None:
            return Data(self.X[startidx:stopidx], self.y[startidx:stopidx])
        else:
            return Data(self.X[startidx:stopidx])


class DoerBase:
    def __init__(self, output_dim_size, n_outputs=1):
        """

        :param output_dim_size: the size of an output per one data example
        :param n_outputs: how many results (size of a returned tuple) per data example is returned
        """
        self.output_dim_size = output_dim_size
        self.n_outputs = n_outputs

    def do(self, data):
        """

        :param data: Data class instance
        :return: result of the computation (numpy or scipy.sparse arrays) of shape (data.shape[0], output_dim_size)
        """
        pass


def do_in_minibatches_sparse(doer, data, batchsize=1000, verbose=1, end_char='') -> csr_matrix:
    """

    :param doer: instance of Doer class
    :param data: instance of Data class
    :param batchsize:
    :param verbose:
    :param end_char:
    :return:
    """

    t0 = time.time()

    if data.X.shape[0] <= batchsize:
        if verbose > 0:
            print('Full batch')
        return doer.do(data)

    if doer.n_outputs > 1:
        result = [lil_matrix((data.X.shape[0], doer.output_dim_size), dtype='float') for n in range(doer.n_outputs)]
    else:
        result = lil_matrix((data.X.shape[0], doer.output_dim_size), dtype='float')

    n_batches = math.floor(data.X.shape[0] / batchsize)

    # all full mini-batches
    for i_batch, pi in etc.pi_range(n_batches):
        if verbose > 0 and pi:
            print("\rBatch={}/{}, Progress={:.2f}, elapsed time={:.2f}, ETA={:.2f}".format(
                i_batch, n_batches, pi.progress, pi.elapsed_time, pi.time_remaining()), end=end_char)
            elapsed_time = pi.elapsed_time
        begin_idx = i_batch*batchsize
        end_idx = (i_batch + 1) * batchsize
        if doer.n_outputs > 1:
            partial_results = doer.do(data.get_part(begin_idx, end_idx))
            for n, part_result in enumerate(partial_results):
                result[n][begin_idx:end_idx] = part_result
        else:
            result[begin_idx:end_idx] = doer.do(data.get_part(begin_idx, end_idx))

    # last not-full mini-batch
    begin_idx = n_batches*batchsize
    if begin_idx < data.X.shape[0]:
        if verbose > 0:
            print("\rBatch={}/{}, elapsed time={:.2f}".format(
                n_batches, n_batches, time.time() - t0), end=end_char)
        if doer.n_outputs > 1:
            partial_results = doer.do(data.get_part(begin_idx))
            for n, part_result in enumerate(partial_results):
                result[n][begin_idx:] = part_result
        else:
            result[begin_idx:] = doer.do(data.get_part(begin_idx))

    if verbose > 0:
        print("")  # adds new line

    # convert to CSR
    if doer.n_outputs > 1:
        result_csr = [r.tocsr() for r in result]
        return tuple(result_csr)
    else:
        return result.tocsr()


def do_in_minibatches_dense(doer, data, batchsize=1000, verbose=1, end_char=''):

    t0 = time.time()

    if data.X.shape[0] <= batchsize:
        if verbose > 0:
            print('Full batch')
        return doer.do(data)

    if doer.n_outputs > 1:
        result = [np.empty((data.X.shape[0], doer.output_dim_size), dtype='float') for n in range(doer.n_outputs)]
    else:
        result = np.empty((data.X.shape[0], doer.output_dim_size), dtype='float')

    n_batches = math.floor(data.X.shape[0] / batchsize)

    # all full mini-batches
    for i_batch, pi in etc.pi_range(n_batches):
        if verbose > 0 and pi:
            print("\rBatch={}/{}, Progress={:.2f}, elapsed time={:.2f}, ETA={:.2f}".format(
                i_batch, n_batches, pi.progress, pi.elapsed_time, pi.time_remaining()), end=end_char)
            elapsed_time = pi.elapsed_time
        begin_idx = i_batch*batchsize
        end_idx = (i_batch + 1) * batchsize
        if doer.n_outputs > 1:
            partial_results = doer.do(data.get_part(begin_idx, end_idx))
            for n, part_result in enumerate(partial_results):
                result[n][begin_idx:end_idx, :] = part_result
        else:
            result[begin_idx:end_idx, :] = doer.do(data.get_part(begin_idx, end_idx))

    # last not-full mini-batch
    begin_idx = n_batches*batchsize
    if begin_idx < data.X.shape[0]:
        if verbose > 0:
            print("\rBatch={}/{}, elapsed time={:.2f}".format(
                n_batches, n_batches, time.time() - t0), end=end_char)
        if doer.n_outputs > 1:
            partial_results = doer.do(data.get_part(begin_idx))
            for n, part_result in enumerate(partial_results):
                result[n][begin_idx:, :] = part_result
        else:
            result[begin_idx:, :] = doer.do(data.get_part(begin_idx))

    if verbose > 0:
        print("")  # adds new line

    return result


def do_in_minibatches_void(doer, data, batchsize=1000, verbose=1, end_char=''):

    t0 = time.time()

    if data.X.shape[0] <= batchsize:
        if verbose > 0:
            print('Full batch')
        return doer.do(data)

    n_batches = math.floor(data.X.shape[0] / batchsize)

    # all full mini-batches
    for i_batch, pi in etc.pi_range(n_batches):
        if verbose > 0 and pi:
            print("\rBatch={}/{}, Progress={:.2f}, elapsed time={:.2f}, ETA={:.2f}".format(
                i_batch, n_batches, pi.progress, pi.elapsed_time, pi.time_remaining()), end=end_char)
            elapsed_time = pi.elapsed_time
        begin_idx = i_batch*batchsize
        end_idx = (i_batch + 1) * batchsize
        doer.do(data.get_part(begin_idx, end_idx))

    # last not-full mini-batch
    begin_idx = n_batches*batchsize
    if begin_idx < data.X.shape[0]:
        if verbose > 0:
            print("\rBatch={}/{}, elapsed time={:.2f}".format(
                n_batches, n_batches, time.time() - t0), end=end_char)
        doer.do(data.get_part(begin_idx))

    if verbose > 0:
        print("")  # adds new line
