import unittest
from scipy.sparse import csr_matrix
from utils.batch_work import *


class BatchIdxNExamples7TastCase(unittest.TestCase):
    def setUp(self):
        self.array = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21]
        ])

    def test_n_batches_batch_size_1(self):
        bs = 1
        batch_idx = BatchIdx(self.array.shape[0], batch_size=bs)

        expected = 7
        actual = batch_idx.n_batches
        self.assertEqual(expected, actual)

    def test_n_batches_batch_size_3(self):
        bs = 3
        batch_idx = BatchIdx(self.array.shape[0], batch_size=bs)

        expected = 3
        actual = batch_idx.n_batches
        self.assertEqual(expected, actual)

    def test_start_idx_batch_size_1_ibatch0(self):
        bs = 1
        ibatch = 0
        batch_idx = BatchIdx(self.array.shape[0], batch_size=bs)

        expected = 0
        actual = batch_idx.start_idx(ibatch)
        self.assertEqual(expected, actual)

    def test_start_idx_batch_size_1_ibatch6(self):
        bs = 1
        ibatch = 6
        batch_idx = BatchIdx(self.array.shape[0], batch_size=bs)

        expected = 6
        actual = batch_idx.start_idx(ibatch)
        self.assertEqual(expected, actual)

    def test_start_idx_batch_size_1_ibatch7(self):
        bs = 1
        ibatch = 7
        batch_idx = BatchIdx(self.array.shape[0], batch_size=bs)
        with self.assertRaises(ValueError):
            batch_idx.start_idx(ibatch)

    def test_end_idx_batch_size_1_ibatch0(self):
        bs = 1
        ibatch = 0
        batch_idx = BatchIdx(self.array.shape[0], batch_size=bs)

        expected = 1
        actual = batch_idx.end_idx(ibatch)
        self.assertEqual(expected, actual)

    def test_end_idx_batch_size_1_ibatch6(self):
        bs = 1
        ibatch = 6
        batch_idx = BatchIdx(self.array.shape[0], batch_size=bs)

        expected = 7
        actual = batch_idx.end_idx(ibatch)
        self.assertEqual(expected, actual)

    def test_end_idx_batch_size_1_ibatch7(self):
        bs = 1
        ibatch = 7
        batch_idx = BatchIdx(self.array.shape[0], batch_size=bs)
        with self.assertRaises(ValueError):
            batch_idx.end_idx(ibatch)

    def test_start_idx_batch_size_3_ibatch0(self):
        bs = 3
        ibatch = 0
        batch_idx = BatchIdx(self.array.shape[0], batch_size=bs)

        expected = 0
        actual = batch_idx.start_idx(ibatch)
        self.assertEqual(expected, actual)

    def test_start_idx_batch_size_3_ibatch2(self):
        bs = 3
        ibatch = 2
        batch_idx = BatchIdx(self.array.shape[0], batch_size=bs)

        expected = 6
        actual = batch_idx.start_idx(ibatch)
        self.assertEqual(expected, actual)

    def test_start_idx_batch_size_3_ibatch3(self):
        bs = 3
        ibatch = 3
        batch_idx = BatchIdx(self.array.shape[0], batch_size=bs)
        with self.assertRaises(ValueError):
            batch_idx.start_idx(ibatch)

    def test_end_idx_batch_size_3_ibatch0(self):
        bs = 3
        ibatch = 0
        batch_idx = BatchIdx(self.array.shape[0], batch_size=bs)

        expected = 3
        actual = batch_idx.end_idx(ibatch)
        self.assertEqual(expected, actual)

    def test_end_idx_batch_size_3_ibatch2(self):
        bs = 3
        ibatch = 2
        batch_idx = BatchIdx(self.array.shape[0], batch_size=bs)

        expected = 7
        actual = batch_idx.end_idx(ibatch)
        self.assertEqual(expected, actual)

    def test_end_idx_batch_size_3_ibatch3(self):
        bs = 3
        ibatch = 3
        batch_idx = BatchIdx(self.array.shape[0], batch_size=bs)
        with self.assertRaises(ValueError):
            batch_idx.end_idx(ibatch)


class BatchedDataTestCase(unittest.TestCase):
    def setUp(self):
        array = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21]
        ])
        self.bd2 = BatchedData(array, 2)
        self.bd7 = BatchedData(array, 7)

    def test_batch_size_2_getitem_0(self):
        item = 0
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        actual = self.bd2[item]
        self.assertTrue(np.array_equal(expected, actual))

    def test_batch_size_2_getitem_1(self):
        item = 1
        expected = np.array([[7, 8, 9], [10, 11, 12]])
        actual = self.bd2[item]
        self.assertTrue(np.array_equal(expected, actual))

    def test_batch_size_2_getitem_3(self):
        item = 3
        expected = np.array([[19, 20, 21]])
        actual = self.bd2[item]
        self.assertTrue(np.array_equal(expected, actual))

    def test_batch_size_7_getitem_0(self):
        item = 0
        expected = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21]
        ])
        actual = self.bd7[item]
        self.assertTrue(np.array_equal(expected, actual))

    def test_batch_size_7_getitem_1(self):
        item = 1
        with self.assertRaises(ValueError):
            self.bd7[item]

    def test_iteration_batch_size_2(self):
        expected = [
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[7, 8, 9], [10, 11, 12]]),
            np.array([[13, 14, 15], [16, 17, 18]]),
            np.array([[19, 20, 21]])
        ]
        actual = [e for e in self.bd2]
        same = True
        for e, a in zip(expected, actual):
            if not np.array_equal(e, a):
                same = False
                break
        self.assertTrue(same)

    def test_length_batch_size_2(self):
        expected = 4
        actual = len(self.bd2)
        self.assertEqual(expected, actual)

    def test_iteration_batch_size_7(self):
        expected = [np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21]
        ])]
        actual = [e for e in self.bd7]
        same = True
        for e, a in zip(expected, actual):
            if not np.array_equal(e, a):
                same = False
                break
        self.assertTrue(same)

    def test_length_batch_size_7(self):
        expected = 1
        actual = len(self.bd7)
        self.assertEqual(expected, actual)


class BatchedDataTestCase2Array1D(unittest.TestCase):
    def setUp(self):
        array = np.arange(10)
        self.bd2 = BatchedData(array, 2)

    def test_batch_size_2_getitem_0(self):
        item = 0
        expected = np.array([0, 1])
        actual = self.bd2[item]
        self.assertTrue(np.array_equal(expected, actual))

    def test_batch_size_2_getitem_1(self):
        item = 1
        expected = np.array([2, 3])
        actual = self.bd2[item]
        self.assertTrue(np.array_equal(expected, actual))

    def test_batch_size_2_getitem_3(self):
        item = 3
        expected = np.array([6, 7])
        actual = self.bd2[item]
        self.assertTrue(np.array_equal(expected, actual))


class MaxFinderSparse(DoerBase):
    def __init__(self, output_dim_size):
        super().__init__(output_dim_size)

    def do(self, data):
        return csr_matrix(np.max(data.X, axis=1, keepdims=True))


class MaxMinFinderSparse(DoerBase):
    def __init__(self):
        super().__init__(output_dim_size=1, n_outputs=2)

    def do(self, data):
        return csr_matrix(np.max(data.X, axis=1, keepdims=True)), csr_matrix(np.min(data.X, axis=1, keepdims=True))


class MaxFinderDense(DoerBase):
    def __init__(self, output_dim_size):
        super().__init__(output_dim_size)

    def do(self, data):
        return np.max(data.X, axis=1, keepdims=True)


class MaxMinFinderDense(DoerBase):
    def __init__(self):
        super().__init__(output_dim_size=1, n_outputs=2)

    def do(self, data):
        return np.max(data.X, axis=1, keepdims=True), np.min(data.X, axis=1, keepdims=True)


class UtilsTest(unittest.TestCase):

    def setUp(self):
        self.data = Data(X=np.array([[1, 0], [0, 1], [2, 0], [0, 2], [3, 0], [0, 3], [5, 1]]))

    def test_do_in_minibatches_sparse_largebatch(self):
        expected = csr_matrix([[1], [1], [2], [2], [3], [3], [5]])
        actual = do_in_minibatches_sparse(doer=MaxFinderSparse(1), data=self.data, batchsize=100)
        self.assertEqual((expected != actual).nnz, 0)

    def test_do_in_minibatches_sparse_tinybatch(self):
        expected = csr_matrix([[1], [1], [2], [2], [3], [3], [5]])
        actual = do_in_minibatches_sparse(doer=MaxFinderSparse(1), data=self.data, batchsize=2)
        self.assertEqual((expected != actual).nnz, 0)

    def test_do_in_minibatches_dense_largebatch(self):
        expected = np.array([[1], [1], [2], [2], [3], [3], [5]])
        actual = do_in_minibatches_dense(doer=MaxFinderDense(1), data=self.data, batchsize=100)
        self.assertTrue(np.array_equal(expected, actual))

    def test_do_in_minibatches_dense_tinybatch(self):
        expected = np.array([[1], [1], [2], [2], [3], [3], [5]])
        actual = do_in_minibatches_dense(doer=MaxFinderDense(1), data=self.data, batchsize=2)
        self.assertTrue(np.array_equal(expected, actual))

    def test_do_in_minibatches_multioutput_tinybatch_dense(self):
        expected1 = np.array([[1], [1], [2], [2], [3], [3], [5]])
        expected2 = np.array([[0], [0], [0], [0], [0], [0], [1]])
        actual1, actual2 = do_in_minibatches_dense(doer=MaxMinFinderDense(), data=self.data, batchsize=2)
        self.assertTrue(np.array_equal(expected1, actual1))
        self.assertTrue(np.array_equal(expected2, actual2))

    def test_do_in_minibatches_multioutput_largebatch_dense(self):
        expected1 = np.array([[1], [1], [2], [2], [3], [3], [5]])
        expected2 = np.array([[0], [0], [0], [0], [0], [0], [1]])
        actual1, actual2 = do_in_minibatches_dense(doer=MaxMinFinderDense(), data=self.data, batchsize=100)
        self.assertTrue(np.array_equal(expected1, actual1))
        self.assertTrue(np.array_equal(expected2, actual2))

    def test_do_in_minibatches_multioutput_tinybatch_sparse(self):
        expected1 = csr_matrix(np.array([[1], [1], [2], [2], [3], [3], [5]]))
        expected2 = csr_matrix(np.array([[0], [0], [0], [0], [0], [0], [1]]))
        actual1, actual2 = do_in_minibatches_sparse(doer=MaxMinFinderSparse(), data=self.data, batchsize=2)
        self.assertEqual((expected1 != actual1).nnz, 0)
        self.assertEqual((expected2 != actual2).nnz, 0)

    def test_do_in_minibatches_multioutput_largebatch_sparse(self):
        expected1 = csr_matrix(np.array([[1], [1], [2], [2], [3], [3], [5]]))
        expected2 = csr_matrix(np.array([[0], [0], [0], [0], [0], [0], [1]]))
        actual1, actual2 = do_in_minibatches_sparse(doer=MaxMinFinderSparse(), data=self.data, batchsize=100)
        self.assertEqual((expected1 != actual1).nnz, 0)
        self.assertEqual((expected2 != actual2).nnz, 0)


if __name__ == '__main__':
    unittest.main()
