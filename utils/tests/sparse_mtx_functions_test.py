import unittest
import numpy as np
from scipy.sparse import csr_matrix
from utils.sparse_mtx_functions import filter_sparse_matrix, blend


class FilterMtxTestCase1(unittest.TestCase):
    def setUp(self):
        m = np.array([[1, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 1, 0, 1, 0],
                      [1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0, 0]])
        self.mtx = csr_matrix(m)

    def test_filter_sparse_matrix_1_mtx(self):
        min_nz_per_row = 1
        min_nz_per_col = 1

        m = np.array([[1, 0, 0, 0, 1, 1],
                      [0, 1, 0, 1, 0, 1],
                      [1, 1, 1, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0]])
        expected = csr_matrix(m)

        actual, _, _, _ = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertEqual((expected != actual).nnz, 0)

    def test_filter_sparse_matrix_1_rows_removed(self):
        min_nz_per_row = 1
        min_nz_per_col = 1
        expected = np.array([1])

        _, actual, _, _ = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertTrue(np.array_equal(expected, actual))

    def test_filter_sparse_matrix_1_cols_removed(self):
        min_nz_per_row = 1
        min_nz_per_col = 1
        expected = np.array([6])

        _, _, actual, _ = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertTrue(np.array_equal(expected, actual))

    def test_filter_sparse_matrix_1_was_changed(self):
        min_nz_per_row = 1
        min_nz_per_col = 1
        expected = True

        _, _, _, actual = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertEqual(expected, actual)

    def test_filter_sparse_matrix_2_mtx(self):
        min_nz_per_row = 1
        min_nz_per_col = 2

        m = np.array([[1, 0, 0, 1, 1],
                      [0, 1, 0, 0, 1],
                      [1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0]])
        expected = csr_matrix(m)

        actual, _, _, _ = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertEqual((expected != actual).nnz, 0)

    def test_filter_sparse_matrix_2_rows_removed(self):
        min_nz_per_row = 1
        min_nz_per_col = 2
        expected = np.array([1])

        _, actual, _, _ = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertTrue(np.array_equal(expected, actual))

    def test_filter_sparse_matrix_2_cols_removed(self):
        min_nz_per_row = 1
        min_nz_per_col = 2
        expected = np.array([3, 6])

        _, _, actual, _ = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertTrue(np.array_equal(expected, actual))

    def test_filter_sparse_matrix_2_was_changed(self):
        min_nz_per_row = 1
        min_nz_per_col = 2
        expected = True

        _, _, _, actual = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertEqual(expected, actual)

    def test_filter_sparse_matrix_3_mtx(self):
        min_nz_per_row = 2
        min_nz_per_col = 2

        m = np.array([[1, 0, 0, 1, 1],
                      [0, 1, 0, 0, 1],
                      [1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0]])
        expected = csr_matrix(m)

        actual, _, _, _ = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertEqual((expected != actual).nnz, 0)

    def test_filter_sparse_matrix_3_rows_removed(self):
        min_nz_per_row = 2
        min_nz_per_col = 2
        expected = np.array([1])

        _, actual, _, _ = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertTrue(np.array_equal(expected, actual))

    def test_filter_sparse_matrix_3_cols_removed(self):
        min_nz_per_row = 2
        min_nz_per_col = 2
        expected = np.array([3, 6])

        _, _, actual, _ = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertTrue(np.array_equal(expected, actual))

    def test_filter_sparse_matrix_3_was_changed(self):
        min_nz_per_row = 2
        min_nz_per_col = 2
        expected = True

        _, _, _, actual = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertEqual(expected, actual)


class FilterMtxTestCase2(unittest.TestCase):
    def setUp(self):
        m = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 1, 0, 1, 0, 1, 0, 1, 0],
                      [1, 1, 1, 0, 0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0, 0, 0, 1]])
        self.mtx = csr_matrix(m)

    def test_filter_sparse_matrix_1_mtx(self):
        min_nz_per_row = 3
        min_nz_per_col = 2

        m = np.array([[1, 0, 1, 1],
                      [0, 1, 1, 1],
                      [1, 1, 1, 0]])

        expected = csr_matrix(m)

        actual, _, _, _ = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col,
            verbose=True)

        self.assertEqual((expected != actual).nnz, 0)

    def test_filter_sparse_matrix_1_rows_removed(self):
        min_nz_per_row = 3
        min_nz_per_col = 2

        expected = np.array([1, 4])

        _, actual, _, _ = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertTrue(np.array_equal(expected, actual))

    def test_filter_sparse_matrix_1_cols_removed(self):
        min_nz_per_row = 3
        min_nz_per_col = 2
        expected = np.array([2, 3, 4, 6, 8])

        _, _, actual, _ = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertTrue(np.array_equal(expected, actual))

    def test_filter_sparse_matrix_1_was_changed(self):
        min_nz_per_row = 3
        min_nz_per_col = 2
        expected = True

        _, _, _, actual = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertEqual(expected, actual)


class FilterMtxTestCaseNoChanges(unittest.TestCase):
    def setUp(self):
        m = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 1, 0, 1, 0, 1, 0, 1, 0],
                      [1, 1, 1, 0, 0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0, 0, 0, 1]])
        self.mtx = csr_matrix(m)

    def test_filter_sparse_matrix_1_mtx(self):
        min_nz_per_row = 0
        min_nz_per_col = 0

        m = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 1, 0, 1, 0, 1, 0, 1, 0],
                      [1, 1, 1, 0, 0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0, 0, 0, 1]])

        expected = csr_matrix(m)

        actual, _, _, _ = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col,
            verbose=True)

        self.assertEqual((expected != actual).nnz, 0)

    def test_filter_sparse_matrix_1_rows_removed(self):
        min_nz_per_row = 0
        min_nz_per_col = 0

        expected = np.array([])

        _, actual, _, _ = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertTrue(np.array_equal(expected, actual))

    def test_filter_sparse_matrix_1_cols_removed(self):
        min_nz_per_row = 0
        min_nz_per_col = 0
        expected = np.array([])

        _, _, actual, _ = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertTrue(np.array_equal(expected, actual))

    def test_filter_sparse_matrix_1_was_changed(self):
        min_nz_per_row = 0
        min_nz_per_col = 0
        expected = False

        _, _, _, actual = filter_sparse_matrix(
            self.mtx, min_nonzeros_per_row=min_nz_per_row,
            min_nonzeros_per_col=min_nz_per_col)

        self.assertEqual(expected, actual)


class BlendTestCase(unittest.TestCase):

    def setUp(self):
        m1 = np.array([[1, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 1, 0, 1, 0],
                      [1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0, 0]])
        self.mtx1 = csr_matrix(m1)
        m2 = np.array([[1, 0, 0, 0, 1, -0.5, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 1, 0, 1, 0],
                       [0, 0, 1, 0, 0, 0, 1],
                       [0, 0, 1, 0, 1, 0, 0]])
        self.mtx2 = csr_matrix(m2)

    def test_correct_result_alpha00(self):
        actual = blend(self.mtx1, self.mtx2, 0.0)
        expected = self.mtx1
        self.assertEqual((expected != actual).nnz, 0)

    def test_correct_result_alpha10(self):
        actual = blend(self.mtx1, self.mtx2, 1.0)
        expected = self.mtx2
        self.assertEqual((expected != actual).nnz, 0)

    def test_correct_result_alpha01(self):
        actual = blend(self.mtx1, self.mtx2, 0.1)
        m = np.array([[1, 0, 0, 0, 1, 0.85, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 1, 0, 1, 0],
                      [0.9, 0.9, 1, 0, 0, 0, 0.1],
                      [0, 0, 1, 0, 1, 0, 0]])
        expected = csr_matrix(m)
        self.assertEqual((expected != actual).nnz, 0)

    def test_unchanged_op1(self):
        _ = blend(self.mtx1, self.mtx2, 0.1)
        actual = self.mtx1
        m = np.array([[1, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 1, 0, 1, 0],
                      [1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0, 0]])
        expected = csr_matrix(m)
        self.assertEqual((expected != actual).nnz, 0)

    def test_unchanged_op2(self):
        _ = blend(self.mtx1, self.mtx2, 0.1)
        actual = self.mtx2
        m = np.array([[1, 0, 0, 0, 1, -0.5, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 1, 0, 1, 0],
                       [0, 0, 1, 0, 0, 0, 1],
                       [0, 0, 1, 0, 1, 0, 0]])
        expected = csr_matrix(m)
        self.assertEqual((expected != actual).nnz, 0)


if __name__ == '__main__':
    unittest.main()