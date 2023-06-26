import unittest

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from utils.sparse_ops import column_calculate_tp_fp_fn


class SparseOpsTest(unittest.TestCase):

    def setUp(self):
        self.gt_labels = csc_matrix([[1, 0, 0],
                                     [1, 0, 1],
                                     [1, 1, 0],
                                     [0, 1, 1],
                                     [0, 1, 0],
                                     [0, 0, 1],
                                     [0, 0, 1]], dtype=bool)

        self.pred_labels = csc_matrix([[0, 1, 1],
                                       [1, 0, 1],
                                       [1, 1, 0],
                                       [1, 0, 1],
                                       [0, 1, 0],
                                       [1, 1, 1],
                                       [0, 1, 0]], dtype=bool)

    def test_column_true_positives(self):
        actual_tp, _, _ = column_calculate_tp_fp_fn(self.gt_labels, self.pred_labels)
        expected_tp = np.array([2, 2, 3], dtype=int)
        self.assertTrue(np.array_equal(actual_tp, expected_tp))

    def test_column_false_positives(self):
        _, actual_fp, _ = column_calculate_tp_fp_fn(self.gt_labels, self.pred_labels)
        expected_fp = np.array([2, 3, 1], dtype=int)
        self.assertTrue(np.array_equal(actual_fp, expected_fp))

    def test_column_false_negatives(self):
        _, _, actual_fn = column_calculate_tp_fp_fn(self.gt_labels, self.pred_labels)
        expected_fn = np.array([1, 1, 1], dtype=int)
        self.assertTrue(np.array_equal(actual_fn, expected_fn))

    def test_column_calculate_tp_fp_fn_assertion_1(self):
        with self.assertRaises(TypeError):
            column_calculate_tp_fp_fn(csr_matrix(self.gt_labels), self.pred_labels)

    def test_column_calculate_tp_fp_fn_assertion_2(self):
        with self.assertRaises(TypeError):
            column_calculate_tp_fp_fn(self.gt_labels, csr_matrix(self.pred_labels))
    #
    # def test_column_f1_per_label(self):
    #     actual_f1 = column_f1_per_label(self.gt_labels, self.pred_labels)
    #     expected_f1 = f1_score(self.gt_labels, self.pred_labels, average='macro')
    #     self.assertEqual(np.mean(actual_f1), expected_f1)

if __name__ == '__main__':
    unittest.main()
