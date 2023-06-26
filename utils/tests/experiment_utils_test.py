import unittest
from utils.experiments_utils import *


class SampleWeightsCalculationTestCase(unittest.TestCase):
    def setUp(self):
        self.label_mtx = csr_matrix(np.array([[0, 1],
                                              [0, 1],
                                              [1, 1],
                                              [0, 1],
                                              [1, 0],
                                              [0, 1],
                                              [1, 0],
                                              [0, 1],
                                              [0, 1]]))

    def test_calculate_class_weights(self):
        expected = np.array([1.5, 9./14.])
        actual = calculate_class_weights(self.label_mtx)
        self.assertTrue(np.array_equal(expected, actual))

    def test_calculate_sample_weights_for_multi_label_problem_mode_sum(self):
        cw = [1.5, 9./14.]
        expected = np.array([cw[1], cw[1], cw[0]+cw[1], cw[1], cw[0], cw[1], cw[0], cw[1], cw[1]])
        actual = calculate_sample_weights_for_multi_label_problem(self.label_mtx, mode='sum')
        self.assertTrue(np.array_equal(expected, actual))

    def test_calculate_sample_weights_for_multi_label_problem_mode_mean(self):
        cw = [1.5, 9./14.]
        expected = np.array([cw[1], cw[1], (cw[0]+cw[1])/2., cw[1], cw[0], cw[1], cw[0], cw[1], cw[1]])
        actual = calculate_sample_weights_for_multi_label_problem(self.label_mtx, mode='mean')
        self.assertTrue(np.array_equal(expected, actual))

    def test_calculate_sample_weights_for_multi_label_problem_mode_max(self):
        cw = [1.5, 9./14.]
        expected = np.array([cw[1], cw[1], max(cw), cw[1], cw[0], cw[1], cw[0], cw[1], cw[1]])
        actual = calculate_sample_weights_for_multi_label_problem(self.label_mtx, mode='max')
        self.assertTrue(np.array_equal(expected, actual))


if __name__ == '__main__':
    unittest.main()
