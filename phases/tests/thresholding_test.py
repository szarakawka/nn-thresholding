import unittest
import numpy as np
from scipy.sparse import csr_matrix
from phases.thresholding import r_cut, s_cut, cs_cut, scale_scores, css_cut, ss_cut, fit_s_cut, fit_cs_cut, fit_r_cut,\
    p_cut, ps_cut, fit_p_cut, fit_ps_cut, fit_cs_cut_parallel


class RCutThresholdingFittingTestCase(unittest.TestCase):

    def setUp(self):
        self.scores = csr_matrix(np.array([[0.0,  0.0,    0.3,    0.9,    2.1],
                                           [3.7,  0.7,    0.2,    0.1,    0.3],
                                           [0.0,  0.0,    0.75,    0.0,    0.0],
                                           [1.1,  0.9,    0.0,    0.0,    1.2]],
                                          dtype=np.float32))
        self.labels = csr_matrix(np.array(
            [[False, False, False, True, True],
             [True, True, False, False, False],
             [False, False, True, False, False],
             [True, False, False, False, True]],
            dtype=np.bool))

    def test_fit_r_cut_th_value(self):
        expected = 2
        actual, _ = fit_r_cut(self.scores, self.labels)
        self.assertEqual(expected, actual)

    def test_fit_r_cut_f1_value(self):
        expected = 1.
        _, actual = fit_r_cut(self.scores, self.labels)
        self.assertEqual(expected, actual)

    def test_no_changes_to_scores(self):
        expected = csr_matrix(np.array([[0.0,  0.0,    0.3,    0.9,    2.1],
                                        [3.7,  0.7,    0.2,    0.1,    0.3],
                                        [0.0,  0.0,    0.75,    0.0,    0.0],
                                        [1.1,  0.9,    0.0,    0.0,    1.2]],
                                       dtype=np.float32))
        _, _ = fit_r_cut(self.scores, self.labels)
        self.assertTrue((expected != self.scores).nnz == 0)


class PCutThresholdingFittingTestCase(unittest.TestCase):

    def setUp(self):
        self.scores = csr_matrix(np.array([np.arange(0., 1.2, 0.1),
                                           [0., 0., 0., 0., 1., 1., 1., 1., 0.5, 0.5, 0.6, 0.6],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.3, 0., 0.],
                                           np.ones(12),
                                           [0., 0., 0., 0., 0., 0.04, 0.01, 0.1, 4.3, 0., 0., 0.1]],
                                          dtype=np.float32).T)
        self.labels = csr_matrix(np.array(
            [[False, False, False, False, False, False, False, False, False, True, True, True],
             [False, False, False, False, False, True, True, True, False, True, True, True],
             [False, False, False, False, False, False, False, False, False, False, False, False],
             [True, True, True, True, True, True, True, True, True, True, True, True],
             [False, False, False, False, False, False, False, False, True, False, False, False]],
            dtype=np.bool).T)

    def test_fit_p_th_value(self):
        expected = np.array([0.25, 0.5, 0., 1., 1./12.])
        actual, _ = fit_p_cut(self.scores, self.labels)
        print(actual)
        self.assertTrue(np.allclose(expected, actual))

    def test_fit_p_cut_f1_value(self):
        expected = 1.
        _, actual = fit_p_cut(self.scores, self.labels)
        self.assertEqual(expected, actual)

    def test_no_changes_to_scores(self):
        expected = csr_matrix(np.array([np.arange(0., 1.2, 0.1),
                                      [0., 0., 0., 0., 1., 1., 1., 1., 0.5, 0.5, 0.6, 0.6],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.3, 0., 0.],
                                      np.ones(12),
                                      [0., 0., 0., 0., 0., 0.04, 0.01, 0.1, 4.3, 0., 0., 0.1]],
                                     dtype=np.float32).T)
        _, _ = fit_p_cut(self.scores, self.labels)
        self.assertTrue((expected != self.scores).nnz == 0)


class CsCutThresholdingFittingTestCase(unittest.TestCase):

    def setUp(self):
        self.scores = csr_matrix(np.array([[0.0,  0.0,    0.3,    0.9,    2.1],
                                           [3.7,  0.7,    0.2,    0.1,    0.3],
                                           [0.0,  0.0,    0.75,    0.0,    0.0],
                                           [1.1,  0.9,    0.0,    0.0,    1.2]],
                                          dtype=np.float32))
        self.labels = csr_matrix(np.array(
            [[False, False, False, True, True],
             [True, False, False, True, False],
             [False, False, True, False, False],
             [True, True, False, False, True]],
            dtype=np.bool))

    def test_fit_cs_cut_th_value(self):
        expected_thres_min = np.array([0.0, 0.699, 0.3, 0.0, 0.3])
        expected_thres_max = np.array([1.1, 0.9, 0.75, 0.1, 1.2])
        actual, _ = fit_cs_cut(self.scores, self.labels)
        self.assertTrue(np.all(np.logical_and(expected_thres_min <= actual, actual < expected_thres_max)))

    def test_fit_cs_cut_f1_value(self):
        expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        _, actual = fit_cs_cut(self.scores, self.labels)
        self.assertTrue(np.array_equal(expected, actual))

    def test_no_changes_to_scores(self):
        expected = csr_matrix(np.array([[0.0,  0.0,    0.3,    0.9,    2.1],
                                        [3.7,  0.7,    0.2,    0.1,    0.3],
                                        [0.0,  0.0,    0.75,    0.0,    0.0],
                                        [1.1,  0.9,    0.0,    0.0,    1.2]],
                                       dtype=np.float32))
        _, _ = fit_cs_cut(self.scores, self.labels)
        self.assertTrue((expected != self.scores).nnz == 0)


class CsCutThresholdingFittingParallelTestCase(unittest.TestCase):

    def setUp(self):
        self.scores = csr_matrix(np.array([[0.0,  0.0,    0.3,    0.9,    2.1],
                                           [3.7,  0.7,    0.2,    0.1,    0.3],
                                           [0.0,  0.0,    0.75,    0.0,    0.0],
                                           [1.1,  0.9,    0.0,    0.0,    1.2]],
                                          dtype=np.float32))
        self.labels = csr_matrix(np.array(
            [[False, False, False, True, True],
             [True, False, False, True, False],
             [False, False, True, False, False],
             [True, True, False, False, True]],
            dtype=np.bool))

    def test_fit_cs_cut_th_value(self):
        expected_thres_min = np.array([0.0, 0.699, 0.3, 0.0, 0.3])
        expected_thres_max = np.array([1.1, 0.9, 0.75, 0.1, 1.2])
        actual, _ = fit_cs_cut_parallel(self.scores, self.labels)
        self.assertTrue(np.all(np.logical_and(expected_thres_min <= actual, actual < expected_thres_max)))

    def test_fit_cs_cut_f1_value(self):
        expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        _, actual = fit_cs_cut_parallel(self.scores, self.labels)
        self.assertTrue(np.array_equal(expected, actual))

    def test_no_changes_to_scores(self):
        expected = csr_matrix(np.array([[0.0,  0.0,    0.3,    0.9,    2.1],
                                        [3.7,  0.7,    0.2,    0.1,    0.3],
                                        [0.0,  0.0,    0.75,    0.0,    0.0],
                                        [1.1,  0.9,    0.0,    0.0,    1.2]],
                                       dtype=np.float32))
        _, _ = fit_cs_cut_parallel(self.scores, self.labels)
        self.assertTrue((expected != self.scores).nnz == 0)


class SCutThresholdingFittingTestCase(unittest.TestCase):

    def setUp(self):
        self.scores = csr_matrix(np.array([[0.0,  0.0,    0.3,    0.9,    2.1],
                                           [3.7,  0.7,    0.2,    0.1,    0.3],
                                           [0.0,  0.0,    0.75,    0.0,    0.0],
                                           [1.1,  0.9,    0.0,    0.0,    1.2]],
                                          dtype=np.float32))
        self.labels = csr_matrix(np.array(
            [[False, False, False, True, True],
             [True, False, False, False, False],
             [False, False, True, False, False],
             [True, True, False, False, True]],
            dtype=np.bool))
        self.ideal_scores = csr_matrix(np.array(
            [[0., 0., 0., 1., 1.],
             [1., 0., 0., 0., 0.],
             [0., 0., 1., 0., 0.],
             [1., 1., 0., 0., 1.]],
            dtype=np.float32))

    def test_fit_s_cut_th_value(self):
        expected_range = [0.7, 0.75]
        actual, _ = fit_s_cut(self.scores, self.labels)
        self.assertTrue(expected_range[0] <= actual < expected_range[1])

    def test_fit_s_cut_f1_value(self):
        expected = 1.
        _, actual = fit_s_cut(self.scores, self.labels)
        self.assertEqual(expected, actual)

    def test_fit_s_cut_th_value_ideal_scores(self):
        expected = 0.5
        actual, _ = fit_s_cut(self.ideal_scores, self.labels)
        self.assertEqual(expected, actual)

    def test_fit_s_cut_f1_value_ideal_scores(self):
        expected = 1.
        _, actual = fit_s_cut(self.ideal_scores, self.labels)
        self.assertEqual(expected, actual)


class ThresholdingPredictionsTestCase(unittest.TestCase):
    def setUp(self):
        self.scores = csr_matrix(np.array(
            [[0.0,  0.0,    0.3,    0.9,    2.1],
             [3.7,  0.7,    0.2,    0.1,    0.3],
             [0.0,  0.0,    0.3,    0.0,    0.0],
             [1.1,  0.9,    0.0,    0.0,    1.2]],
            dtype=np.float32))
        self.scaled_scores = csr_matrix(np.array(
            [[0.0,      0.0,        0.3/2.1,    0.9/2.1,    1.],
             [1.,       0.7/3.7,    0.2/3.7,    0.1/3.7,    0.3/3.7],
             [0.0,      0.0,        1.,         0.0,        0.0],
             [1.1/1.2,  0.9/1.2,    0.0,        0.0,        1.]],
            dtype=np.float32))

    def test_scale_scores(self):
        actual = scale_scores(self.scores)
        self.assertTrue(np.all(np.isclose(actual.toarray(), self.scaled_scores.toarray())))

    def test_p_cut_t1(self):
        th = [0.25, 1/12, 0.9, 0.5, 1.]
        expected = csr_matrix(np.array(
            [[False, False, True, True, True],
             [True, False, True, True, True],
             [False, False, True, False, False],
             [False, False, False, False, True]],
            dtype=np.bool))
        actual = p_cut(self.scores, th)
        self.assertEqual(expected.__ne__(actual).nnz, 0)

    def test_p_cut_t1_x05_ensure_false(self):
        th = [0.25, 1/12, 0.9, 0.5, 1.]
        expected1 = csr_matrix(np.array(
            [[False, False, True, True, True],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, True]],
            dtype=np.bool))
        expected2 = csr_matrix(np.array(
            [[False, False, False, True, True],
             [False, False, False, False, False],
             [False, False, True, False, False],
             [False, False, False, False, True]],
            dtype=np.bool))
        actual = p_cut(self.scores, th, proportionality_const=0.5, ensure_no_empty_predictions_flag=False)
        self.assertTrue(expected1.__ne__(actual).nnz == 0 or expected2.__ne__(actual).nnz == 0)

    def test_p_cut_t1_x05_ensure_true(self):
        th = [0.25, 1/12, 0.9, 0.5, 1.]
        expected1 = csr_matrix(np.array(
            [[False, False, True, True, True],
             [False, False, False, False, False],
             [False, False, True, False, False],
             [False, False, False, False, True]],
            dtype=np.bool))
        expected2 = csr_matrix(np.array(
            [[False, False, False, True, True],
             [True, False, False, False, False],
             [False, False, True, False, False],
             [False, False, False, False, True]],
            dtype=np.bool))
        actual = p_cut(self.scores, th, proportionality_const=0.5)
        self.assertTrue(expected1.__ne__(actual).nnz == 0 or expected2.__ne__(actual).nnz == 0)

    def test_ps_cut_t1(self):
        th = [0.0, 1/12, 0.9, 0.5, 1.]
        expected = csr_matrix(np.array(
            [[False, False, True, True, True],
             [False, False, True, True, True],
             [False, False, True, False, False],
             [False, False, False, False, True]],
            dtype=np.bool))
        actual = ps_cut(self.scores, th)
        self.assertEqual(expected.__ne__(actual).nnz, 0)

    def test_ss_cut_th03(self):
        th = 0.3
        expected = csr_matrix(np.array(
            [[False, False, False, True, True],
             [True, False, False, False, False],
             [False, False, True, False, False],
             [True, True, False, False, True]],
            dtype=np.bool))
        actual = ss_cut(self.scores, th)
        self.assertEqual(expected.__ne__(actual).nnz, 0)

    def test_css_cut_th_v1(self):
        th = np.array([0.8,  0.6,    0.6,   0.8,    0.])
        expected = csr_matrix(np.array(
            [[False, False, False, False, True],
             [True, False, False, False, True],
             [False, False, True, False, False],
             [True, True, False, False, True]],
            dtype=np.bool))
        actual = css_cut(self.scores, th)
        self.assertEqual(expected.__ne__(actual).nnz, 0)

    def test_cs_cut_th_v2(self):
        th = np.array([0.5,  0.6,    0.6,   0.6,    0.])
        expected = csr_matrix(np.array(
            [[False, False, False, True, True],
             [True, True, False, False, True],
             [False, False, True, False, False],
             [True, True, False, False, True]],
            dtype=np.bool))
        actual = cs_cut(self.scores, th)
        self.assertEqual(expected.__ne__(actual).nnz, 0)

    def test_cs_cut_th_v1(self):
        th = [1.5,  0.6,    0.25,   0.6,    1.5]
        expected = csr_matrix(np.array(
            [[False, False, True, True, True],
             [True, True, False, False, False],
             [False, False, True, False, False],
             [False, True, False, False, False]],
            dtype=np.bool))
        actual = cs_cut(self.scores, th)
        self.assertEqual(expected.__ne__(actual).nnz, 0)

    def test_s_cut_th03(self):
        th = 0.3
        expected = csr_matrix(np.array(
            [[False, False, False, True, True],
             [True, True, False, False, False],
             [False, False, True, False, False],
             [True, True, False, False, True]],
            dtype=np.bool))
        actual = s_cut(self.scores, th)
        self.assertEqual(expected.__ne__(actual).nnz, 0)

    def test_s_cut_th1(self):
        th = 1.0
        expected = csr_matrix(np.array(
            [[False, False, False, False, True],
             [True, False, False, False, False],
             [False, False, True, False, False],
             [True, False, False, False, True]],
            dtype=np.bool))
        actual = s_cut(self.scores, th)
        self.assertEqual(expected.__ne__(actual).nnz, 0)

    def test_r_cut_r1(self):
        r = 1
        expected = csr_matrix(np.array(
            [[False, False, False, False, True],
             [True, False, False, False, False],
             [False, False, True, False, False],
             [False, False, False, False, True]],
            dtype=np.bool))
        actual = r_cut(self.scores, r)
        self.assertEqual(expected.__ne__(actual).nnz, 0)

    def test_r_cut_r3(self):
        r = 3
        expected = csr_matrix(np.array(
            [[False, False, True, True, True],
             [True, True, False, False, True],
             [False, False, True, False, False],
             [True, True, False, False, True]],
            dtype=np.bool))
        actual = r_cut(self.scores, r)
        self.assertEqual(expected.__ne__(actual).nnz, 0)


if __name__ == '__main__':
    unittest.main()
