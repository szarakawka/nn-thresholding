from nn_thresholding.thresholding_models_in_keras import ModelT2, ModelT3, dense_outputs_to_sparse_predictions
from nn_thresholding.hyperparameters_config import HP
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from tensorflow.python.keras.models import Model
import unittest


class ModelT2SanityCheckTestCase(unittest.TestCase):
    def setUp(self):
        self.n_samples = 6
        self.n_labels = 15

        self.params = HP()
        self.params.embed_size = 4
        self.params.trainable_embeddings = False
        self.params.model_version = 0
        self.params.n_outputs = self.n_labels

        self.initial_embeddings = np.array([[-2, -1, 0, 1],
                                            [1, 2, 3, 4],
                                            [4, 3, 2, 1],
                                            [2, 3, 4, 1],
                                            [3, 4, 1, 2],
                                            [3, 4, 2, 1],
                                            [0, 0, 1, 1],
                                            [1, 1, 0, 0],
                                            [-1, -1, -1, -1],
                                            [-2, -2, -2, 1],
                                            [1, 1, 1, 1],
                                            [1., 0, 0, 0],
                                            [0, 1., 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])
        scores = lil_matrix((self.n_samples, self.n_labels), dtype=np.float32)
        scores.rows = [[0, 1, 8, 11, 13],
                       [0, 3, 6, 8, 14],
                       [3, 4, 5],
                       [1, 2, 12, 13],
                       [9],
                       [5, 7, 10, 11, 13, 14]]
        scores.data = [[0.1, 1.1, 8.1, 11.5, 1.5],
                       [0.9, 3.2, 6.2, 0.8, 0.3],
                       [3.5, 2.4, 5.5],
                       [1.1, 2.2, 12.2, 1.3],
                       [9.3],
                       [5.1, 7.2, 1.3, 1.1, 1.3, 9.2]]
        self.scores = scores.tocsr()
        label_mtx = lil_matrix((self.n_samples, self.n_labels), dtype=np.bool)
        label_mtx.rows = [[0, 4, 11],
                          [3],
                          [7, 8, 9, 10, 11, 12],
                          [4, 14],
                          [1, 3],
                          [6]]
        label_mtx.data = [3*[True], 1*[True], 6*[True], 2*[True], 2*[True], 1*[True]]
        self.label_mtx = label_mtx.tocsr()

    def test_better_testing(self):
        sample_ids = [0, 1]
        expected0 = 11.5 * np.array([1., 0, 0, 0]) + 8.1 * np.array([-1, -1, -1, -1]) + 1.5 * np.array([0, 0, 1, 0]) + \
                    1.1 * np.array([1, 2, 3, 4]) + 0.1 * np.array([-2, -1, 0, 1])
        expected1 = 6.2 * np.array([0, 0, 1, 1]) + 3.2 * np.array([2, 3, 4, 1]) + 0.9 * np.array([-2, -1, 0, 1]) + \
                    0.8 * np.array([-1, -1, -1, -1]) + 0.3 * np.array([0, 0, 0, 1])
        expected = np.vstack((expected0, expected1))
        actual = np.matmul(self.scores[sample_ids, :].toarray(), self.initial_embeddings)
        self.assertTrue(np.alltrue(np.isclose(expected, actual)))

    def test_build_model_t2_len4_resulting_embedding_sample0(self):
        whole_model = ModelT2(self.params, initial_embeddings=self.initial_embeddings)
        model = Model(inputs=whole_model.model.inputs,
                      outputs=whole_model.model.get_layer('effective_embedding').output)
        sample_ids = [0]
        expected = np.matmul(self.scores[sample_ids, :].toarray(), self.initial_embeddings)
        actual = model.predict(self.scores[sample_ids, :])
        self.assertTrue(np.alltrue(np.isclose(expected, actual)))

    def test_build_model_t2_len4_resulting_embedding_sample1(self):
        whole_model = ModelT2(self.params, initial_embeddings=self.initial_embeddings)
        model = Model(inputs=whole_model.model.inputs,
                      outputs=whole_model.model.get_layer('effective_embedding').output)
        sample_ids = [1]
        expected = np.matmul(self.scores[sample_ids, :].toarray(), self.initial_embeddings)
        actual = model.predict(self.scores[sample_ids, :])
        self.assertTrue(np.alltrue(np.isclose(expected, actual)))

    def test_build_model_t2_len4_resulting_embedding_samples01(self):
        whole_model = ModelT2(self.params, initial_embeddings=self.initial_embeddings)
        model = Model(inputs=whole_model.model.inputs,
                      outputs=whole_model.model.get_layer('effective_embedding').output)
        sample_ids = [0, 1]
        expected = np.matmul(self.scores[sample_ids, :].toarray(), self.initial_embeddings)
        actual = model.predict(self.scores[sample_ids, :])
        self.assertTrue(np.alltrue(np.isclose(expected, actual)))

    def test_build_model_t2_len4_resulting_embedding_all_samples(self):
        whole_model = ModelT2(self.params, initial_embeddings=self.initial_embeddings)
        model = Model(inputs=whole_model.model.inputs,
                      outputs=whole_model.model.get_layer('effective_embedding').output)
        sample_ids = np.arange(self.scores.shape[0])
        expected = np.matmul(self.scores[sample_ids, :].toarray(), self.initial_embeddings)
        actual = model.predict(self.scores[sample_ids, :])
        self.assertTrue(np.alltrue(np.isclose(expected, actual)))

    def test_predict_in_batches_batch_size_1(self):
        batch_size = 1
        whole_model = ModelT2(self.params, initial_embeddings=self.initial_embeddings)
        expected = csr_matrix(np.round(whole_model.model.predict(self.scores)).astype(np.float32))
        actual = whole_model.predict_in_batches(self.scores, batch_size=batch_size,
                                                ensure_prediction_for_each_sample_flag=False)
        self.assertTrue((expected != actual).nnz == 0)

    def test_predict_in_batches_batch_size_2(self):
        batch_size = 2
        whole_model = ModelT2(self.params, initial_embeddings=self.initial_embeddings)
        expected = csr_matrix(np.round(whole_model.model.predict(self.scores)).astype(np.float32))
        actual = whole_model.predict_in_batches(self.scores, batch_size=batch_size,
                                                ensure_prediction_for_each_sample_flag=False)
        self.assertTrue((expected != actual).nnz == 0)

    def test_predict_in_batches_batch_size_10(self):
        batch_size = 10
        whole_model = ModelT2(self.params, initial_embeddings=self.initial_embeddings)
        expected = csr_matrix(np.round(whole_model.model.predict(self.scores)).astype(np.float32))
        actual = whole_model.predict_in_batches(self.scores, batch_size=batch_size,
                                                ensure_prediction_for_each_sample_flag=False)
        self.assertTrue((expected != actual).nnz == 0)

    def test_predict_in_batches_batch_dtype_float64(self):
        batch_size = 2
        whole_model = ModelT2(self.params, initial_embeddings=self.initial_embeddings)
        expected = csr_matrix(np.round(whole_model.model.predict(self.scores)).astype(np.float32))
        actual = whole_model.predict_in_batches(self.scores.astype(np.float64), batch_size=batch_size,
                                                ensure_prediction_for_each_sample_flag=False)
        self.assertTrue((expected != actual).nnz == 0)


class ModelT3SanityCHeckTestCase(unittest.TestCase):

    def setUp(self):
        self.n_samples = 6
        self.n_labels = 15

        self.params = HP()
        self.params.embed_size = 4
        self.params.trainable_embeddings = False
        self.params.model_version = 0
        self.params.n_outputs = self.n_labels

        self.initial_embeddings = np.array([[-2, -1, 0, 1],
                                            [1, 2, 3, 4],
                                            [4, 3, 2, 1],
                                            [2, 3, 4, 1],
                                            [3, 4, 1, 2],
                                            [3, 4, 2, 1],
                                            [0, 0, 1, 1],
                                            [1, 1, 0, 0],
                                            [-1, -1, -1, -1],
                                            [-2, -2, -2, 1],
                                            [1, 1, 1, 1],
                                            [1., 0, 0, 0],
                                            [0, 1., 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])
        scores = lil_matrix((self.n_samples, self.n_labels), dtype=np.float32)
        scores.rows = [[0, 1],
                       [0, 1, 8, 11, 13],
                       [0, 3, 6, 8, 14],
                       [3, 4, 5],
                       [1, 2, 12, 13],
                       [9],
                       [5, 7, 10, 11, 13, 14]]
        scores.data = [[1., 1.],
                       [0.1, 1.1, 8.1, 11.5, 1.5],
                       [0.9, 3.2, 6.2, 0.8, 0.3],
                       [3.5, 2.4, 5.5],
                       [1.1, 2.2, 12.2, 1.3],
                       [9.3],
                       [5.1, 7.2, 1.3, 1.1, 1.3, 9.2]]
        self.scores = scores.tocsr()
        label_mtx = lil_matrix((self.n_samples, self.n_labels), dtype=np.bool)
        label_mtx.rows = [[0, 4, 11],
                          [3],
                          [7, 8, 9, 10, 11, 12],
                          [4, 14],
                          [1, 3],
                          [6]]
        label_mtx.data = [3*[True], 1*[True], 6*[True], 2*[True], 2*[True], 1*[True]]
        self.label_mtx = label_mtx.tocsr()
        self.css_thresholds = np.linspace(0.0, 14.0, self.n_labels)

    def test_predict_in_batches_batch_size_2_initial_embeddings_zero(self):
        batch_size = 2
        expected_logits = self.scores.toarray() - self.css_thresholds
        expected_probs = (1./(1. + np.exp(-expected_logits)))
        model_t3 = ModelT3(self.params, css_thresholds=self.css_thresholds,
                           initial_embeddings=np.zeros((self.n_labels, self.params.embed_size)))
        actual_probs = model_t3.model.predict(self.scores, batch_size=batch_size)
        self.assertTrue(np.alltrue(np.isclose(expected_probs, actual_probs)))

    def test_logits_sample0(self):
        sample_ids = [0]
        scores = self.scores[sample_ids, :]

        model_t3 = ModelT3(self.params, css_thresholds=self.css_thresholds, initial_embeddings=self.initial_embeddings)

        m_t3_logit_t2 = Model(inputs=model_t3.model.inputs, outputs=model_t3.model.get_layer('logits').output)
        expected = m_t3_logit_t2.predict(scores) + scores.toarray() - self.css_thresholds

        model_t3_logit = Model(inputs=model_t3.model.inputs, outputs=model_t3.model.get_layer('t3_logits').output)
        actual = model_t3_logit.predict(scores)

        self.assertTrue(np.alltrue(np.isclose(expected, actual)))

    def test_logits(self):
        scores = self.scores

        model_t3 = ModelT3(self.params, css_thresholds=self.css_thresholds, initial_embeddings=self.initial_embeddings)

        m_t3_logit_t2 = Model(inputs=model_t3.model.inputs, outputs=model_t3.model.get_layer('logits').output)
        expected = m_t3_logit_t2.predict(scores) + scores.toarray() - self.css_thresholds

        model_t3_logit = Model(inputs=model_t3.model.inputs, outputs=model_t3.model.get_layer('t3_logits').output)
        actual = model_t3_logit.predict(scores)

        self.assertTrue(np.alltrue(np.isclose(expected, actual)))


class DenseOutputsToSparsePredictionsTestCase(unittest.TestCase):
    def setUp(self):
        self.predictions_float = np.array([[0.90110805, 0.76981139, 0.83966795],
                                           [0.45136449, 0.17397073, 0.27692739],
                                           [0.56340316, 0.85531379, 0.03155763],
                                           [0.03269164, 0.21930204, 0.39700875],
                                           [0.50879399, 0.63975930, 0.44548482]]).astype(np.float32)

    def test_dense_outputs_to_sparse_predictions_ensure_true(self):
        ensure = True
        threshold = 0.5
        expected = csr_matrix(np.array([[1, 1, 1],
                                        [1, 0, 0],
                                        [1, 1, 0],
                                        [0, 0, 1],
                                        [1, 1, 0]]))
        actual = dense_outputs_to_sparse_predictions(self.predictions_float, ensure, threshold)
        self.assertTrue((expected != actual).nnz == 0)

    def test_dense_outputs_to_sparse_predictions_ensure_false(self):
        ensure = False
        threshold = 0.5
        expected = csr_matrix(np.array([[1, 1, 1],
                                        [0, 0, 0],
                                        [1, 1, 0],
                                        [0, 0, 0],
                                        [1, 1, 0]]))
        actual = dense_outputs_to_sparse_predictions(self.predictions_float, ensure, threshold)
        self.assertTrue((expected != actual).nnz == 0)


if __name__ == '__main__':
    unittest.main()
