import unittest
from dataio.dataset import DatasetInterpreter, Dataset, MetaInfoContainer, Entity
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix, coo_matrix


def construct_random_dummy_dataset(n_docs, n_features):

    def create_dummy_label_mtx(n_docs, label_choice):
        row = []
        col = []
        data = []
        for n in range(n_docs):
            for l in label_choice[n % len(label_choice)]:
                row.append(n)
                col.append(l)
                data.append(1.)
        return coo_matrix((data, (row, col))).tocsr()

    def create_dummy_feature_mtx(n_docs, n_features):
        row = []
        col = []
        data = []
        for n in range(n_docs):
            features = np.sort(np.random.choice(n_features, np.random.randint(1, n_features + 1), replace=False))
            for f in features:
                row.append(n)
                col.append(f)
                data.append(1.)
        return coo_matrix((data, (row, col)), shape=(n_docs, n_features)).tocsr()

    label_choice = [[0], [1], [0, 1]]
    # These are for easy tests
    docs_mi = MetaInfoContainer([Entity(orig_id=None, name="Text_{:03d}".format(n)) for n in range(n_docs)])
    labels_mi = MetaInfoContainer([Entity(orig_id=None, name="Label_{}".format(n)) for n in range(len(label_choice))])
    features_mi = MetaInfoContainer([Entity(orig_id=None, name="Feature_{:03d}".format(n)) for n in range(n_features)])

    return Dataset(
        name="DummyDataset",
        label_mtx=create_dummy_label_mtx(n_docs, label_choice),
        feature_mtx=create_dummy_feature_mtx(n_docs, n_features),
        interpreter=DatasetInterpreter(docs_mi, labels_mi, features_mi))


class DatasetInterpreterDummyDatasetNoOriginalsTest(unittest.TestCase):

    def setUp(self):
        self.n_docs = 9
        self.n_features = 15
        self.dummy_dataset = construct_random_dummy_dataset(self.n_docs, self.n_features)

    def test_no_features(self):
        self.assertEqual(self.n_features, self.dummy_dataset.n_features)

    def test_no_labels(self):
        self.assertEqual(2, self.dummy_dataset.n_labels)

    def test_no_docs(self):
        self.assertEqual(self.n_docs, self.dummy_dataset.n_docs)

    def test_doc_string_doc0_default(self):
        doc_id = 0
        expected = '"Text_000"'
        self.assertEqual(expected, self.dummy_dataset.interpreter.doc_string(doc_id))

    def test_doc_string_doc0_title_id(self):
        doc_id = 0
        expected = '"Text_000" (id=0)'
        self.assertEqual(expected, self.dummy_dataset.interpreter.doc_string(doc_id, include_id=True))

    def test_doc_string_doc0_title_id_orig_id(self):
        doc_id = 0
        expected = '"Text_000" (id=0, orig_id=None)'
        self.assertEqual(expected, self.dummy_dataset.interpreter.doc_string(doc_id, include_id=True, include_orig_id=True))

    def test_doc_string_doc0_title_orig_id(self):
        doc_id = 0
        expected = '"Text_000" (orig_id=None)'
        self.assertEqual(expected, self.dummy_dataset.interpreter.doc_string(doc_id, include_orig_id=True))

    def test_doc_string_doc0_title_id_content(self):
        doc_id = 0
        expected = '"Text_000" (id=0): "<no original text data available>"'
        self.assertEqual(expected, self.dummy_dataset.interpreter.doc_string(doc_id, include_id=True, include_content=True))

    def test_doc_string_doc0_title_id_orig_id_content(self):
        doc_id = 0
        expected = '"Text_000" (id=0, orig_id=None): "<no original text data available>"'
        self.assertEqual(expected, self.dummy_dataset.interpreter.doc_string(doc_id, include_id=True, include_orig_id=True,
                                                                             include_content=True))

    def test_doc_labels_doc0(self):
        doc_id = 0
        expected = '"Label_0"'
        self.assertEqual(expected, self.dummy_dataset.interpreter.doc_labels(doc_id, self.dummy_dataset.label_mtx))

    def test_doc_labels_doc17(self):
        doc_id = 8
        expected = '"Label_0", "Label_1"'
        self.assertEqual(expected, self.dummy_dataset.interpreter.doc_labels(doc_id, self.dummy_dataset.label_mtx))

    def test_label_docs_label0(self):
        label_id = 0
        expected = '"Text_000", "Text_002", "Text_003", "Text_005", "Text_006", "Text_008"'
        self.assertEqual(expected, self.dummy_dataset.interpreter.label_docs(label_id, self.dummy_dataset.label_mtx))

    def test_doc_features_doc0(self):
        doc_id = 0
        feat_mtx = self.dummy_dataset.feature_mtx.copy()
        aux = np.zeros((1, self.n_features), dtype=feat_mtx.dtype)
        aux[0, 1] = 0.5
        aux[0, 6] = 0.9
        aux[0, 3] = 0.0211
        aux[0, 13] = 11.9
        feat_mtx_stub = scipy.sparse.vstack((csr_matrix(aux, shape=(1, self.n_features)), feat_mtx[1:, :]), format='csr')
        expected = "Feature_001 (0.500), Feature_003 (0.021), Feature_006 (0.900), Feature_013 (11.900)"
        self.assertEqual(expected, self.dummy_dataset.interpreter.doc_features(doc_id, feat_mtx_stub))

    def test_doc_features_doc0_no_values(self):
        doc_id = 0
        feat_mtx = self.dummy_dataset.feature_mtx.copy()
        aux = np.zeros((1, self.n_features), dtype=feat_mtx.dtype)
        aux[0, 1] = 0.5
        aux[0, 6] = 0.9
        aux[0, 3] = 0.0211
        aux[0, 13] = 11.9
        aux = csr_matrix(aux, shape=(1, self.n_features))
        feat_mtx = feat_mtx[1:, :]
        feat_mtx_stub = scipy.sparse.vstack((aux, feat_mtx), format='csr')
        expected = "Feature_001, Feature_003, Feature_006, Feature_013"
        self.assertEqual(expected, self.dummy_dataset.interpreter.doc_features(doc_id, feat_mtx_stub, print_values=False))

    def test_feature_docs(self):
        feature_id = 0
        feat_mtx = self.dummy_dataset.feature_mtx.copy()
        aux = np.zeros((self.n_docs, 1), dtype=feat_mtx.dtype)
        aux[1, 0] = 0.5
        aux[6, 0] = 0.9
        aux[3, 0] = 0.0211
        feat_mtx_stub = scipy.sparse.hstack((csr_matrix(aux), feat_mtx[:, 1:]), format='csr')
        expected = "Text_001 (0.500), Text_003 (0.021), Text_006 (0.900)"
        self.assertEqual(expected, self.dummy_dataset.interpreter.feature_docs(feature_id, feat_mtx_stub))

    def test_feature_docs_no_values(self):
        feature_id = 0
        feat_mtx = self.dummy_dataset.feature_mtx.copy()
        aux = np.zeros((self.n_docs, 1), dtype=feat_mtx.dtype)
        aux[1, 0] = 0.5
        aux[6, 0] = 0.9
        aux[3, 0] = 0.0211
        feat_mtx_stub = scipy.sparse.hstack((csr_matrix(aux), feat_mtx[:, 1:]), format='csr')
        expected = "Text_001, Text_003, Text_006"
        self.assertEqual(expected, self.dummy_dataset.interpreter.feature_docs(feature_id, feat_mtx_stub, print_values=False))

    def test_info_about_nearest_neighbors_no_dists(self):
        doc_id = 0
        nns = np.zeros((self.n_docs, 4), dtype=np.int32)
        nns[0, :] = [2, 6, 5, 7]
        dists = np.zeros((self.n_docs, 4), dtype=np.float32)
        dists[0, :] = [0.123, 0.321, 0.4, 0.82]

        expected = "1.Text_002, 2.Text_006, 3.Text_005, 4.Text_007"
        self.assertEqual(expected, self.dummy_dataset.interpreter.info_about_nearest_neighbors(nns[doc_id, :]))

    def test_info_about_nearest_neighbors_dists(self):
        doc_id = 0
        nns = np.zeros((self.n_docs, 4), dtype=np.int32)
        nns[0, :] = [2, 6, 5, 7]
        dists = np.zeros((self.n_docs, 4), dtype=np.float32)
        dists[0, :] = [0.123, 0.321, 0.4, 0.82]

        expected = "1.Text_002 (0.123), 2.Text_006 (0.321), 3.Text_005 (0.400), 4.Text_007 (0.820)"
        self.assertEqual(expected,
                         self.dummy_dataset.interpreter.info_about_nearest_neighbors(
                             nns[doc_id, :], dists[doc_id, :]))


def construct_fixed_dummy_dataset():
    name = "DummyFixedDataset"
    docs_mi = [
        Entity(orig_id=123, name='long_text'),
        Entity(orig_id=321, name='two_word_text_with_one_label'),
        Entity(orig_id=12, name='empty_text_with_many_labels'),
        Entity(orig_id=44, name='other_long_text_with_many_labels'),
        Entity(orig_id=66, name='other text')]
    labels_mi = [
        Entity(orig_id=2, name='label_2'),
        Entity(orig_id=3, name='label_3'),
        Entity(orig_id=5, name='some_label'),
        Entity(orig_id=10, name='some_label'),
        Entity(orig_id=13, name='some_label'),
        Entity(orig_id=14, name='label_'),
        Entity(orig_id=21, name='empty_label'),
        Entity(orig_id=23, name='big_label'),
        Entity(orig_id=27, name='almost_empty_label')]
    features_mi = [
        Entity(orig_id=1, name='1'),
        Entity(orig_id=2, name='2'),
        Entity(orig_id=3, name='3'),
        Entity(orig_id=4, name='not_used_feature'),
        Entity(orig_id=5, name='5'),
        Entity(orig_id=6, name='6'),
        Entity(orig_id=7, name='not_used_feature'),
        Entity(orig_id=8, name='8'),
        Entity(orig_id=9, name='9'),
        Entity(orig_id=10, name='not_used_feature'),
        Entity(orig_id=11, name='')]
    interpreter = DatasetInterpreter(docs_mi, labels_mi, features_mi)
    label_mtx = csr_matrix(np.array(
        [[1, 0, 0, 0, 1, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0, 1, 0],
         [1, 1, 1, 0, 1, 1, 0, 1, 0],
         [0, 0, 1, 0, 1, 0, 0, 0, 1]]))
    feature_mtx = csr_matrix(np.array(
        [[1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0]]))

    return Dataset(
        name=name,
        label_mtx=label_mtx,
        feature_mtx=feature_mtx,
        interpreter=interpreter)


class DatasetInterpreterCompareDocsTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = construct_fixed_dummy_dataset()

    def test_get_all_on_features_and_labels_of_selected_docs_features_1(self):
        list_of_docs = [0, 1]
        expected = [7]
        actual, _ = DatasetInterpreter._get_all_on_features_and_labels_of_selected_docs(
            list_of_docs,
            self.dataset.feature_mtx,
            self.dataset.label_mtx)
        self.assertEqual(expected, actual)

    def test_get_all_on_features_and_labels_of_selected_docs_labels_1(self):
        list_of_docs = [0, 1]
        expected = [7]
        _, actual = DatasetInterpreter._get_all_on_features_and_labels_of_selected_docs(
            list_of_docs,
            self.dataset.feature_mtx,
            self.dataset.label_mtx)
        self.assertEqual(expected, actual)


class DatasetFilterTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = construct_fixed_dummy_dataset()

        min_label_size = 2
        min_feature_size = 2
        min_labels_per_doc = 1
        min_features_per_doc = 1

        self.dataset.filter(
            min_label_size=min_label_size,
            min_feature_size=min_feature_size,
            min_labels_per_doc=min_labels_per_doc,
            min_features_per_doc=min_features_per_doc,
            verbose=0)

    def test_filter_docs_mi(self):
        # expected_docs_mi = [
        #     Entity(orig_id=123, name='long_text'),
        #     Entity(orig_id=321, name='two_word_text_with_one_label'),
        #     Entity(orig_id=44, name='other_long_text_with_many_labels'),
        #     Entity(orig_id=66, name='other text')]
        expected = [123, 321, 44, 66]
        actual = [en.orig_id for en in self.dataset.interpreter.docs_meta_info]
        self.assertEqual(expected, actual)

    def test_filter_labels_mi(self):
        # expected_labels_mi = [
        #     Entity(orig_id=2, name='label_2'),
        #     Entity(orig_id=5, name='some_label'),
        #     Entity(orig_id=13, name='some_label'),
        #     Entity(orig_id=14, name='label_'),
        #     Entity(orig_id=23, name='big_label')]
        expected = [2, 5, 13, 14, 23]
        actual = [en.orig_id for en in self.dataset.interpreter.labels_meta_info]
        self.assertEqual(expected, actual)

    def test_filter_features_mi(self):
        # expected_features_mi = [
        #     Entity(orig_id=1, name='1'),
        #     Entity(orig_id=2, name='2'),
        #     Entity(orig_id=3, name='3'),
        #     Entity(orig_id=5, name='5'),
        #     Entity(orig_id=6, name='6'),
        #     Entity(orig_id=8, name='8')]
        expected = [1, 2, 3, 5, 6, 8]
        actual = [en.orig_id for en in self.dataset.interpreter.features_meta_info]
        self.assertEqual(expected, actual)

    def test_filter_label_mtx(self):
        expected = csr_matrix(np.array(
            [[1, 0, 1, 1, 1],
             [0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 0, 0]]))
        actual = self.dataset.label_mtx
        self.assertEqual((expected != actual).nnz, 0)

    def test_filter_feature_mtx(self):
        expected = csr_matrix(np.array(
            [[1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 1],
             [1, 1, 1, 0, 1, 0],
             [0, 0, 1, 1, 0, 0]]))
        actual = self.dataset.feature_mtx
        self.assertEqual((expected != actual).nnz, 0)


if __name__ == '__main__':
    unittest.main()
