from dataio.data_format_transformer import from_fastText_string_predictions, from_fastText_predictions,\
    list_of_lists_to_sparse_matrix, labels_of_test_data_instance_string_fastText_format,\
    labels_of_test_data_from_fastText_format, apply_stratification
import unittest
from tempfile import NamedTemporaryFile
from scipy.sparse import csr_matrix


class DataFormatTransformerTest(unittest.TestCase):

    def test_list_of_lists_to_sparse_matrix_1(self):
        ll = [[0], [0, 2], [0, 1], [1, 2], [1], [2], [2]]
        expected = csr_matrix([[1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=bool)
        actual = list_of_lists_to_sparse_matrix(ll)
        self.assertEqual((expected != actual).nnz, 0)

    def test_list_of_lists_to_sparse_matrix_2(self):
        ll = [[0], [0, 2], [0, 1], [1, 2], [1], [2], [2]]
        expected = csr_matrix([[1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=bool)
        actual = list_of_lists_to_sparse_matrix(ll, n_labels=3)
        self.assertEqual((expected != actual).nnz, 0)

    def test_list_of_lists_to_sparse_matrix_3(self):
        ll = [[0], [0, 2], [0, 4], [1, 2], [1], [2], [2]]
        expected = csr_matrix([[1, 0, 0, 0, 0], [1, 0, 1, 0, 0], [1, 0, 0, 0, 1], [0, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], dtype=bool)
        actual = list_of_lists_to_sparse_matrix(ll)
        self.assertEqual((expected != actual).nnz, 0)

    def test_from_fastText_string_predictions(self):
        in_string = ['__label__123 0.23 __label__43 0.12 __label__5028 0.0012e-7', '__label__88 0.72 __label__33 0.01']
        result = from_fastText_string_predictions(in_string)
        expected_result = [([123, 43, 5028], [0.23, 0.12, 0.0012e-7]), ([88, 33], [0.72, 0.01])]
        self.assertTrue(result == expected_result)

    def test_from_fastText_predictions(self):
        in_list = ['__label__123 0.23 __label__43 0.12 __label__5028 0.0012e-7\n', '__label__88 0.72 __label__33 0.01\n']

        with NamedTemporaryFile('w') as f:
            f.writelines(in_list)
            f.seek(0)
            result = from_fastText_predictions(f.name)

        expected_result = [([123, 43, 5028], [0.23, 0.12, 0.0012e-7]), ([88, 33], [0.72, 0.01])]
        self.assertTrue(result == expected_result)

    def test_labels_of_test_data_instance_string_fastText_format_1(self):
        line = '__label__24883 Alfred Habdank Skarbek Korzybski (July 3, 1879   March 1, 1950) was a Polish-American ' \
               'philosopher and scientist. He is most remembered for developing the theory of general semantics. '
        self.assertEqual(labels_of_test_data_instance_string_fastText_format(line), [24883])

    def test_labels_of_test_data_instance_string_fastText_format_2(self):
        line = '__label__14655 __label__22235 This article is about the demographic features of the population of ' \
               'American Samoa, including population density, ethnicity, education level, health of the populace, ' \
               'economic status, religious affiliations and other aspects of the population. '
        self.assertEqual(labels_of_test_data_instance_string_fastText_format(line), [14655, 22235])

    def test_labels_of_test_data_from_fastText_format(self):
        content = ['__label__24883 Alfred Habdank Skarbek Korzybski (July 3, 1879   March 1, 1950) was a '
                   'Polish-American philosopher and scientist. He is most remembered for developing the theory of '
                   'general semantics.\n',
                   '__label__12910 Americium is a synthetic element that has the symbol Am and atomic number 95. A '
                   'radioactive metallic element, americium is an actinide that was obtained in 1944 by Glenn T. '
                   'Seaborg who was bombarding plutonium with neutrons and was the fourth transuranic element to be '
                   'discovered. It was named for the Americas, by analogy with europium. Americium is widely used in '
                   'commercial ionization chamber smoke detectors, as well as in neutron sources and industrial '
                   'gauges.\n',
                   '__label__14655 __label__22235 This article is about the demographic features of the population of '
                   'American Samoa, including population density, ethnicity, education level, health of the populace, '
                   'economic status, religious affiliations and other aspects of the population.\n']

        with NamedTemporaryFile('w') as f:
            f.writelines(content)
            f.seek(0)
            result = labels_of_test_data_from_fastText_format(f.name)

        expected_result = [[24883], [12910], [14655, 22235]]
        self.assertTrue(result == expected_result)

    def test_labels_of_test_data_from_fastText_format_diff_ending(self):
        content = ['__label__24883 Alfred Habdank Skarbek Korzybski (July 3, 1879   March 1, 1950) was a '
                   'Polish-American philosopher and scientist. He is most remembered for developing the theory of '
                   'general semantics.\n',
                   '__label__12910 Americium is a synthetic element that has the symbol Am and atomic number 95. A '
                   'radioactive metallic element, americium is an actinide that was obtained in 1944 by Glenn T. '
                   'Seaborg who was bombarding plutonium with neutrons and was the fourth transuranic element to be '
                   'discovered. It was named for the Americas, by analogy with europium. Americium is widely used in '
                   'commercial ionization chamber smoke detectors, as well as in neutron sources and industrial '
                   'gauges.\n',
                   '__label__14655 __label__22235 This article is about the demographic features of the population of '
                   'American Samoa, including population density, ethnicity, education level, health of the populace, '
                   'economic status, religious affiliations and other aspects of the population.']

        with NamedTemporaryFile('w') as f:
            f.writelines(content)
            f.seek(0)
            result = labels_of_test_data_from_fastText_format(f.name)

        expected_result = [[24883], [12910], [14655, 22235]]
        self.assertTrue(result == expected_result)

    def test_apply_stratification(self):
        data = csr_matrix([[1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=bool)
        stratification = [[0, 2, 4], [1, 3], [5, 6]]
        fold = 0
        [train_data, test_data] = apply_stratification(data, stratification, fold)
        expected_train_data = csr_matrix([[1, 0, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool)
        expected_test_data = csr_matrix([[1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=bool)
        self.assertEqual((expected_train_data != train_data).nnz, 0)
        self.assertEqual((expected_test_data != test_data).nnz, 0)


if __name__ == '__main__':
    unittest.main()
