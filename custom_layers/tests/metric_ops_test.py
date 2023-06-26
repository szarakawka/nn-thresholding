from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras.models import Model
import numpy as np
import unittest
from custom_layers.metric_ops import macro_precision, micro_precision, macro_recall, micro_recall, macro_f1score, _f1scores_per_label
from tensorflow_addons.metrics import F1Score
# from evaluation.metrics import evaluate_multilabel_predictions_f1


class CustomMetricsTestCase1(unittest.TestCase):
    def setUp(self):
        self.y_test = np.array([
            [1, 0, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        self.y_pred = np.array([
            [1, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1]
        ])

    def test_macro_precision(self):
        expected = np.mean([0.75, 1., 0.6])

        x = Input(shape=(3, ))
        model = Model(inputs=x, outputs=x)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=[macro_precision])
        result = model.evaluate(self.y_pred, self.y_test)
        self.assertAlmostEqual(expected, result[1])

    def test_micro_precision(self):
        expected = 0.7

        x = Input(shape=(3, ))
        model = Model(inputs=x, outputs=x)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=[micro_precision])
        result = model.evaluate(self.y_pred, self.y_test)
        self.assertAlmostEqual(expected, result[1])

    def test_macro_recall(self):
        expected = np.mean([1., 0.5, 1.])

        x = Input(shape=(3, ))
        model = Model(inputs=x, outputs=x)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=[macro_recall])
        result = model.evaluate(self.y_pred, self.y_test)
        self.assertAlmostEqual(expected, result[1])

    def test_micro_recall(self):
        expected = 7./8.

        x = Input(shape=(3, ))
        model = Model(inputs=x, outputs=x)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=[micro_recall])
        result = model.evaluate(self.y_pred, self.y_test)
        self.assertAlmostEqual(expected, result[1])

    def test_macro_f1score(self):
        precisions = np.array([0.75, 1., 0.6])
        recalls = np.array([1., 0.5, 1.])
        f1_per_label = 2.*precisions*recalls/(precisions+recalls)
        print(f1_per_label)
        expected_maF = np.mean(f1_per_label)
        from sklearn.metrics import f1_score
        expected_2 = f1_score(self.y_test, self.y_pred, average='macro')
        print('macro f1score:', expected_maF, ", from scipy:", expected_2)

        x = Input(shape=(3, ))
        model = Model(inputs=x, outputs=x)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=[macro_f1score])
        result = model.evaluate(self.y_pred, self.y_test)
        print("expected={}, actual={}".format(expected_maF, result[1]))
        self.assertAlmostEqual(expected_maF, result[1])


class AddonsMetricsTestCase1(unittest.TestCase):
    def setUp(self):
        self.y_test = np.array([
            [1, 0, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        self.y_pred = np.array([
            [1, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1]
        ])

    def test_macro_f1score(self):
        precisions = np.array([0.75, 1., 0.6])
        recalls = np.array([1., 0.5, 1.])
        f1_per_label = 2.*precisions*recalls/(precisions+recalls)
        print(f1_per_label)
        expected_maF = np.mean(f1_per_label)
        from sklearn.metrics import f1_score
        expected_2 = f1_score(self.y_test, self.y_pred, average='macro')
        print('macro f1score:', expected_maF, ", from scipy:", expected_2)

        x = Input(shape=(3, ))
        model = Model(inputs=x, outputs=x)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=[F1Score(num_classes=3, average='macro', threshold=0.5, name='f1_score_addon')])
        result = model.evaluate(self.y_pred, self.y_test)
        print("expected={}, actual={}".format(expected_maF, result[1]))
        self.assertAlmostEqual(expected_maF, result[1])

    def test_micro_f1score(self):
        from sklearn.metrics import f1_score
        expected = f1_score(self.y_test, self.y_pred, average='micro')

        x = Input(shape=(3, ))
        model = Model(inputs=x, outputs=x)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=[F1Score(num_classes=3, average='micro', threshold=0.5, name='f1_score_addon')])
        result = model.evaluate(self.y_pred, self.y_test)
        print("expected={}, actual={}".format(expected, result[1]))
        self.assertAlmostEqual(expected, result[1])


if __name__ == '__main__':
    unittest.main()
