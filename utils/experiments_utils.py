from numpy import vstack
from enum import Enum
from dataio.stratifying_data import apply_stratification_better
from scipy.sparse import csr_matrix
import numpy as np


class DataPartitionMode(Enum):
    test_train = 0
    train_train = 1
    val_train = 2
    testval_train = 3


def train_test_partition_according_to_mode(X_data, t_data, stratification, test_folds, val_folds, mode):

    # extract train, validation and test data:
    train_data, val_data, test_data = apply_stratification_better(X_data, t_data, stratification,
                                                                  which_folds_is_test=test_folds,
                                                                  which_folds_is_validation=val_folds,
                                                                  verbose=0)

    if mode == DataPartitionMode.train_train:
        (x_train, t_train) = train_data
        x_test = x_train
        t_test = t_train
    elif mode == DataPartitionMode.test_train:
        (x_train, t_train) = train_data
        (x_test, t_test) = test_data
    elif mode == DataPartitionMode.testval_train:
        (x_train, t_train) = train_data
        (x_test, t_test) = test_data
        (x_val, t_val) = val_data
        x_test = vstack((x_test, x_val))
        t_test = vstack((t_test, t_val))
    elif mode == DataPartitionMode.val_train:
        (x_train, t_train) = train_data
        (x_test, t_test) = val_data
    else:
        raise ValueError('Bad mode')

    return x_train, t_train, x_test, t_test


def calculate_class_weights(label_mtx):
    """
    Formula taken from:
    http://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    :param label_mtx:
    :return:
    """
    label_cardinality = np.array(np.sum(label_mtx, axis=0)).flatten()
    n_samples, n_labels = label_mtx.shape
    # laplace n=1 smoothing (without this, there is a high risk of dividing by zero)
    label_cardinality += 1
    return n_samples / (n_labels * label_cardinality)


def calculate_sample_weights_for_multi_label_problem(label_mtx, mode='sum'):
    """
    Utility function for weighting sample components of loss function. Used in generators for Keras fit method.

    :param label_mtx: scipy.sparse.csr_matrix of sample-label indicators. shape: (n_samples, n_labels)
    :param mode: string 'sum', 'max' or 'mean': determines how to obtain a weight of a sample, basing on class_weights
    of classes assigned to that sample
    :return: numpy array of shape (n_samples) with weight for each sample
    """

    assert isinstance(label_mtx, csr_matrix)

    class_weights = calculate_class_weights(label_mtx)
    sample_weigths = np.zeros(shape=(label_mtx.shape[0]))

    weight_for_unlabeled_test_sample = np.mean(class_weights)

    if mode == 'sum':
        op = np.sum
    elif mode == 'mean':
        op = np.mean
    elif mode == 'max':
        op = np.max
    else:
        raise ValueError("Bad mode argument. Use: 'sum', 'mean' or 'max'.")

    for i_sample, (start_s, stop_s) in enumerate(zip(label_mtx.indptr, label_mtx.indptr[1:])):
        i_sample_classes_idx = label_mtx.indices[start_s:stop_s]
        cw = class_weights[i_sample_classes_idx]
        sample_weigths[i_sample] = op(cw) if cw.size > 0 else weight_for_unlabeled_test_sample

    return sample_weigths
