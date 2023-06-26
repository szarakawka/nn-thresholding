import numpy as np


def ensure_float32(input_array):
    return input_array.astype(np.float32)


def does_contain_no_nans(input_array):
    return False if np.sum(np.isnan(input_array)) > 0 else True


def is_ndarray(input_array):
    return isinstance(input_array, np.ndarray)


def has_shape_as_expected(input_array, shape):
    return input_array.shape == shape


def check_if_everything_ok(input_array, shape=None):
    if not is_ndarray(input_array):
        raise AssertionError('Not an array')
    if not does_contain_no_nans(input_array):
        raise AssertionError('Contains NaNs!')
    if shape is not None:
        if not has_shape_as_expected(input_array, shape):
            raise AssertionError('Wrong shape {}, instead of expected {}'.format(input_array.shape, shape))
    print("Input array is ok.")
