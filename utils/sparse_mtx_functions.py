import numpy as np
from scipy.sparse import csr_matrix


def filter_sparse_matrix(sparse_mtx_to_filter, min_nonzeros_per_row, min_nonzeros_per_col, max_nonzeros_per_col='inf',
                         verbose=0):
    """
        Iteratively removes:
            - rows with less than min_nonzeros_per_row non-zero elements
            - cols with less than min_nonzeros_per_col non-zero elements
        until these operations do not change the matrix
    """
    mtx = sparse_mtx_to_filter.tocsr(copy=True)
    rows_to_leave = np.arange(mtx.shape[0])
    cols_to_leave = np.arange(mtx.shape[1])

    filtering_iterations = 0
    do_filtering = True
    while do_filtering:
        do_filtering = False
        filtering_iterations += 1
        if verbose > 0:
            print('Filtering: iter ', filtering_iterations, ": ", end=' ')

        # remove rows
        row_sizes = np.array(np.sum(mtx > 0, axis=1)).flatten()
        to_leave_mask = row_sizes >= min_nonzeros_per_row
        nb_rows_to_delete = len(to_leave_mask) - np.sum(to_leave_mask)
        if nb_rows_to_delete > 0:
            do_filtering = True
        rows_to_leave = rows_to_leave[to_leave_mask]
        mtx = mtx[to_leave_mask, :].tocsc(copy=True)

        if verbose > 0:
            print(nb_rows_to_delete, " rows &", end=' ')

        # remove cols
        col_sizes = np.array(np.sum(mtx > 0, axis=0)).flatten()
        to_leave_mask = col_sizes >= min_nonzeros_per_col
        if max_nonzeros_per_col != 'inf':
            to_leave_mask = np.logical_and(to_leave_mask, col_sizes < max_nonzeros_per_col)
        nb_cols_to_delete = len(to_leave_mask) - np.sum(to_leave_mask)
        if nb_cols_to_delete > 0:
            do_filtering = True
        cols_to_leave = cols_to_leave[to_leave_mask]
        mtx = mtx[:, to_leave_mask].tocsr(copy=True)

        if verbose > 0:
            print(nb_cols_to_delete, " cols deleted.")

    removed_rows = np.setdiff1d(np.arange(sparse_mtx_to_filter.shape[0]), rows_to_leave)
    removed_cols = np.setdiff1d(np.arange(sparse_mtx_to_filter.shape[1]), cols_to_leave)

    was_changed = filtering_iterations > 1

    return mtx, removed_rows, removed_cols, was_changed


def remove_columns_in_csr_matrix(matrix, columns_to_remove):
    mtx = matrix.tocsc(copy=True)
    to_leave_mask = np.ones(mtx.shape[1], dtype=np.bool)
    to_leave_mask[columns_to_remove] = False
    return mtx[:, to_leave_mask].tocsr(copy=True)


def remove_rows_in_csr_matrix(matrix, rows_to_remove):
    to_leave_mask = np.ones(matrix.shape[0], dtype=np.bool)
    to_leave_mask[rows_to_remove] = False
    return matrix[to_leave_mask, :]


def dense_outputs_to_sparse_predictions(predictions_float, ensure_at_least_one_predicted=True, threshold=0.5):
    """

    :param predictions_float: np.array [nSamples, nLabels] OR EagerTensor
    :param ensure_at_least_one_predicted:  bool flag
    :param threshold:
    :return: csr_matrix [nSamples, nLabels]
    """

    # binary_predictions = (predictions_float > threshold).astype(np.int32)
    binary_predictions = np.array(predictions_float > threshold, dtype=np.int32)    # need when predictions_float is an EagerTensor
    if ensure_at_least_one_predicted:
        which_rows = np.sum(binary_predictions, axis=1) == 0
        binary_predictions[np.arange(binary_predictions.shape[0])[which_rows],
                           np.argmax(predictions_float, axis=1)[which_rows]] = 1.
    return csr_matrix(binary_predictions)


def blend(sm1: csr_matrix, sm2: csr_matrix, alpha: float):
    assert alpha >= 0.0
    assert alpha <= 1.0

    tmp = sm1*(1.0-alpha) + sm2*alpha
    return tmp.sorted_indices()
