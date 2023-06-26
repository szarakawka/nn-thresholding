import numpy as np
from scipy.sparse import isspmatrix_csr, lil_matrix, csr_matrix
from utils.etc import pi_range, pi_sequence_timer
from sklearn.metrics import f1_score
from dataio.stratifying_data import iterative_stratification, apply_stratification_better
import os.path
import pickle
import tqdm
from multiprocessing import Pool


def scale_scores(scores: csr_matrix):
    """
    Scales scores dividing each score by the highest score in a given sample.

    :param scores: scipy.sparse.csr_matrix of shape [nSamples, nLabels]
    :return: scipy.sparse.csr_matrix of shape [nSamples, nLabels]
    """
    if not isinstance(scores, csr_matrix):
        raise ValueError('Scores has to be of csr_matrix format!')

    scores_scaled = scores.copy()

    for n, (start, stop) in enumerate(zip(scores.indptr, scores.indptr[1:])):
        max_value = np.max(scores.data[start:stop])
        assert max_value != 0
        scores_scaled.data[start:stop] = scores_scaled.data[start:stop] / max_value
    return scores_scaled


def r_cut(scores, rank):
    assert isspmatrix_csr(scores)
    predictions = lil_matrix(scores.shape, dtype=np.bool)
    for i, (start_s, stop_s) in enumerate(zip(scores.indptr, scores.indptr[1:])):
        aux_idx = scores.data[start_s:stop_s].argsort()[-rank:]
        predicted_labels_idxs = scores.indices[start_s:stop_s][aux_idx]
        predictions.rows[i] = predicted_labels_idxs.tolist()
        predictions.data[i] = np.ones(predicted_labels_idxs.shape, dtype=np.bool).tolist()
    return predictions.tocsr()


def s_cut(scores, th, ensure_no_empty_predictions_flag=True):
    assert isspmatrix_csr(scores)
    predictions = scores.__gt__(th)
    if ensure_no_empty_predictions_flag:
        return ensure_no_empty_predictions(predictions, scores)
    else:
        return predictions


def p_cut(scores, th, ensure_no_empty_predictions_flag=True, proportionality_const=1.):
    """
    p-cut, based on Lewis, David D. "An evaluation of phrasal and clustered representations on a text categorization
    task." Proceedings of the 15th annual international ACM SIGIR conference on Research and development in
    information retrieval. 1992.
    It could return less predictions per label than proportion suggests, if there is less positive scores for this
    label than needed (because taking random zero scored samples is worthless).

    :param scores: sparse csr_matrix with scores of size [nSamples x nLabels]
    :param th: array of proportions learned during training of length nLabels
    :param ensure_no_empty_predictions_flag:
    :param proportionality_const:
    :return: sparse csr_matrix with bool predictions [nSamples x nLabels]
    """

    assert isspmatrix_csr(scores)
    scores_t = scores.transpose(copy=True).tocsr()
    thresholds = np.array(th)
    thresholds *= proportionality_const
    thresholds = np.maximum(np.minimum(thresholds, 1.0), 0.0)
    n_to_take = [int(n) for n in np.floor(thresholds * scores.shape[0])]
    assert scores.shape[1] == len(n_to_take)

    predictions_t = lil_matrix(scores_t.shape, dtype=np.bool)
    for i_label, (start_s, stop_s) in enumerate(zip(scores_t.indptr, scores_t.indptr[1:])):
        n = n_to_take[i_label]
        if n == 0:
            predictions_t.rows[i_label] = []
            predictions_t.data[i_label] = []
        else:
            sc_values = scores_t.data[start_s:stop_s]
            order = np.argsort(sc_values)[-n:]    # if n > len(order), then it works as if n == len(order)
            x_idx = scores_t.indices[start_s:stop_s]
            predicted_x_idxs = x_idx[order]
            predictions_t.rows[i_label] = [x for x in predicted_x_idxs]
            predictions_t.data[i_label] = [True for _ in predicted_x_idxs]
    predictions_t = predictions_t.tocsc()
    predictions = predictions_t.transpose()     # this will be csr_matrix

    if ensure_no_empty_predictions_flag:
        return ensure_no_empty_predictions(predictions, scores)
    else:
        return predictions


def ss_cut(scores, th, ensure_no_empty_predictions_flag=True):
    scaled = scale_scores(scores)
    return s_cut(scaled, th, ensure_no_empty_predictions_flag)


def ps_cut(scores, th, ensure_no_empty_predictions_flag=True, proportionality_const=1.):
    scaled = scale_scores(scores)
    return p_cut(scaled, th, ensure_no_empty_predictions_flag, proportionality_const)


def ds_cut():
    # """
    #
    # :param scores: scipy.sparse.csr_matrix [nSamples, nLabels]
    # :param ds: list or 1d-array of length n of distinctive threshold values per rank up to rank n
    # :return: predictions: scipy.sparse.csr_matrix [nSamples, nLabels] with boolean values
    # """

    raise NotImplementedError


def cs_cut(scores, thresholds, ensure_no_empty_predictions_flag=True):
    assert isspmatrix_csr(scores)
    thresholds = np.array(thresholds)
    assert scores.shape[1] == len(thresholds)

    predictions = lil_matrix(scores.shape, dtype=np.bool)
    for i, (start_s, stop_s) in enumerate(zip(scores.indptr, scores.indptr[1:])):
        col_idx = scores.indices[start_s:stop_s]
        thresholds_values = thresholds[col_idx]
        sc_values = scores.data[start_s:stop_s]
        aux_mask = sc_values.__gt__(thresholds_values)
        predicted_labels_idxs = col_idx[aux_mask]
        predictions.rows[i] = predicted_labels_idxs.tolist()
        predictions.data[i] = np.ones(predicted_labels_idxs.shape, dtype=np.bool).tolist()
    predictions = predictions.tocsr()
    if ensure_no_empty_predictions_flag:
        return ensure_no_empty_predictions(predictions, scores)
    else:
        return predictions


def css_cut(scores, th, ensure_no_empty_predictions_flag=True):
    scaled = scale_scores(scores)
    return cs_cut(scaled, th, ensure_no_empty_predictions_flag)


def ensure_no_empty_predictions(predictions, scores):
    """ Ensures that at least one label is always returned.
    :param predictions: csr_matrix
    :param scores: csr_matrix
    :return predictions with all rows with at least one nonzero element, csr_matrix
    """
    n_labels_per_sample = np.array(np.sum(predictions, axis=1), dtype=np.int32).flatten()
    if not np.any(n_labels_per_sample == 0):
        return predictions
    else:
        predictions_lil = predictions.tolil()
        rows_to_take_max_from = np.where(n_labels_per_sample == 0)[0]
        for row_idx in rows_to_take_max_from:
            highest_scoring_label_idx = np.argmax(scores[row_idx, :].toarray())
            predictions_lil.rows[row_idx] = [highest_scoring_label_idx]
            predictions_lil.data[row_idx] = [True]
        return predictions_lil.tocsr()


# -------------------------------Thresholding-learning-methods-----------------------------------------------


def fit_r_cut(train_scores, train_labels, verbose=1):
    # r_candidates = np.arange(5) + 1
    r_candidates = np.unique(np.ceil(np.linspace(1, np.percentile(train_labels.sum(axis=1), 90), 50)).astype(np.int32))

    # loop over rank candidates
    f1_per_r_value = np.zeros(len(r_candidates))

    for i_r, (r_value, pi) in enumerate(pi_sequence_timer(r_candidates)):
        if pi and verbose > 0:
            print("\rRank={}/{}, Progress={:.2f}, elapsed time={:.2f}, ETA={:.2f}".format(
                i_r+1, len(r_candidates), pi.progress, pi.elapsed_time, pi.time_remaining()), end='')

        predictions = r_cut(train_scores, r_value)
        f1_per_r_value[i_r] = f1_score(train_labels, predictions, average='macro')

    where_best = np.argmax(f1_per_r_value)
    # logging.debug("thr_candidates = {}".format(thr_candidates))
    # logging.debug("maF per th values = {}".format(f1_per_th_value))
    return r_candidates[where_best], f1_per_r_value[where_best]


def fit_p_cut(train_scores, train_labels, verbose=1):
    [n_samples, n_labels] = train_labels.shape
    labels_t = train_labels.transpose().tocsr()
    proportions = np.zeros(n_labels, dtype=np.float32)

    for i_label, (start_s, stop_s) in enumerate(zip(labels_t.indptr, labels_t.indptr[1:])):
        proportions[i_label] = (stop_s - start_s) / n_samples

    return proportions, 1.0


def _get_thr_candidates(non_zero_scores):
    """
    Gives up to 50 threshold candidate values.
    :param non_zero_scores: np.array
    :return: np.array of candidate threshold values
    """
    min_nonzero_score = np.min(non_zero_scores)
    scores = np.hstack((min_nonzero_score/2., non_zero_scores))
    thr_candidates = np.unique(np.percentile(scores, np.arange(0, 102, 2)))   # check max 50 threshold values
    return thr_candidates


def fit_s_cut(train_scores, train_labels, verbose=1):
    """
    FBR0 or FBR1 variants (from Yang paper) are not implemented, because it is not obvious how to define 'small'
    category.

    :param train_scores: scipy.sparse.csr_matrix of shape [nTrainExamples, nLabels]
    :param train_labels: scipy.sparse.csr_matrix of shape [nTrainExamples, nLabels] binary values
    :param verbose:
    :return: numpy.ndarray of shape [nLabels] with optimal threshold values,
             numpy.ndarray of shape [nLabels] bestF1scores at optimal thresholds
    """

    thr_candidates = _get_thr_candidates(train_scores.data)

    # loop over thresholds
    f1_per_th_value = np.zeros(len(thr_candidates))

    for i_th, (th_value, pi) in enumerate(pi_sequence_timer(thr_candidates)):
        if pi and verbose > 0:
            print("\rThreshold={}/{}, Progress={:.2f}, elapsed time={:.2f}, ETA={:.2f}".format(
                i_th+1, len(thr_candidates), pi.progress, pi.elapsed_time, pi.time_remaining()), end='')

        predictions = s_cut(train_scores, th_value)
        f1_per_th_value[i_th] = f1_score(train_labels, predictions, average='macro')

    where_best = np.argmax(f1_per_th_value)
    # logging.debug("thr_candidates = {}".format(thr_candidates))
    # logging.debug("maF per th values = {}".format(f1_per_th_value))
    return thr_candidates[where_best], f1_per_th_value[where_best]


def fit_ss_cut(train_scores, train_labels, verbose=1):
    scaled = scale_scores(train_scores)
    return fit_s_cut(scaled, train_labels, verbose)


def fit_ps_cut(train_scores, train_labels, verbose=1):
    # scaled = scale_scores(train_scores)               # scores are not important for fitting p-cut or ps-cut
    return fit_p_cut(train_scores, train_labels, verbose)


def fit_cs_cut(train_scores, train_labels, verbose=1, default_threshold_value=1.0, n_processes=1):
    """
    In this version, threshold values is not th[np.argmax(f1)], but (th[np.argmax(f1)]+th[np.argmax(f1)-1])/2.

    :param train_scores: scipy.sparse.csr_matrix of shape [nTrainExamples, nLabels]
    :param train_labels: scipy.sparse.csr_matrix of shape [nTrainExamples, nLabels] binary values
    :param default_threshold_value: in a degraded case, there may exist labels without any train examples, then
    this is the value of the threshold for them
    :param verbose:
    :param n_processes: if > 1, then parallel version is called
    :return: numpy.ndarray of shape [nLabels] with optimal threshold values,
             numpy.ndarray of shape [nLabels] bestF1scores at optimal thresholds
    """

    if n_processes > 1:
        return NotImplementedError
        # return fit_cs_cut_parallel(train_scores, train_labels, verbose, default_threshold_value, n_processes)

    # single process version, but with nice status printing:

    n_samples, n_labels = train_labels.shape
    thresholds = np.empty(n_labels)
    best_f1 = np.empty(n_labels)

    train_scores_t = train_scores.transpose(copy=True).tocsr()
    train_labels_t = train_labels.transpose(copy=True).tocsr()

    for i_label, pi in pi_range(n_labels):
        if pi and verbose > 0:
            print("\rLabel={}/{}, Progress={:.2f}, elapsed time={:.2f}, ETA={:.2f}".format(
                i_label+1, n_labels, pi.progress, pi.elapsed_time, pi.time_remaining()), end='')

        label_indicator = train_labels_t[i_label, :]
        scores_per_label = train_scores_t[i_label, :]

        if len(scores_per_label.data) == 0:     # degraded case: a label without any train examples
            thresholds[i_label] = default_threshold_value
            best_f1[i_label] = 0.
            continue

        thr_candidates = _get_thr_candidates(scores_per_label.data)

        # loop over thresholds
        f1_per_th_value = np.zeros(len(thr_candidates))
        for i_th, (th_value, pi_2) in enumerate(pi_sequence_timer(thr_candidates)):
            if pi_2 and verbose > 1:
                print("\rThreshold={}/{}, Progress={:.2f}, elapsed time={:.2f}, ETA={:.2f}".format(
                    i_th+1, len(thr_candidates), pi_2.progress, pi_2.elapsed_time, pi_2.time_remaining()), end='')

            predictions = s_cut(scores_per_label, th_value)
            f1_per_th_value[i_th] = f1_score(label_indicator.toarray().flatten(),
                                             predictions.astype(np.float32).toarray().flatten())

        where_best = np.argmax(f1_per_th_value)

        thresholds[i_label] = thr_candidates[where_best]
        best_f1[i_label] = f1_per_th_value[where_best]

    return thresholds, best_f1


def _worker_process(args):
    [label_indicator, scores_per_label, default_threshold_value] = args
    result_thr = default_threshold_value
    result_f1 = 0.
    if len(scores_per_label.data) == 0:     # degraded case: a label without any train examples
        return result_thr, result_f1

    thr_candidates = _get_thr_candidates(scores_per_label.data)

    # loop over thresholds
    f1_per_th_value = np.zeros(len(thr_candidates))
    for i_th, th_value in enumerate(thr_candidates):
        predictions = s_cut(scores_per_label, th_value)
        f1_per_th_value[i_th] = f1_score(label_indicator.toarray().flatten(),
                                         predictions.astype(np.float32).toarray().flatten())
    where_best = np.argmax(f1_per_th_value)
    result_thr = thr_candidates[where_best]
    result_f1 = f1_per_th_value[where_best]

    return result_thr, result_f1


def fit_cs_cut_parallel(train_scores, train_labels, verbose=1, default_threshold_value=1.0, n_processes=6):

    from tqdm.contrib.concurrent import process_map

    train_scores_t = train_scores.transpose(copy=True).tocsr()
    train_labels_t = train_labels.transpose(copy=True).tocsr()

    gen = ((train_labels_t[i_label, :], train_scores_t[i_label, :], default_threshold_value)
           for i_label in range(train_labels.shape[1]))

    if verbose > 0:
        results = process_map(_worker_process, gen, max_workers=n_processes, total=train_labels.shape[1])
    else:
        pool = Pool(n_processes)
        results = pool.map(_worker_process, gen)

    thresholds = np.asarray([e[0] for e in results])
    best_f1 = np.asarray([e[1] for e in results])
    return thresholds, best_f1


def fit_css_cut(train_scores, train_labels, verbose=1, default_threshold_value=1.0,  n_processes=1):
    scaled = scale_scores(train_scores)
    if n_processes > 1:
        # raise NotImplementedError
        return fit_cs_cut_parallel(scaled, train_labels, verbose, default_threshold_value, n_processes)
    else:
        return fit_cs_cut(scaled, train_labels, verbose, default_threshold_value)


# ----------------------- grid search over methods to find the best one --------------------------------------


traditional_methods_dict = {
    "s": (fit_s_cut, s_cut),
    "ss": (fit_ss_cut, ss_cut),
    "cs": (fit_cs_cut, cs_cut),
    "css": (fit_css_cut, css_cut),
    "r": (fit_r_cut, r_cut),
    "p": (fit_p_cut, p_cut),
    "ps": (fit_ps_cut, ps_cut)}


def single_run_of_thresholding_optimization_based_on_train_metrics(_, hyperparams, **kwargs):
    scores = kwargs['scores_train']
    gt_labels = kwargs['labels']
    (fit_method, predict_method) = traditional_methods_dict[hyperparams['thr_method']]

    # thresholding training
    verbose = kwargs['verbose'] if 'verbose' in kwargs else 0
    thresholds, f1 = fit_method(scores, gt_labels, verbose=verbose)

    # save thresholds
    with open(os.path.join(kwargs['run_dir'], hyperparams['thr_method'] + '.pkl'), 'wb') as f:
        pickle.dump(thresholds, f, pickle.HIGHEST_PROTOCOL)

    if isinstance(f1, np.ndarray):
        f1 = np.mean(f1)        # macroF1 in case of CS and CSS

    return {'ma_f': f1}


def single_run_of_thresholding_optimization_based_on_validation_metrics(_, hyperparams, **kwargs):

    scores = kwargs['scores_train']
    gt_labels = kwargs['labels']
    (fit_method, predict_method) = traditional_methods_dict[hyperparams['thr_method']]

    stratification = iterative_stratification(gt_labels, n_folds=10)
    (scores_train, gt_labels_train), (scores_validation, gt_labels_validation), _ = apply_stratification_better(
        scores, gt_labels, stratification, [], which_folds_is_validation=[0])

    # thresholding training
    verbose = kwargs['verbose'] if 'verbose' in kwargs else 0
    thresholds, _ = fit_method(scores_train, gt_labels_train, verbose=verbose)

    # save thresholds
    with open(os.path.join(kwargs['run_dir'], hyperparams['thr_method'] + '.pkl'), 'wb') as f:
        pickle.dump(thresholds, f, pickle.HIGHEST_PROTOCOL)

    # validation
    predictions = predict_method(scores_validation, thresholds)

    # evaluating
    ma_f = f1_score(y_true=gt_labels_validation, y_pred=predictions, average='macro')
    mi_f = f1_score(y_true=gt_labels_validation, y_pred=predictions, average='micro')

    return {'ma_f': ma_f, 'mi_f': mi_f}


results_mapper_dict_for_thresholding_optimization = {
    'ma_f': lambda x: x['ma_f'],
    'mi_f': lambda x: x['mi_f'],
}
