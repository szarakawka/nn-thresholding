import scipy.sparse.csc
import numpy as np


def column_calculate_tp_fp_fn(gt, pred):
    """ Calculates true positives, false positives and false negatives per label (per column) in predictions (pred)
    compared to ground truth (gt).
    
    :param gt: csc_matrix of shape [nSamples, nLabels] with ground truth bipartition
    :param pred: csc_matrix of shape [nSamples, nLabels] with predicted bipartition
    :return: np.array [nLabels] with counts of true positives per label, np.array [nLabels] with counts of false
    positives per label, np.array [nLabels] with counts of false negatives per label
    """
    if not (isinstance(gt, scipy.sparse.csc.csc_matrix) and isinstance(pred, scipy.sparse.csc.csc_matrix)):
        raise TypeError

    tp = np.zeros(gt.shape[1], dtype=int)
    fp = np.zeros(gt.shape[1], dtype=int)
    fn = np.zeros(gt.shape[1], dtype=int)
    for i_label, (gt_start, gt_stop, p_start, p_stop) in enumerate(zip(gt.indptr, gt.indptr[1:],
                                                                       pred.indptr, pred.indptr[1:])):
        tp[i_label] = np.intersect1d(gt.indices[gt_start:gt_stop], pred.indices[p_start:p_stop],
                                     assume_unique=True).size
        fp[i_label] = np.setdiff1d(pred.indices[p_start:p_stop], gt.indices[gt_start:gt_stop],
                                   assume_unique=True).size
        fn[i_label] = np.setdiff1d(gt.indices[gt_start:gt_stop], pred.indices[p_start:p_stop],
                                   assume_unique=True).size
    return tp, fp, fn


def column_f1_per_label(gt, pred):
    """ Efficiently calculates f1_scores per label (per column) of predictions (pred) compared to ground truth (gt).
    
    :param gt: csc_matrix of shape [nSamples, nLabels] with ground truth bipartition
    :param pred: csc_matrix of shape [nSamples, nLabels] with predicted bipartition
    :return: np.array [nLabels] with f1_scores per label
    """
    tp, fp, fn = column_calculate_tp_fp_fn(gt, pred)
    f1_per_label = 2. * tp / (2. * tp + fp + fn)
    return f1_per_label
