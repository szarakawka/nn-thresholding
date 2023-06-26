import os.path
import argparse
import numpy as np
import pickle
import json
import logging
from scipy.sparse import csr_matrix, csc_matrix
from scipy.stats import ttest_rel
from typing import List
from bin.paths import TestCVPhasesComponentsPaths, TrainCVPhasesComponentsPaths, PickableComponents, Subset
from bin.kadrnn_pipeline_test import find_best_fold_according_to_metric
from evaluation.metrics import evaluate_predictions_f1_per_class, evaluate_predictions


th_classic_methods = ['css']
th_neural_methods = ['t2max', 't2none']


def determine_method_with_higher_metric_per_label(list_of_f1_per_label_per_method: List):
    a = np.vstack(list_of_f1_per_label_per_method)
    best_method_per_label = a.argmax(axis=0)
    return best_method_per_label


def are_methods_significantly_different_for_label(per_fold_results_method1, per_fold_results_method2, alpha_level=0.05):
    t_stat, p_value = ttest_rel(per_fold_results_method1, per_fold_results_method2)
    p_value = 1. if np.isnan(p_value) else p_value
    return True if p_value < alpha_level else False


def are_methods_significantly_different_per_label(list_of_f1_per_label_per_method):
    assert len(list_of_f1_per_label_per_method) == 2
    n_labels = list_of_f1_per_label_per_method[0].shape[1]
    logging.debug('n_labels = {}'.format(n_labels))
    are_methods_significantly_different = np.zeros(n_labels, dtype=np.bool)
    for i_label in range(n_labels):
        m1_res = list_of_f1_per_label_per_method[0][:, i_label]
        m2_res = list_of_f1_per_label_per_method[1][:, i_label]
        are_methods_significantly_different[i_label] = are_methods_significantly_different_for_label(m1_res, m2_res)
    return are_methods_significantly_different


def determine_which_method_is_best_per_label_with_statistical_significance(
        list_of_f1_per_label_per_fold_per_method: List, default_method=0):
    avg_f1_per_labels_per_th_method = [results.mean(axis=0) for results in list_of_f1_per_label_per_fold_per_method]
    best_method_per_label = determine_method_with_higher_metric_per_label(avg_f1_per_labels_per_th_method)
    are_methods_significantly_different = are_methods_significantly_different_per_label(
        list_of_f1_per_label_per_fold_per_method)
    best_method_per_label[~are_methods_significantly_different] = default_method
    return best_method_per_label, are_methods_significantly_different


def get_f1_per_labels_from_i_fold(root_data_dir: str, scorer: str, th_version: str, i_fold: int):
    components_fps = TrainCVPhasesComponentsPaths(root_data_dir, i_fold, scorer, th_version)
    val_predictions = components_fps.get_pickable_component(PickableComponents.PREDICTIONS, Subset.VALIDATION)
    val_gt_labels = components_fps.get_pickable_component(PickableComponents.LABEL_MTX, Subset.VALIDATION)
    f1_per_labels = evaluate_predictions_f1_per_class(val_predictions, val_gt_labels)
    f1_per_labels[np.isnan(f1_per_labels)] = 0.
    return f1_per_labels


# TODO change this fixed 10
def get_f1_per_labels_per_folds(root_data_dir, scorer, th_version):
    return np.vstack([get_f1_per_labels_from_i_fold(root_data_dir, scorer, th_version, i_fold) for i_fold in range(10)])


def combine_predictions(predictions, method_indicator_per_label) -> csr_matrix:
    """
    :param predictions: list of csr_matrix [nExamples, nLabels] of len n_methods
    :param method_indicator_per_label: np.array of size [1, nLabels] with index indicators (form 0 to n_methods-1)

    :returns csr_matrix  [nExamples, nLabels]
    """

    # transform predictions into csr_matrices
    csc_predictions = [csc_matrix(pred_mat, dtype=int) for pred_mat in predictions]

    # combine
    predictions_hybrid = csc_predictions[0]
    predictions_hybrid[:, method_indicator_per_label != 0] = 0
    predictions_hybrid.eliminate_zeros()
    for i, pred_mtx in zip(range(1, len(csc_predictions)), csc_predictions[1:]):
        pred_mtx[:, method_indicator_per_label != i] = 0
        predictions_hybrid += pred_mtx
        predictions_hybrid.eliminate_zeros()

    # transform back to csr_matrix
    predictions_hybrid = csr_matrix(predictions_hybrid, dtype=bool)

    return predictions_hybrid


def hybrid_model(root_data_dir: str, th_methods: List, scorer: str = 'kadrnn', test_mode: str = 'single_best') -> None:

    for th_method in th_methods:
        assert th_method in th_classic_methods or th_method in th_neural_methods

    # 1. get f1_per_labels per method
    val_f1_per_labels_per_th_method = []
    if test_mode == 'single_best':
        for th in th_methods:
            best_fold = find_best_fold_according_to_metric(root_data_dir, scorer, th)
            val_f1_per_labels_per_th_method.append(get_f1_per_labels_from_i_fold(root_data_dir, scorer, th, best_fold))
    elif test_mode in ['soft_ensemble', 'hard_ensemble']:
        for th in th_methods:
            a = get_f1_per_labels_per_folds(root_data_dir, scorer, th)
            val_f1_per_labels_per_th_method.append(a.mean(axis=0))
    else:
        raise ValueError('Bad argument test_mode. Should be one of the following: [\'single_best\', '
                         '\'soft_ensemble\', \'hard_ensemble\'')

    # 2. based on val predictions determine which method to use (on per class level) -> new_method_indicator_per_class
    method_indicator_per_label = determine_method_with_higher_metric_per_label(val_f1_per_labels_per_th_method)

    # 3. load test predictions from all the methods
    which_cv_model = test_mode if test_mode != 'single_best' else 0
    test_predictions = []
    for th in th_methods:
        p = TestCVPhasesComponentsPaths(root_data_dir, scorer, th, which_cv_model)
        test_predictions.append(p.get_pickable_component(PickableComponents.PREDICTIONS))

    # 4. combine predictions based on new_method_indicator_per_class
    combined_predictions = combine_predictions(test_predictions, method_indicator_per_label)

    # 5. save this combined predictions
    p = TestCVPhasesComponentsPaths(root_data_dir, scorer, th_methods[0], 0)
    hybrid_dir = os.path.join(p.test_scores_dir, '_'.join(th_methods))
    os.makedirs(hybrid_dir, exist_ok=True)
    with open(os.path.join(hybrid_dir, 'predictions_' + test_mode + '.pkl'), 'wb') as f:
        pickle.dump(combined_predictions, f, pickle.HIGHEST_PROTOCOL)

    # 6. evaluate results
    gt_test = p.get_pickable_component(PickableComponents.LABEL_MTX)
    results = evaluate_predictions(combined_predictions, gt_test)

    # 7. save results
    with open(os.path.join(hybrid_dir, 'results_' + test_mode + '.json'), 'w') as f:
        json.dump(results, f)

    # 8. save summary of this hybrid model
    with open(os.path.join(hybrid_dir, 'indicators_' + test_mode + '.pkl'), 'wb') as f:
        pickle.dump(method_indicator_per_label, f, pickle.HIGHEST_PROTOCOL)

    counts = {name: int(sum(method_indicator_per_label == n)) for n, name in enumerate(th_methods)}
    with open(os.path.join(hybrid_dir, 'hybrid_count_info_' + test_mode + '.json'), 'w') as f:
        json.dump(counts, f)


def hybrid_model_statistical_significance(root_data_dir: str, th_methods: List, scorer: str = 'kadrnn',
                                          test_mode: str = 'soft_ensemble') -> None:

    for th_method in th_methods:
        assert th_method in th_classic_methods or th_method in th_neural_methods

    default_method_name = 'css' if 'css' in th_methods else th_methods[0]
    default_method_index = th_methods.index(default_method_name)

    # 1. get f1_per_labels per fold per method (shape: [n_methods, n_folds, n_labels]
    val_f1_per_labels_per_fold_per_th_method = []
    for th in th_methods:
        val_f1_per_labels_per_fold_per_th_method.append(get_f1_per_labels_per_folds(root_data_dir, scorer, th))

    # 2. based on val predictions determine which method to use (on per class level) -> new_method_indicator_per_class
    method_indicator_per_label, are_methods_significantly_better = determine_which_method_is_best_per_label_with_statistical_significance(
        val_f1_per_labels_per_fold_per_th_method, default_method=default_method_index)

    # 3. load test predictions from all the methods
    test_predictions = []
    for th in th_methods:
        p = TestCVPhasesComponentsPaths(root_data_dir, scorer, th, test_mode)
        test_predictions.append(p.get_pickable_component(PickableComponents.PREDICTIONS))

    # 4. combine predictions based on new_method_indicator_per_class
    combined_predictions = combine_predictions(test_predictions, method_indicator_per_label)

    # 5. save this combined predictions
    p = TestCVPhasesComponentsPaths(root_data_dir, scorer, th_methods[0], 0)
    hybrid_dir = os.path.join(p.test_scores_dir, '_'.join(th_methods))
    os.makedirs(hybrid_dir, exist_ok=True)
    with open(os.path.join(hybrid_dir, 'predictions_ttest_' + test_mode + '.pkl'), 'wb') as f:
        pickle.dump(combined_predictions, f, pickle.HIGHEST_PROTOCOL)

    # 6. evaluate results
    gt_test = p.get_pickable_component(PickableComponents.LABEL_MTX)
    results = evaluate_predictions(combined_predictions, gt_test)

    # 7. save results
    with open(os.path.join(hybrid_dir, 'results_ttest_' + test_mode + '.json'), 'w') as f:
        json.dump(results, f)

    # 8. save summary of this hybrid model
    with open(os.path.join(hybrid_dir, 'indicators_ttest_' + test_mode + '.pkl'), 'wb') as f:
        pickle.dump(method_indicator_per_label, f, pickle.HIGHEST_PROTOCOL)

    counts = {name: int(sum(method_indicator_per_label == n)) for n, name in enumerate(th_methods)}
    counts['significantly_different'] = int(sum(are_methods_significantly_better))
    with open(os.path.join(hybrid_dir, 'hybrid_count_info_ttest_' + test_mode + '.json'), 'w') as f:
        json.dump(counts, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kadrnn pipeline for hybrid models')
    parser.add_argument('data_dir', help='root data directory with valfoldX subdirectories')
    parser.add_argument('scorer', help='scorer algorithm [\'kadrnn\', \'leml\'])')
    parser.add_argument('test_mode', help='Options are: [\'single_best\', \'soft_ensemble\' \'hard_ensemble\']')
    parser.add_argument(
        '--no_stat_significance', dest='stat_significance', action='store_false',
        help='whether to do t_test per label')
    parser.add_argument(
        '--all', dest='all', action='store_true',
        help='do hybrids for all scorers and test modes')
    parser.set_defaults(scorer='kadrnn')
    parser.set_defaults(test_mode='hard_ensemble')
    parser.set_defaults(stat_significance=True)
    parser.set_defaults(all=False)
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

    if args.all:
        scorer_available = ['kadrnn'] # , 'leml']
        test_modes_available = ['soft_ensemble', 'hard_ensemble']
        if not args.stat_significance:
            test_modes_available.append('single_best')
    else:
        scorer_available = [args.scorer]
        test_modes_available = [args.test_mode]

    for scorer in scorer_available:
        for test_mode in test_modes_available:
            for m1 in th_classic_methods:
                for m2 in th_neural_methods:
                    if args.stat_significance:
                        hybrid_model_statistical_significance(args.data_dir, [m1, m2], scorer, test_mode)
                    else:
                        hybrid_model(args.data_dir, [m1, m2], scorer, test_mode)
