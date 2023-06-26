import logging
from dataio.dataset import Dataset
import pickle
import json
import os.path
import os
import shutil
from scipy.sparse import csr_matrix

from utils.fixes import correct_absolute_paths
from utils.sparse_mtx_functions import dense_outputs_to_sparse_predictions, blend
from typing import List, Tuple
from datetime import datetime
from bin.default_configs import *
from meta.grid_search import grid_search
from bin.paths import TestCVPhasesComponentsPaths, TrainCVPhasesComponentsPaths, PickableComponents, Subset
from phases.find_knns import compute_self_k_nns_in_batches, compute_k_nns_in_batches
from phases.scoring import single_run_of_scoring_optimization, calculate_scores_powered, \
    results_mapper_dict_for_scores_optimization
from phases.thresholding import single_run_of_thresholding_optimization_based_on_train_metrics, \
    results_mapper_dict_for_thresholding_optimization, traditional_methods_dict, \
    single_run_of_thresholding_optimization_based_on_validation_metrics, scale_scores
from nn_thresholding.thresholding_models_in_keras import hopt_optimize_thresnet_phase,\
    find_paths_of_best_hyperparams_found_by_hopt,\
    predict_using_thresnet_from_hopt, get_probabilities_using_thresnet_from_hopt
from nn_thresholding.hyperparameters_config import get_hp
from evaluation.metrics import evaluate_multilabel_scoring_multimetrics, evaluate_predictions


def is_thresnet(th_version):
    return True if th_version.startswith('t') else False


def conditional_execution(components_fps, condition_file_paths: List, short_operation_description: str,
                          operation, **kwargs):
    for cfp in condition_file_paths:
        if os.path.exists(cfp):
            logging.info('{} already exists. Skipping {}...'.format(cfp, short_operation_description))
            return
    logging.info('Starting {} phase...'.format(short_operation_description))
    start_time = datetime.now()
    operation(components_fps, **kwargs)
    time_elapsed = datetime.now() - start_time
    logging.info('End of {} phase, which took {}.'.format(short_operation_description, time_elapsed))


def make_val_fold_dirs_phase(components_fps: 'TrainCVPhasesComponentsPaths') -> None:
    logging.debug('I\'m in making val fold dirs phase!')

    whole_train_dataset = Dataset.load(os.path.join(components_fps.root_data_dir, 'dataset_train.pkl'))
    if whole_train_dataset.stratification is None:
        nfolds = 10
        logging.info("Your dataset is not stratified. We first do stratification of it into {nfolds} folds."
                     .format(**locals()))
        whole_train_dataset.stratify(nfolds)
        whole_train_dataset.save(os.path.join(components_fps.root_data_dir, 'dataset_train.pkl'))
    else:
        logging.info("Your dataset is already stratified. We create subfolders and save each dataset separatelly")

    for ifold in range(len(whole_train_dataset.stratification)):
        logging.debug('\tFold {ifold} creation'.format(**locals()))
        fold_fps = TrainCVPhasesComponentsPaths(components_fps.root_data_dir, ifold)
        os.makedirs(fold_fps.fold_dir, exist_ok=True)
        train_dataset, val_dataset, _ = whole_train_dataset.data_subsets_from_stratification([ifold], [])
        train_dataset.save(fold_fps.dataset_train_fp)
        val_dataset.save(fold_fps.dataset_validation_fp)


def nn_search_train_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in nn search phase!')

    nn_config = load_config_or_create_and_save_the_default_one(components_fps.nn_config_fp, default_nn_config)
    train_dataset = Dataset.load(components_fps.dataset_train_fp)

    dists, nn_idxs = compute_self_k_nns_in_batches(train_dataset.feature_mtx, nn_config['k'],
                                                   metric=nn_config['metric'], batchsize=200, verbose=1)

    components_fps.save_pickable_component(PickableComponents.NNS, Subset.TRAIN, [nn_idxs, dists])


def nn_search_validation_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in validation nearest neighbors search phase!')

    with open(components_fps.nn_config_fp, 'r') as f:
        nn_config = json.load(f)

    train_dataset = Dataset.load(components_fps.dataset_train_fp)
    validation_dataset = Dataset.load(components_fps.dataset_validation_fp)

    dists, nn_idxs = compute_k_nns_in_batches(train_dataset.feature_mtx, validation_dataset.feature_mtx,
                                              k=nn_config['k'],
                                              metric=nn_config['metric'], batchsize=200, verbose=1)

    components_fps.save_pickable_component(PickableComponents.NNS, Subset.VALIDATION, [nn_idxs, dists])


def optimize_scores_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in scores optimization phase!')
    score_optim_config = load_config_or_create_and_save_the_default_one(components_fps.scoring_optimization_config_fp,
                                                                        default_scoring_optimization_config)
    train_dataset = Dataset.load(components_fps.dataset_train_fp)

    [nn_idxs, dists] = components_fps.get_pickable_component(PickableComponents.NNS, Subset.TRAIN)
    # with open(components_fps.nns_train_fp, 'rb') as f:
    #     [nn_idxs, dists] = pickle.load(f)

    run_dir, _, _, best_hyperparameters = grid_search(
        score_optim_config, single_run_of_scoring_optimization,
        results_mapper_dict_for_scores_optimization[score_optim_config['optimized_metric']],
        mode='max',
        output_dir=components_fps.scorer_dir,
        nn_idxs=nn_idxs, dists=dists,
        labels=train_dataset.label_mtx)

    scoring_config = {
        'optimization_dir': run_dir,
        'optimized_metric': score_optim_config['optimized_metric'],
        'best_hyperparameters': best_hyperparameters
    }

    with open(components_fps.scoring_config_fp, 'w') as f:
        json.dump(scoring_config, f)


def plot_score_optimization_hyperparameters_impact_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('Trying to plot scoring optimization hyperparameter impact')

    from meta.plot_hyperparameters_impact import save_figures_of_hyperparameters_impact,\
        save_figures_of_two_hyperparameters_impact

    with open(components_fps.scoring_config_fp, 'r') as f:
        scoring_config = json.load(f)

    run_dir = scoring_config['optimization_dir']
    save_figures_of_hyperparameters_impact(run_dir, measure_name='average_precision')

    save_figures_of_two_hyperparameters_impact(run_dir, components_fps.scoring_optimization_plot_fp,
                                               'k', 'alpha', 'average_precision')
    file_name = '{}_and_{}_impact_on_{}_grid.eps'.format('k', 'alpha', 'average_precision')
    save_figures_of_two_hyperparameters_impact(run_dir, os.path.join(run_dir, file_name),
                                               'k', 'alpha', 'average_precision')
    file_name = '{}_and_{}_impact_on_{}_grid.eps'.format('k', 'alpha', 'precision@1')
    save_figures_of_two_hyperparameters_impact(run_dir, os.path.join(run_dir, file_name),
                                               'k', 'alpha', 'precision@k', metric_index=0)
    file_name = '{}_and_{}_impact_on_{}_grid.eps'.format('k', 'alpha', 'precision@5')
    save_figures_of_two_hyperparameters_impact(run_dir, os.path.join(run_dir, file_name),
                                               'k', 'alpha', 'precision@k', metric_index=4)


def train_scores_calculation_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in train scores calculation phase!')
    train_dataset = Dataset.load(components_fps.dataset_train_fp)
    [nn_idxs, dists] = components_fps.get_pickable_component(PickableComponents.NNS, Subset.TRAIN)
    # with open(components_fps.nns_train_fp, 'rb') as f:
    #     [nn_idxs, dists] = pickle.load(f)
    with open(components_fps.scoring_config_fp, 'r') as f:
        scoring_config = json.load(f)

    k = scoring_config['best_hyperparameters']['k']
    alpha = scoring_config['best_hyperparameters']['alpha']

    train_scores = calculate_scores_powered(nn_idxs[:, :k], dists[:, :k], train_dataset.label_mtx, alpha, verbose=1)

    components_fps.save_pickable_component(PickableComponents.SCORES, Subset.TRAIN, train_scores)


def validation_scores_calculation_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in validation scores calculation phase!')
    train_dataset = Dataset.load(components_fps.dataset_train_fp)
    with open(components_fps.nns_validation_fp, 'rb') as f:
        [validation_nn_idxs, validation_dists] = pickle.load(f)
    with open(components_fps.scoring_config_fp, 'r') as f:
        scoring_config = json.load(f)

    k = scoring_config['best_hyperparameters']['k']
    alpha = scoring_config['best_hyperparameters']['alpha']

    validation_scores = calculate_scores_powered(validation_nn_idxs[:, :k], validation_dists[:, :k],
                                                 train_dataset.label_mtx, alpha, verbose=1)

    components_fps.save_pickable_component(PickableComponents.SCORES, Subset.VALIDATION, validation_scores)


def load_train_scores(components_fps: 'TrainCVPhasesComponentsPaths') -> csr_matrix:
    encoding = 'latin1' if components_fps.scorer == 'leml' else 'ASCII'
    with open(components_fps.scores_train_fp, 'rb') as f:
        train_scores = pickle.load(f, encoding=encoding)
    return train_scores


def load_validation_scores(components_fps: 'TrainCVPhasesComponentsPaths') -> csr_matrix:
    encoding = 'latin1' if components_fps.scorer == 'leml' else 'ASCII'
    with open(components_fps.scores_validation_fp, 'rb') as f:
        validation_scores = pickle.load(f, encoding=encoding)
    return validation_scores


def train_scores_evaluation_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in train scores evaluation phase!')
    scores = load_train_scores(components_fps)
    train_dataset = Dataset.load(components_fps.dataset_train_fp)
    results = evaluate_multilabel_scoring_multimetrics(train_dataset.label_mtx, scores, 5)
    with open(components_fps.scores_train_evaluation_fp, 'w') as f:
        json.dump(results, f)


def validation_scores_evaluation_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in validation scores evaluation phase!')
    scores = load_validation_scores(components_fps)
    validation_dataset = Dataset.load(components_fps.dataset_validation_fp)
    results = evaluate_multilabel_scoring_multimetrics(validation_dataset.label_mtx, scores, 5)
    with open(components_fps.scores_validation_evaluation_fp, 'w') as f:
        json.dump(results, f)


def optimize_traditional_thresholding_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in optimization of traditional thresholding methods phase!')
    thresholding_optim_config = load_config_or_create_and_save_the_default_one(
        components_fps.thresholding_optimization_config_fp, default_thresholding_optimization_config)

    train_dataset = Dataset.load(components_fps.dataset_train_fp)
    scores_train = load_train_scores(components_fps)

    if thresholding_optim_config['selection_method'] == 'validation_based':
        single_run = single_run_of_thresholding_optimization_based_on_validation_metrics
    else:
        single_run = single_run_of_thresholding_optimization_based_on_train_metrics

    run_dir, _, _, best_hyperparameters = grid_search(
        thresholding_optim_config, single_run,
        results_mapper_dict_for_thresholding_optimization[thresholding_optim_config['optimized_metric']],
        mode='max',
        output_dir=components_fps.fold_thresholding_dir,
        scores_train=scores_train,
        labels=train_dataset.label_mtx,
        verbose=1)

    thresholding_config = {
        'optimization_dir': run_dir,
        'optimized_metric': thresholding_optim_config['optimized_metric'],
        'best_hyperparameters': best_hyperparameters,
        'selection_method': thresholding_optim_config['selection_method']
    }

    with open(components_fps.thresholding_config_fp, 'w') as f:
        json.dump(thresholding_config, f)


def optimize_neural_thresholding_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    """

    :param components_fps:
    :return:
    """

    logging.debug('I\'m in optimization of thresnet thresholding methods phase!')

    css_thresholds_path = os.path.join(components_fps.fold_scores_dir,
                                       'css', 'thresholds.pkl') if components_fps.th_version.startswith('t3') else None
    params = get_hp(components_fps.th_version, css_thresholds_path=css_thresholds_path)

    # Check if training was already performed
    run_dir = components_fps.fold_thresholding_dir
    if os.path.exists(os.path.join(run_dir, 'models')):
        logging.info("Training output dir {} already exists. Skipping.".format(run_dir))
    else:
        train_dataset = Dataset.load(components_fps.dataset_train_fp)
        scores_train = load_train_scores(components_fps)

        params.n_outputs = train_dataset.label_mtx.shape[1]

        validation_dataset = Dataset.load(components_fps.dataset_validation_fp)
        scores_validation = load_validation_scores(components_fps)

        hopt_optimize_thresnet_phase(run_dir, params, scores_train, train_dataset.label_mtx,
                                     scores_validation=scores_validation,
                                     labels_validation=validation_dataset.label_mtx)

    mode = 'max' if params.with_f_metrics else 'min'
    best_hyper_params_config_path, best_model_weights_path = find_paths_of_best_hyperparams_found_by_hopt(
        run_dir, mode=mode)

    thresholding_config = {
        'optimization_dir': run_dir,
        'best_hyperparameters_path': best_hyper_params_config_path,
        'best_model_weights_path': best_model_weights_path,
    }

    with open(components_fps.thresholding_config_fp, 'w') as f:
        json.dump(thresholding_config, f)


def neural_train_prediction_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in neural train prediction phase!')
    with open(components_fps.thresholding_config_fp, 'r') as f:
        thresholding_config = json.load(f)
    scores_train = load_train_scores(components_fps)
    predictions = predict_using_thresnet_from_hopt(
        scores=scores_train,
        path_to_model_config=thresholding_config['best_hyperparameters_path'],
        path_to_weights=thresholding_config['best_model_weights_path'])
    components_fps.save_pickable_component(PickableComponents.PREDICTIONS, Subset.TRAIN, predictions)


def neural_validation_prediction_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in neural validation prediction phase!')
    with open(components_fps.thresholding_config_fp, 'r') as f:
        thresholding_config = json.load(f)
    scores_validation = load_validation_scores(components_fps)
    predictions = predict_using_thresnet_from_hopt(
        scores=scores_validation,
        path_to_model_config=thresholding_config['best_hyperparameters_path'],
        path_to_weights=thresholding_config['best_model_weights_path'])
    components_fps.save_pickable_component(PickableComponents.PREDICTIONS, Subset.VALIDATION, predictions)


def plot_traditional_thresholding_optimization_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in plot of traditional thresholding methods optimization phase!')
    logging.warning('Not implemented yet')    # TODO


def thresholds_copying_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    """
    When thresholding optimization caches optimal thresholds for each of the methods, here it is sufficient to just
    copy picked thresholds for best method into main directory under thresholds.pkl name.
    :param components_fps:
    :return:
    """
    logging.debug('I\'m in thresholds copying phase!')
    with open(components_fps.thresholding_config_fp, 'r') as f:
        thresholding_config = json.load(f)
    src_file = os.path.join(thresholding_config['optimization_dir'],
                            thresholding_config['best_hyperparameters']['thr_method'] + '.pkl')
    root_dir = os.path.dirname(os.path.dirname(__file__))
    shutil.copy(os.path.join(root_dir, src_file), os.path.join(root_dir, components_fps.thresholds_fp))


def traditional_thresholds_calculation_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    """
    :param components_fps:
    :return:
    """
    train_dataset = Dataset.load(components_fps.dataset_train_fp)
    scores_train = load_train_scores(components_fps)
    (fit_method, predict_method) = traditional_methods_dict[components_fps.th_version]

    # thresholding training
    thresholds, _ = fit_method(scores_train, train_dataset.label_mtx, verbose=1, n_processes=6)

    # save thresholds
    os.makedirs(os.path.dirname(components_fps.thresholds_fp), exist_ok=True)
    with open(components_fps.thresholds_fp, 'wb') as f:
        pickle.dump(thresholds, f, pickle.HIGHEST_PROTOCOL)


def train_prediction_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in train prediction phase!')
    # with open(components_fps.thresholding_config_fp, 'r') as f:
    #     thresholding_config = json.load(f)
    # (_, predict_method) = traditional_methods_dict[thresholding_config['best_hyperparameters']['thr_method']]
    (fit_method, predict_method) = traditional_methods_dict[components_fps.th_version]
    scores_train = load_train_scores(components_fps)
    with open(components_fps.thresholds_fp, 'rb') as f:
        thresholds = pickle.load(f)
    predictions = predict_method(scores_train, thresholds)
    components_fps.save_pickable_component(PickableComponents.PREDICTIONS, Subset.TRAIN, predictions)


def validation_prediction_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in validation prediction phase!')
    # with open(components_fps.thresholding_config_fp, 'r') as f:
    #     thresholding_config = json.load(f)
    # (_, predict_method) = traditional_methods_dict[thresholding_config['best_hyperparameters']['thr_method']]
    (fit_method, predict_method) = traditional_methods_dict[components_fps.th_version]
    scores_validation = load_validation_scores(components_fps)
    with open(components_fps.thresholds_fp, 'rb') as f:
        thresholds = pickle.load(f)
    predictions = predict_method(scores_validation, thresholds)
    components_fps.save_pickable_component(PickableComponents.PREDICTIONS, Subset.VALIDATION, predictions)


def train_prediction_all_methods_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in train prediction all methods phase!')
    scores_train = load_train_scores(components_fps)
    thresholding_config = load_config_or_create_and_save_the_default_one(
        components_fps.thresholding_config_fp)
    thr_optim_dir = thresholding_config['optimization_dir']
    root_dir = os.path.dirname(os.path.dirname(__file__))

    for method_name, v in traditional_methods_dict.items():
        (_, predict_method) = v
        with open(os.path.join(root_dir, thr_optim_dir, method_name+'.pkl'), 'rb') as f:
            thresholds = pickle.load(f)
        predictions = predict_method(scores_train, thresholds)
        with open(os.path.join(root_dir, thr_optim_dir, 'predictions_' + method_name + '_train.pkl'), 'wb') as f:
            pickle.dump(predictions, f, pickle.HIGHEST_PROTOCOL)


def validation_prediction_all_methods_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in validation prediction all methods phase!')
    scores_validation = load_validation_scores(components_fps)
    thresholding_config = load_config_or_create_and_save_the_default_one(
        components_fps.thresholding_config_fp)
    thr_optim_dir = thresholding_config['optimization_dir']
    root_dir = os.path.dirname(os.path.dirname(__file__))

    for method_name, v in traditional_methods_dict.items():
        (_, predict_method) = v
        with open(os.path.join(root_dir, thr_optim_dir, method_name+'.pkl'), 'rb') as f:
            thresholds = pickle.load(f)
        predictions = predict_method(scores_validation, thresholds)
        with open(os.path.join(root_dir, thr_optim_dir, 'predictions_' + method_name + '_validation.pkl'), 'wb') as f:
            pickle.dump(predictions, f, pickle.HIGHEST_PROTOCOL)


def train_evaluation_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in train evaluation phase!')
    with open(components_fps.predictions_train_fp, 'rb') as f:
        predictions = pickle.load(f)
    train_dataset = Dataset.load(components_fps.dataset_train_fp)
    results = evaluate_predictions(predictions, train_dataset.label_mtx)
    with open(components_fps.results_train_fp, 'w') as f:
        json.dump(results, f)


def validation_evaluation_phase(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in validation evaluation phase!')
    with open(components_fps.predictions_validation_fp, 'rb') as f:
        predictions = pickle.load(f)
    validation_dataset = Dataset.load(components_fps.dataset_validation_fp)
    results = evaluate_predictions(predictions, validation_dataset.label_mtx)
    with open(components_fps.results_validation_fp, 'w') as f:
        json.dump(results, f)


# --------------------------------- TEST PHASE -----------------------------

def nn_search_test_phase(components_fps: 'TestCVPhasesComponentsPaths'):
    logging.debug('I\'m in test nearest neighbors search phase!')

    with open(components_fps.nn_config_fp, 'r') as f:
        nn_config = json.load(f)

    train_dataset = Dataset.load(components_fps.dataset_train_fp)
    test_dataset = Dataset.load(components_fps.dataset_test_fp)

    dists, nn_idxs = compute_k_nns_in_batches(train_dataset.feature_mtx, test_dataset.feature_mtx,
                                              k=nn_config['k'],
                                              metric=nn_config['metric'], batchsize=200, verbose=1)

    components_fps.save_pickable_component(PickableComponents.NNS, [nn_idxs, dists])


def test_scores_calculation_phase(components_fps: 'TestCVPhasesComponentsPaths'):
    logging.debug('I\'m in test scores calculation phase!')
    train_dataset = Dataset.load(components_fps.dataset_train_fp)
    [test_nn_idxs, test_nn_dists] = components_fps.get_pickable_component(PickableComponents.NNS)
    with open(components_fps.scoring_config_fp, 'r') as f:
        scoring_config = json.load(f)
    k = scoring_config['best_hyperparameters']['k']
    alpha = scoring_config['best_hyperparameters']['alpha']

    test_scores = calculate_scores_powered(test_nn_idxs[:, :k], test_nn_dists[:, :k], train_dataset.label_mtx,
                                           alpha, verbose=1)

    components_fps.save_pickable_component(PickableComponents.SCORES, test_scores)


def load_test_scores(components_fps: 'TestCVPhasesComponentsPaths') -> csr_matrix:
    encoding = 'latin1' if components_fps.scorer == 'leml' else 'ASCII'
    with open(components_fps.scores_test_fp, 'rb') as f:
        test_scores = pickle.load(f, encoding=encoding)
    return test_scores


def test_scores_evaluation_phase(components_fps: 'TestCVPhasesComponentsPaths'):
    logging.debug('I\'m in test scores evaluation phase!')
    scores = load_test_scores(components_fps)
    test_dataset = Dataset.load(components_fps.dataset_test_fp)
    results = evaluate_multilabel_scoring_multimetrics(test_dataset.label_mtx, scores, 5)
    with open(components_fps.scores_test_evaluation_fp, 'w') as f:
        json.dump(results, f)


def test_prediction(components_fps: 'TestCVPhasesComponentsPaths') -> csr_matrix:
    # with open(components_fps.thresholding_config_fp, 'r') as f:
    #     thresholding_config = json.load(f)
    # (_, predict_method) = traditional_methods_dict[thresholding_config['best_hyperparameters']['thr_method']]
    (_, predict_method) = traditional_methods_dict[components_fps.th_version]
    scores_test = load_test_scores(components_fps)
    with open(components_fps.thresholds_fp, 'rb') as f:
        thresholds = pickle.load(f)
    predictions = predict_method(scores_test, thresholds)
    return predictions


def test_prediction_phase(components_fps: 'TestCVPhasesComponentsPaths'):
    logging.debug('I\'m in test prediction phase!')
    predictions = test_prediction(components_fps)
    os.makedirs(os.path.dirname(components_fps.predictions_test_fp), exist_ok=True)
    with open(components_fps.predictions_test_fp, 'wb') as f:
        pickle.dump(predictions, f, pickle.HIGHEST_PROTOCOL)


def neural_test_prediction(components_fps: 'TestCVPhasesComponentsPaths') -> csr_matrix:
    with open(components_fps.thresholding_config_fp, 'r') as f:
        thresholding_config = json.load(f)
    scores_test = load_test_scores(components_fps)
    best_hp_path = correct_absolute_paths(thresholding_config['best_hyperparameters_path'])
    best_model_weights = correct_absolute_paths(thresholding_config['best_model_weights_path'])
    predictions = predict_using_thresnet_from_hopt(
        scores=scores_test,
        path_to_model_config=best_hp_path,
        path_to_weights=best_model_weights)
    return predictions


def neural_test_probabilities(components_fps: 'TestCVPhasesComponentsPaths') -> csr_matrix:
    with open(components_fps.thresholding_config_fp, 'r') as f:
        thresholding_config = json.load(f)
    scores_test = load_test_scores(components_fps)
    best_hp_path = correct_absolute_paths(thresholding_config['best_hyperparameters_path'])
    best_model_weights = correct_absolute_paths(thresholding_config['best_model_weights_path'])
    probabilities = get_probabilities_using_thresnet_from_hopt(
        scores=scores_test,
        path_to_model_config=best_hp_path,
        path_to_weights=best_model_weights)
    return probabilities


def neural_test_prediction_phase(components_fps: 'TestCVPhasesComponentsPaths'):
    logging.debug('I\'m in neural test prediction phase!')
    predictions = neural_test_prediction(components_fps)
    os.makedirs(os.path.dirname(components_fps.predictions_test_fp), exist_ok=True)
    with open(components_fps.predictions_test_fp, 'wb') as f:
        pickle.dump(predictions, f, pickle.HIGHEST_PROTOCOL)


def test_evaluation_phase(components_fps: 'TestCVPhasesComponentsPaths'):
    logging.debug('I\'m in test evaluation phase!')
    with open(components_fps.predictions_test_fp, 'rb') as f:
        predictions = pickle.load(f)
    dataset = Dataset.load(components_fps.dataset_test_fp)
    results = evaluate_predictions(predictions, dataset.label_mtx)
    with open(components_fps.results_test_fp, 'w') as f:
        json.dump(results, f)


# def test_prediction_all_methods_phase(components_fps: 'TestCVPhasesComponentsPaths'):
#     logging.debug('I\'m in test prediction all methods phase!')
#
#     scores_test = load_test_scores(components_fps)
#
#     thresholding_config = load_config_or_create_and_save_the_default_one(
#         components_fps.thresholding_config_fp)
#     thr_optim_dir = thresholding_config['optimization_dir']
#     root_dir = os.path.dirname(os.path.dirname(__file__))
#
#     for method_name, v in traditional_methods_dict.items():
#         (_, predict_method) = v
#
#         with open(os.path.join(root_dir, thr_optim_dir, method_name+'.pkl'), 'rb') as f:
#             thresholds = pickle.load(f)
#         predictions = predict_method(scores_test, thresholds)
#
#         with open(os.path.join(root_dir, thr_optim_dir, 'predictions_' + method_name + '_test.pkl'), 'wb') as f:
#             pickle.dump(predictions, f, pickle.HIGHEST_PROTOCOL)


# TODO make this glob based, not parameter based
def validation_folds_results_summary(components_fps: 'TrainCVPhasesComponentsPaths', up_to_fold: 'int' = 10,
                                     methods: Tuple[str] = ('t2max_v1', 't2none_v1', 'css')):
    logging.debug('I\'m in validation results from all folds gathering phase!')
    metrics = ['F_mi', 'F_ma']
    all_results = {k: {k2: [] for k2 in methods} for k in metrics}
    for val_fold in range(up_to_fold):
        val_dir = os.path.join(components_fps.root_data_dir, 'val_fold{}'.format(val_fold))
        for method in methods:
            results_file_path = os.path.join(val_dir, components_fps.scorer, method, 'results_validation.json')
            with open(results_file_path, 'r') as f:
                vm_results = json.load(f)
            for metric in metrics:
                all_results[metric][method].append(float(vm_results[metric]))
    os.makedirs(os.path.dirname(components_fps.all_validation_results_fp), exist_ok=True)
    with open(components_fps.all_validation_results_fp, 'w') as f:
        json.dump(all_results, f)


def validation_folds_boxplots(components_fps: 'TrainCVPhasesComponentsPaths'):
    logging.debug('I\'m in boxplots plotting of validation results!')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy as np

    rcParams.update({'figure.autolayout': True})

    with open(components_fps.all_validation_results_fp, 'r') as f:
        all_results = json.load(f)

    method_names_mapping = {
        'css': 'CSS',
        't21none_v1': 'ThresNet',
        't3none': 'ThresNet v2',
        't2': 'ThresNet',
        't3': 'ThresNet v2'
    }

    metrics_mapping = {
        'F_mi': '$F^{mi}$',
        'F_ma': '$F^{ma}$',
        'F_avg': '$F^{avg}$',
    }

    names = None
    v = []
    for metric in ['F_mi', 'F_ma']:
        names, values = zip(*[(method_names_mapping[orig_name], np.array(l)) for
                              orig_name, l in all_results[metric].items() if orig_name in method_names_mapping.keys()])
        values = np.vstack(values[::-1]).transpose()
        v.append(values)

    f_avg = np.average(np.stack(v), axis=0)
    v.append(f_avg)

    for n, metric in enumerate(['F_mi', 'F_ma', 'F_avg']):
        plt.figure()
        plt.boxplot(v[n])
        # plt.ylabel(metric)
        plt.ylabel(metrics_mapping[metric])
        plt.xticks(list(range(1, len(names)+1)), names[::-1], rotation=70)
        plt.savefig(os.path.join(components_fps.val_summary_dir, 'boxplot_'+metric+'.png'))


def find_used_number_of_folds(components_fps: 'TestCVPhasesComponentsPaths') -> int:
    with open(components_fps.all_validation_results_fp, 'r') as f:
        validation_results_summary = json.load(f)
    n_folds = len(validation_results_summary["F_mi"]["css"])
    return n_folds


def hard_voting_phase(components_fps: 'TestCVPhasesComponentsPaths') -> None:
    logging.debug('I\'m in hard voting predictions phase!')

    thresnet = is_thresnet(components_fps.th_version)
    n_folds = find_used_number_of_folds(components_fps)

    components_fps.set_thresholding_dirs(0)
    predictions = neural_test_prediction(components_fps) if thresnet else test_prediction(components_fps)
    total_predictions = predictions.astype(int)
    for i_fold in range(1, n_folds):
        components_fps.set_thresholding_dirs(i_fold)
        predictions = neural_test_prediction(components_fps) if thresnet else test_prediction(components_fps)
        total_predictions += predictions.astype(int)

    avg_predictions = total_predictions / n_folds
    hard_voting_predictions = avg_predictions >= 0.5

    components_fps.save_pickable_component(PickableComponents.PREDICTIONS, hard_voting_predictions)


def soft_voting_phase(components_fps: 'TestCVPhasesComponentsPaths') -> None:
    logging.debug('I\'m in soft voting predictions phase!')

    thresnet = is_thresnet(components_fps.th_version)
    n_folds = find_used_number_of_folds(components_fps)

    if not thresnet:
        components_fps.set_thresholding_dirs(0)
        with open(components_fps.thresholds_fp, 'rb') as f:
            aggr_thresholds = pickle.load(f)

        for i_fold in range(1, n_folds):
            components_fps.set_thresholding_dirs(i_fold)
            with open(components_fps.thresholds_fp, 'rb') as f:
                aggr_thresholds += pickle.load(f)

        avg_thresholds = aggr_thresholds / n_folds

        (_, predict_method) = traditional_methods_dict[components_fps.th_version]
        scores_test = load_test_scores(components_fps)
        soft_voting_predictions = predict_method(scores_test, avg_thresholds)

    else:
        components_fps.set_thresholding_dirs(0)
        total_probabilities = neural_test_probabilities(components_fps)
        for i_fold in range(1, n_folds):
            components_fps.set_thresholding_dirs(i_fold)
            total_probabilities += neural_test_probabilities(components_fps)

        avg_probabilities = total_probabilities / n_folds
        soft_voting_predictions = dense_outputs_to_sparse_predictions(avg_probabilities,
                                                                      ensure_at_least_one_predicted=True)

    components_fps.save_pickable_component(PickableComponents.PREDICTIONS, soft_voting_predictions)
