import os.path
import argparse
from bin.kadrnn_phases import *
import numpy as np


def train_pipeline_for_fold(data_dir: str, ifold: int, do_plots: bool, thresholding_comparison: bool,
                            scorer: str = 'kadrnn', th_version: str = 'css'):
    """
    Parameter data_dir field, in which there is dataset_train.pkl and config.json file
    :param data_dir:
    :param ifold:
    :param do_plots:
    :param thresholding_comparison:
    :param scorer:
    :param th_version:
    :return:
    """
    components_fps = TrainCVPhasesComponentsPaths(data_dir, ifold=ifold, scorer=scorer, th_version=th_version)

    if not os.path.exists(components_fps.dataset_train_fp):
        err_msg = '{} has to exist. Create it first.'.format(components_fps.dataset_train_fp)
        logging.error(err_msg)
        raise AssertionError(err_msg)

    if scorer == 'kadrnn':
        # nn-search block
        conditional_execution(components_fps, [components_fps.nns_train_fp],
                              'nearest neighbors search between train instances', nn_search_train_phase)
        conditional_execution(components_fps, [components_fps.nns_validation_fp],
                              'nearest neighbors search between validation and train instances',
                              nn_search_validation_phase)

        # scoring block
        conditional_execution(components_fps, [components_fps.scoring_config_fp],
                              'scoring optimization', optimize_scores_phase)
        conditional_execution(components_fps, [components_fps.scores_train_fp],
                              'calculating optimal scores on one-out train data', train_scores_calculation_phase)
        conditional_execution(components_fps, [components_fps.scores_validation_fp],
                              'calculating scores on validation data', validation_scores_calculation_phase)

    conditional_execution(components_fps, [components_fps.scores_train_evaluation_fp],
                          'evaluating train scores', train_scores_evaluation_phase)
    conditional_execution(components_fps, [components_fps.scores_validation_evaluation_fp],
                          'evaluating validation scores', validation_scores_evaluation_phase)
    if do_plots:
        conditional_execution(components_fps, [components_fps.scoring_optimization_plot_fp],
                              'scoring optimization plot making', plot_score_optimization_hyperparameters_impact_phase)

    thresnet = is_thresnet(th_version)

    # thresholding block
    if thresnet:
        conditional_execution(components_fps, [components_fps.thresholding_config_fp],
                              'neural thresholding methods optimization', optimize_neural_thresholding_phase)
        # train prediciton block
        conditional_execution(components_fps, [components_fps.predictions_train_fp],
                              'neural train prediction', neural_train_prediction_phase)
        conditional_execution(components_fps, [components_fps.predictions_validation_fp],
                              'neural validation prediction', neural_validation_prediction_phase)

    else:
        # conditional_execution(components_fps, [components_fps.thresholding_config_fp],
        #                      'traditional thresholding methods optimization', optimize_traditional_thresholding_phase)
        # train prediciton block
        conditional_execution(components_fps, [components_fps.thresholds_fp],
                              'calculating optimal thresholds', traditional_thresholds_calculation_phase)
        conditional_execution(components_fps, [components_fps.predictions_train_fp],
                              'train prediction', train_prediction_phase)
        conditional_execution(components_fps, [components_fps.predictions_validation_fp],
                              'validation prediction', validation_prediction_phase)

        # if do_plots:
        #     conditional_execution(components_fps, [components_fps.thresholding_optimization_plot_fp],
        #                           'thresholding optimization plot making',
        #                           plot_traditional_thresholding_optimization_phase)
        # if thresholding_comparison:
        #     conditional_execution(components_fps, [],
        #                           'train prediction for all thresholding methods', train_prediction_all_methods_phase)
        #     conditional_execution(components_fps, [],
        #                'validation prediction for all thresholding methods', validation_prediction_all_methods_phase)

    # train evaluation block
    conditional_execution(components_fps, [components_fps.results_train_fp],
                          'train evaluation', train_evaluation_phase)
    conditional_execution(components_fps, [components_fps.results_validation_fp],
                          'validation evaluation', validation_evaluation_phase)


def train_pipeline_range_of_folds(data_dir: str, scorer: str = 'kadrnn', th_version: str = 'css',
                                  do_plots: bool = False, thresholding_comparison: bool = False,
                                  up_to_fold: int = 10):

    for ifold in range(up_to_fold):
        train_pipeline_for_fold(data_dir, ifold, do_plots, thresholding_comparison, scorer, th_version)


def repair_types(components_fps: 'TrainCVPhasesComponentsPaths'):
    show_types(components_fps)
    for subset in [Subset.TRAIN, Subset.VALIDATION]:
        [nn_idxs, dists] = components_fps.get_pickable_component(PickableComponents.NNS, subset)
        nn_idxs = nn_idxs.astype(np.int32)
        dists = dists.astype(np.float32)
        components_fps.save_pickable_component(PickableComponents.NNS, subset, [nn_idxs, dists])
    show_types(components_fps)


def show_types(components_fps: 'TrainCVPhasesComponentsPaths'):

    for subset in [Subset.TRAIN, Subset.VALIDATION]:
        print('Subset: ', subset)
        dataset = components_fps.get_pickable_component(PickableComponents.DATASET, subset)
        print(type(dataset))
        print(dataset.label_mtx.dtype)  # int32
        print(dataset.feature_mtx.dtype)  # float32
        [nn_idxs, dists] = components_fps.get_pickable_component(PickableComponents.NNS, subset)
        print(nn_idxs.dtype)  # float64 !!
        print(nn_idxs[0, :5])
        print(dists.dtype)  # float64 !!
        print(dists[0, :5])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kadrnn pipeline')
    parser.add_argument('data_dir', help='data directory')
    parser.add_argument('ifold', help='which fold is validation fold: int from 0 to 9, or, negative numbers, e.g. "-4",'
                                      'which means do pipeline for folds up to (not including) a given (4) fold')
    parser.add_argument('scorer', help='scorer algorithm [\'kadrnn\', \'leml\'])')
    parser.add_argument('th_version', help='["t2", "t3"]')
    parser.add_argument(
        '--plots', dest='do_plots', action='store_true',
        help='If used, the plots showing hyperparameters impact on scores and thresholding optimization \
        will be made and saved.')
    parser.add_argument(
        '--t_comparison', dest='thresholding_comparison', action='store_true',
        help='If used, predictions are calculated for every threholding method in comparison.')
    parser.set_defaults(scorer='kadrnn')
    parser.set_defaults(th_version='css')
    parser.set_defaults(do_plots=False)
    parser.set_defaults(thresholding_comaprison=False)
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

    i_fold = int(args.ifold, 10)

    if i_fold < 0:
        train_pipeline_range_of_folds(args.data_dir, args.scorer, args.th_version, args.do_plots,
                                      args.thresholding_comparison, -i_fold)
    else:
        i_fold = int(args.ifold, 10)
        train_pipeline_for_fold(args.data_dir, i_fold, args.do_plots, args.thresholding_comparison, args.scorer,
                                args.th_version)
