import argparse
from bin.kadrnn_phases import *
from bin.paths import TestCVPhasesComponentsPaths


def testing_pipeline_from_best_valfold(
        root_data_dir: str, scorer: str = 'kadrnn', th_version: str = 'css') -> None:
    """
    :param root_data_dir:
    :param scorer: ['kadrnn', 'leml']
    :param th_version: ['css', 't2max', 't2none']
    :return:
    """

    components_fps = TestCVPhasesComponentsPaths(
        root_data_dir, scorer, th_version, which_cv_model='single_best')

    # thresholding block
    thresnet = is_thresnet(th_version)
    if thresnet:
        conditional_execution(components_fps, [components_fps.predictions_test_fp],
                              'neural test prediction', neural_test_prediction_phase)
    else:
        conditional_execution(components_fps, [components_fps.predictions_test_fp],
                              'test prediction', test_prediction_phase)

    # evaluation block
    conditional_execution(components_fps, [components_fps.results_test_fp],
                          'test evaluation', test_evaluation_phase)


def testing_pipeline_hard_voting_ensemble(root_data_dir: str, scorer: str = 'kadrnn', th_version: str = 'css') -> None:
    """

    :param root_data_dir:
    :param scorer:
    :param th_version:
    :return:
    """

    components_fps = TestCVPhasesComponentsPaths(
        root_data_dir, scorer, th_version, which_cv_model='hard_ensemble')

    # hard voting block
    conditional_execution(components_fps, [components_fps.predictions_test_fp],
                          'hard voting test prediction', hard_voting_phase)

    # evaluation block
    conditional_execution(components_fps, [components_fps.results_test_fp],
                          'test evaluation', test_evaluation_phase)


def testing_pipeline_soft_voting_ensemble(root_data_dir: str, scorer: str = 'kadrnn', th_version: str = 'css') -> None:
    components_fps = TestCVPhasesComponentsPaths(
        root_data_dir, scorer, th_version, which_cv_model='soft_ensemble')

    # hard voting block
    conditional_execution(components_fps, [components_fps.predictions_test_fp],
                          'soft voting test prediction', soft_voting_phase)

    # evaluation block
    conditional_execution(components_fps, [components_fps.results_test_fp],
                          'test evaluation', test_evaluation_phase)


def test_pipeline(root_data_dir: str, scorer: str = 'kadrnn', th_version: str = 'css', test_mode: str = 'single_best'):

    components_fps = TestCVPhasesComponentsPaths(root_data_dir, scorer, th_version)

    if scorer == 'kadrnn':
        # nn-search block
        conditional_execution(components_fps, [components_fps.nns_test_fp],
                              'nearest neighbors search between test and train instances', nn_search_test_phase)

        # scoring block
        conditional_execution(components_fps, [components_fps.scores_test_fp],
                              'calculating scores on test data', test_scores_calculation_phase)

    conditional_execution(components_fps, [components_fps.scores_test_evaluation_fp],
                          'evaluating train scores', test_scores_evaluation_phase)

    if test_mode == 'single_best':
        testing_pipeline_from_best_valfold(root_data_dir, scorer, th_version)
    elif test_mode == 'soft_ensemble':
        testing_pipeline_soft_voting_ensemble(root_data_dir, scorer, th_version)
    elif test_mode == 'hard_ensemble':
        testing_pipeline_hard_voting_ensemble(root_data_dir, scorer, th_version)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kadrnn pipeline in test phase')
    parser.add_argument('data_dir', help='root data directory with valfoldX subdirectories')
    parser.add_argument('scorer', help='scorer algorithm [\'kadrnn\', \'leml\'])')
    parser.add_argument('th_version', help='["css", "t2none", "t2max", "t2", "t3"]')
    parser.add_argument('test_mode', help='Options are: [\'single_best\', \'soft_ensemble\' \'hard_ensemble\']')
    parser.set_defaults(scorer='kadrnn')
    parser.set_defaults(th_version='css')
    parser.set_defaults(test_mode='hard_ensemble')
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

    # if args.test_mode == 'hard_ensemble':
    #     testing_pipeline_hard_voting_ensemble(args.data_dir, scorer=args.scorer, th_version=args.th_version)
    # elif args.test_mode == 'soft_ensemble':
    #     testing_pipeline_soft_voting_ensemble(args.data_dir, scorer=args.scorer, th_version=args.th_version)
    # elif args.test_mode == 'single_best':
    #     testing_pipeline_from_best_valfold(args.data_dir, scorer=args.scorer, th_version=args.th_version)

    # test_pipeline(args.data_dir, args.scorer, args.th_version, args.test_mode)

    # # for scorer in ['kadrnn', 'leml']:
    for scorer in ['kadrnn', 'leml']:
        for th_version in ['t2', 't3', 'css']:
            testing_pipeline_soft_voting_ensemble(args.data_dir, scorer=scorer, th_version=th_version)
            testing_pipeline_hard_voting_ensemble(args.data_dir, scorer=scorer, th_version=th_version)
            testing_pipeline_from_best_valfold(args.data_dir, scorer=scorer, th_version=th_version)
