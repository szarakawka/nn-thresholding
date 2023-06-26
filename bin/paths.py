import os.path
from enum import Enum, auto
import pickle
from dataio.dataset import Dataset
import json
import numpy as np


class Subset(Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'


class PickableComponents(Enum):
    DATASET = auto()
    FEATURE_MTX = auto()
    LABEL_MTX = auto()
    NNS = auto()
    SCORES = auto()
    PREDICTIONS = auto()
    THRESHOLDS = auto()


class TrainCVPhasesComponentsPaths:

    def __init__(self, root_data_dir: str, ifold: int, scorer: str = 'kadrnn', th_version: str = 'css'):
        self.root_data_dir = root_data_dir
        self.scorer = scorer

        self.fold_dir = os.path.join(root_data_dir, 'val_fold{}'.format(ifold))
        self.dataset_train_fp = os.path.join(self.fold_dir, 'dataset_train.pkl')
        self.dataset_validation_fp = os.path.join(self.fold_dir, 'dataset_validation.pkl')
        self.fold_scores_dir = os.path.join(self.fold_dir, scorer)
        self.scorer_dir = os.path.join(root_data_dir, 'scorers', scorer) if scorer == 'kadrnn' else self.fold_scores_dir

        # nn-search
        self.nn_config_fp = os.path.join(self.scorer_dir, 'config_nn_search.json')
        self.nns_train_fp = os.path.join(self.fold_scores_dir, 'nns_train.pkl')
        self.nns_validation_fp = os.path.join(self.fold_scores_dir, 'nns_validation.pkl')

        # scoring
        self.scoring_optimization_config_fp = os.path.join(self.scorer_dir, 'scoring_optimization_config.json')
        self.scoring_optimization_plot_fp = os.path.join(self.scorer_dir, 'scoring_optimization.eps')
        self.scoring_config_fp = os.path.join(self.scorer_dir, 'scoring_config.json')
        self.scores_train_fp = os.path.join(self.fold_scores_dir, 'scores_train.pkl')
        self.scores_train_evaluation_fp = os.path.join(self.fold_scores_dir, 'scores_train_evaluation.json')
        self.scores_validation_fp = os.path.join(self.fold_scores_dir, 'scores_validation.pkl')
        self.scores_validation_evaluation_fp = os.path.join(self.fold_scores_dir, 'scores_validation_evaluation.json')

        # thresholding
        self.th_version = th_version
        self.fold_thresholding_dir = os.path.join(self.fold_scores_dir, th_version)
        self.thresholding_optimization_config_fp = os.path.join(self.fold_thresholding_dir,
                                                                'thresholding_optimization_config.json')
        self.thresholding_optimization_plot_fp = os.path.join(self.fold_thresholding_dir,
                                                              'thresholding_optimization.eps')
        self.thresholding_config_fp = os.path.join(self.fold_thresholding_dir, 'thresholding_config.json')
        self.thresholds_fp = os.path.join(self.fold_thresholding_dir, 'thresholds.pkl')

        # prediction
        self.predictions_train_fp = os.path.join(self.fold_thresholding_dir, 'predictions_train.pkl')
        self.results_train_fp = os.path.join(self.fold_thresholding_dir, 'results_train.json')
        self.predictions_validation_fp = os.path.join(self.fold_thresholding_dir, 'predictions_validation.pkl')
        self.results_validation_fp = os.path.join(self.fold_thresholding_dir, 'results_validation.json')

        # validation results summary
        self.val_summary_dir = os.path.join(self.root_data_dir, 'val_summary', self.scorer)
        self.boxplot_fmi_fma_fp = os.path.join(self.val_summary_dir, 'boxplot_valfolds_fmi_fma.png')
        self.boxplot_favg_fp = os.path.join(self.val_summary_dir, 'boxplot_valfolds_favg.png')
        self.boxplots_fp = os.path.join(self.val_summary_dir, 'boxplots.png')
        self.all_validation_results_fp = os.path.join(self.val_summary_dir, 'all_validation_results.json')

        self.mapping_direct_train_pickles = {
            PickableComponents.DATASET: self.dataset_train_fp,
            PickableComponents.NNS: self.nns_train_fp,
            PickableComponents.SCORES: self.scores_train_fp,
            PickableComponents.PREDICTIONS: self.predictions_train_fp,
            PickableComponents.THRESHOLDS: self.thresholds_fp,
        }
        self.mapping_direct_val_pickles = {
            PickableComponents.DATASET: self.dataset_validation_fp,
            PickableComponents.NNS: self.nns_validation_fp,
            PickableComponents.SCORES: self.scores_validation_fp,
            PickableComponents.PREDICTIONS: self.predictions_validation_fp,
            PickableComponents.THRESHOLDS: self.thresholds_fp,
        }

    def get_pickable_component(self, which_component: 'PickableComponents', which_set: 'Subset'):
        if which_component in self.mapping_direct_train_pickles.keys():
            if which_set is Subset.TRAIN:
                with open(self.mapping_direct_train_pickles[which_component], 'rb') as f:
                    return pickle.load(f)
            elif which_set is Subset.VALIDATION:
                with open(self.mapping_direct_val_pickles[which_component], 'rb') as f:
                    return pickle.load(f)
            raise ValueError('which_set parameter has bad value.')
        elif which_component in [PickableComponents.FEATURE_MTX, PickableComponents.LABEL_MTX]:
            if which_set == Subset.TRAIN:
                d = Dataset.load(self.dataset_train_fp)
            elif which_set == Subset.VALIDATION:
                d = Dataset.load(self.dataset_validation_fp)
            else:
                raise ValueError('which_set parameter has bad value.')
            return d.feature_mtx if which_component == PickableComponents.FEATURE_MTX else d.label_mtx
        else:
            raise ValueError('which_component parameter has bad value.')

    def save_pickable_component(self, which_component: 'PickableComponents', which_set: 'Subset', pickable_obj):
        if which_component in self.mapping_direct_train_pickles.keys():
            if which_set is Subset.TRAIN:
                fp = self.mapping_direct_train_pickles[which_component]
            elif which_set is Subset.VALIDATION:
                fp = self.mapping_direct_val_pickles[which_component]
            else:
                raise ValueError('which_set parameter has bad value.')

            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, 'wb') as f:
                pickle.dump(pickable_obj, f, pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError('which_component parameter has bad value.')


class TestCVPhasesComponentsPaths:

    def __init__(self, root_data_dir: str, scorer: str = 'kadrnn', th_version: str = 'css',
                 which_cv_model=None):
        """

        :param root_data_dir:
        :param scorer:
        :param th_version:
        :param which_cv_model: [fold_num (e.g. 0, 1, 2, ...), 'single_best', 'soft_ensemble', 'hard_ensemble']
        """

        self.root_data_dir = root_data_dir
        self.scorer = scorer
        self.th_version = th_version

        self.test_data_dir = os.path.join(root_data_dir, 'test_data')
        self.scorer = scorer
        self.scorer_dir = os.path.join(root_data_dir, 'scorers', scorer)

        self.dataset_train_fp = os.path.join(self.root_data_dir, 'dataset_train.pkl')
        self.dataset_test_fp = os.path.join(self.test_data_dir, 'dataset_test.pkl')
        self.test_scores_dir = os.path.join(self.test_data_dir, scorer)

        # nn-search
        self.nn_config_fp = os.path.join(self.scorer_dir, 'config_nn_search.json')
        self.nns_test_fp = os.path.join(self.test_scores_dir, 'nns_test.pkl')

        # scoring
        self.scoring_optimization_config_fp = os.path.join(self.scorer_dir, 'scoring_optimization_config.json')
        self.scoring_optimization_plot_fp = os.path.join(self.scorer_dir, 'scoring_optimization.eps')
        self.scoring_config_fp = os.path.join(self.scorer_dir, 'scoring_config.json')
        self.scores_test_fp = os.path.join(self.test_scores_dir, 'scores_test.pkl')
        self.scores_test_evaluation_fp = os.path.join(self.test_scores_dir, 'scores_test_evaluation.json')

        # val summary
        self.val_summary_dir = os.path.join(root_data_dir, 'val_summary', self.scorer)
        self.all_validation_results_fp = os.path.join(self.val_summary_dir, 'all_validation_results.json')

        # thresholding
        self.th_version = th_version
        self.thresholding_out_dir = None
        self.thresholder_dir = None
        self.thresholding_config_fp = None
        self.thresholds_fp = None
        self.predictions_test_fp = None
        self.results_test_fp = None

        self.thresholding_out_dir = os.path.join(self.test_scores_dir, self.th_version)

        if which_cv_model is not None:
            if isinstance(which_cv_model, int):
                i_fold = which_cv_model
                self.set_thresholding_dirs(ifold=i_fold)
                self.predictions_test_fp = os.path.join(self.thresholding_out_dir,
                                                        'predictions_val_fold_{}_model.pkl'.format(i_fold))
                self.results_test_fp = os.path.join(self.thresholding_out_dir,
                                                    'results_val_fold_{}_model.json'.format(i_fold))
            elif which_cv_model == 'single_best':
                best_i_fold = self.find_best_fold_according_to_metric('F_ma')
                self.set_thresholding_dirs(ifold=best_i_fold)
                self.predictions_test_fp = os.path.join(self.thresholding_out_dir, 'predictions_single_best_model.pkl')
                self.results_test_fp = os.path.join(self.thresholding_out_dir, 'results_single_best_model.json')
            elif which_cv_model == 'soft_ensemble':
                self.predictions_test_fp = os.path.join(self.thresholding_out_dir, 'predictions_soft_ensemble.pkl')
                self.results_test_fp = os.path.join(self.thresholding_out_dir, 'results_soft_ensemble.json')

            elif which_cv_model == 'hard_ensemble':
                self.predictions_test_fp = os.path.join(self.thresholding_out_dir, 'predictions_hard_ensemble.pkl')
                self.results_test_fp = os.path.join(self.thresholding_out_dir, 'results_hard_ensemble.json')
            else:
                raise AssertionError('Bad which_cv_model argument.')

        self.mapping_direct_test_pickles = {
            PickableComponents.DATASET: self.dataset_test_fp,
            PickableComponents.NNS: self.nns_test_fp,
            PickableComponents.SCORES: self.scores_test_fp,
            PickableComponents.PREDICTIONS: self.predictions_test_fp,
        }

    def set_thresholding_dirs(self, ifold):
        self.thresholder_dir = os.path.join(self.root_data_dir,
                                            'val_fold{}'.format(ifold),
                                            self.scorer,
                                            self.th_version)
        self.thresholding_config_fp = os.path.join(self.thresholder_dir,
                                                   'thresholding_config.json')
        self.thresholds_fp = os.path.join(self.thresholder_dir, 'thresholds.pkl')

    def get_pickable_component(self, which_component: 'PickableComponents', subset: 'Subset' = Subset.TEST):
        if subset != Subset.TEST:
            raise AssertionError('Only TEST subset here.')
        if which_component in self.mapping_direct_test_pickles.keys():
            with open(self.mapping_direct_test_pickles[which_component], 'rb') as f:
                return pickle.load(f)
        elif which_component in [PickableComponents.FEATURE_MTX, PickableComponents.LABEL_MTX]:
            d = Dataset.load(self.dataset_test_fp)
            return d.feature_mtx if which_component == PickableComponents.FEATURE_MTX else d.label_mtx
        else:
            raise ValueError('which_component parameter has bad value.')

    def save_pickable_component(self, which_component: 'PickableComponents', pickable_obj):
        if which_component in self.mapping_direct_test_pickles.keys():
            fp = self.mapping_direct_test_pickles[which_component]
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, 'wb') as f:
                pickle.dump(pickable_obj, f, pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError('which_component parameter has bad value.')

    def find_best_fold_according_to_metric(self, metric: str = 'F_ma'):
        with open(self.all_validation_results_fp, 'r') as f:
            all_results = json.load(f)
        results = all_results[metric][self.th_version]
        return np.argmax(results)
