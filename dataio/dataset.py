from scipy.sparse import vstack
from scipy.sparse import find as sparse_find
import numpy as np
import pickle
from copy import deepcopy
from dataio.stratifying_data import iterative_stratification, get_train_val_test_idxs_from_stratification
from utils.sparse_mtx_functions import filter_sparse_matrix
from typing import List


class Entity:
    def __init__(self, orig_id, name, content=None):
        self.orig_id = orig_id
        self.name = name
        self.content = content

    def to_string(self, position=None, include_name=True, include_orig_id=False, include_content=False):
        result = ""
        if include_name:
            result += '"{}"'.format(self.name)

        if position is not None or include_orig_id:
            if result != "":
                result += ' '
            result += '('

            if position is not None:
                result += 'id={}'.format(position)

            if include_orig_id:
                if position is not None:
                    result += ', '
                if self.orig_id is not None:
                    result += 'orig_id={}'.format(self.orig_id)
                else:
                    result += 'orig_id=None'

            result += ')'

        if include_content:
            if result != "":
                result += ': '
            if self.content is not None:
                result += '"{}"'.format(self.content)
            else:
                result += '"<no original text data available>"'

        return result


class MetaInfoContainer:
    def __init__(self, list_of_entities):
        self._list_of_entities = list_of_entities
        self._orig_id_2_entity_idx = None
        self._name_2_entity_idx = None
        self.create_inv_dicts()

    def __getitem__(self, item):
        return self._list_of_entities[item]

    def __iter__(self):
        for en in self._list_of_entities:
            yield en

    def __len__(self):
        return len(self._list_of_entities)

    def by_id(self, item):
        return self._list_of_entities[item]

    def create_inv_dicts(self):
        self._name_2_entity_idx = {en.name: n for n, en in enumerate(self._list_of_entities)}
        self._orig_id_2_entity_idx = {en.orig_id: n for n, en in enumerate(self._list_of_entities)}

    def idx_by_name(self, name):
        return self._name_2_entity_idx[name]

    def idx_by_orig_id(self, orig_id):
        return self._orig_id_2_entity_idx[orig_id]

    def update_container(self, positions_to_be_left):
        new_list_of_entries = [self._list_of_entities[pos] for pos in positions_to_be_left]
        self._list_of_entities = new_list_of_entries
        self.create_inv_dicts()


class DatasetInterpreter:
    def __init__(self, docs_meta_info, labels_meta_info=None, features_meta_info=None):
        self.docs_meta_info = MetaInfoContainer(docs_meta_info)
        self.labels_meta_info = MetaInfoContainer(labels_meta_info)
        self.features_meta_info = MetaInfoContainer(features_meta_info)

    def doc_string(self, doc_id, include_name=True, include_id=False, include_orig_id=False, include_content=False):
        return self.docs_meta_info[doc_id].to_string(position=doc_id if include_id else None,
                                                     include_name=include_name,
                                                     include_orig_id=include_orig_id,
                                                     include_content=include_content)

    def doc_labels(self, doc_id, label_mtx, print_original_ids=False):
        result = ""
        _, labels, _ = sparse_find(label_mtx[doc_id, :])
        for l_id in labels:
            result += self.labels_meta_info[l_id].to_string(include_orig_id=print_original_ids) + ', '
        return result[:-2]

    def label_docs(self, label_id, label_mtx, print_original_ids=False):
        result = ""
        docs, _, _ = sparse_find(label_mtx[:, label_id])
        for doc_id in docs:
            result += self.docs_meta_info[doc_id].to_string(include_orig_id=print_original_ids) + ', '
        return result[:-2]

    def doc_features(self, doc_id, features_mtx, print_values=True):
        result = ""
        _, features_ids, features_values = sparse_find(features_mtx[doc_id, :])
        order = np.argsort(features_values)[::-1]
        for (feature_id, feature_value) in zip(features_ids[order], features_values[order]):
            if print_values:
                result += self.features_meta_info[feature_id].name + ' ({:.3f}), '.format(feature_value)
            else:
                result += self.features_meta_info[feature_id].name + ', '
        return result[:-2]

    def feature_docs(self, feature_id, features_mtx, print_values=True):
        result = ""
        docs_ids, _, features_values = sparse_find(features_mtx[:, feature_id])
        for (doc_id, feature_value) in zip(docs_ids, features_values):
            if print_values:
                result += self.docs_meta_info[doc_id].name + ' ({:.3f}), '.format(feature_value)
            else:
                result += self.docs_meta_info[doc_id].name + ', '
        return result[:-2]

    def info_about_nearest_neighbors(self, nns, dists=None, max_k=None):
        if max_k is None:
            max_k = len(nns)
        result = ""
        nns = np.array(nns, dtype=np.int32)
        if dists is not None:
            for n, (nn_doc_id, nn_dists) in enumerate(zip(nns, dists)):
                if n == max_k:
                    break
                result += '{}.'.format(n+1) + self.docs_meta_info[nn_doc_id].name + ' ({:.3f}), '.format(nn_dists)
        else:
            for n, nn_doc_id in enumerate(nns):
                if n == max_k:
                    break
                result += '{}.'.format(n+1) + self.docs_meta_info[nn_doc_id].name + ', '
        return result[:-2]

    def info_about_nearest_neighbors_in_dataset_subpart(self, train_docs_ids, nns, dists=None, max_k=None):
        if max_k is None:
            max_k = len(nns)
        result = ""
        nns = np.array(nns, dtype=np.int32)
        if dists is not None:
            for n, (nn_subpart_doc_id, nn_dist) in enumerate(zip(nns, dists)):
                if n == max_k:
                    break
                result += '{}.'.format(n+1) + self.docs_meta_info[train_docs_ids[nn_subpart_doc_id]].name +\
                          ' ({:.3f}), '.format(nn_dist)
        else:
            for n, nn_subpart_doc_id in enumerate(nns):
                if n == max_k:
                    break
                result += '{}.'.format(n+1) + self.docs_meta_info[train_docs_ids[nn_subpart_doc_id]].name + ', '
        return result[:-2]

    def info_about_commons_in_docs(self, list_of_doc_ids, feature_mtx, label_mtx):
        all_on_features_ids, all_on_labels_ids = DatasetInterpreter._get_all_on_features_and_labels_of_selected_docs(
            list_of_doc_ids, feature_mtx, label_mtx)

        print("Docs: ", end='')
        print(*[self.docs_meta_info[doc_id].name for doc_id in list_of_doc_ids], sep=' ', end=' have in common:')
        print('Features:')
        for feature_id in all_on_features_ids:
            print(self.features_meta_info[feature_id].name, ': ',
                  np.array(feature_mtx[list_of_doc_ids, feature_id]).flatten())

        print('Labels:')
        print(*[self.labels_meta_info[label_id].name for label_id in all_on_labels_ids], sep=' ')

    @staticmethod
    def _get_all_on_features_and_labels_of_selected_docs(list_of_doc_ids, feature_mtx, label_mtx):
        def get_all_on_cols_ids(rows_ids, mtx):
            rows = mtx[rows_ids, :]
            return np.arange(rows.shape[1])[np.array(np.sum(rows, axis=0)).flatten() == rows.shape[0]]

        all_on_features_ids = get_all_on_cols_ids(list_of_doc_ids, feature_mtx)
        all_on_labels_ids = get_all_on_cols_ids(list_of_doc_ids, label_mtx)
        return all_on_features_ids, all_on_labels_ids

    def update(self, docs_ids_left, labels_ids_left, features_ids_left):
        self.docs_meta_info.update_container(docs_ids_left)
        self.labels_meta_info.update_container(labels_ids_left)
        self.features_meta_info.update_container(features_ids_left)


class Dataset:
    def __init__(self, name, label_mtx, feature_mtx, interpreter=None, stratification=None):
        self.name = name
        assert label_mtx.shape[0] == feature_mtx.shape[0]
        self.label_mtx = label_mtx
        self.feature_mtx = feature_mtx
        self.interpreter = interpreter
        self.stratification = stratification

    @property
    def n_docs(self):
        if self.label_mtx is not None:
            return self.label_mtx.shape[0]
        else:
            return None

    @property
    def n_labels(self):
        if self.label_mtx is not None:
            return self.label_mtx.shape[1]
        else:
            return None

    @property
    def n_features(self):
        if self.feature_mtx is not None:
            return self.feature_mtx.shape[1]
        else:
            return None

    @staticmethod
    def load(filepath) -> 'Dataset':
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def save_feature_mtx(self, filepath, protocol=pickle.HIGHEST_PROTOCOL):
        with open(filepath, 'wb') as f:
            pickle.dump(self.feature_mtx, f, protocol=protocol)

    def save_label_mtx(self, filepath, protocol=pickle.HIGHEST_PROTOCOL):
        with open(filepath, 'wb') as f:
            pickle.dump(self.label_mtx, f, protocol=protocol)

    def stratify(self, n_folds, verbose=0):
        if verbose > 0:
            print('Creating {}-fold stratification of data.'.format(n_folds))
        self.stratification = iterative_stratification(labels_mtx=self.label_mtx, n_folds=n_folds)

    def data_subsets_from_stratification(self, val_folds: List, test_folds: List):
        """
        Divides the dataset into three subsets: train, val and test according to the stratification.
        If stratification is not yet done - an error is raised. if val_folds is None or empty, val_dataset
        is not created. Similarly test_dataset. Interpreters, if available, are preserved.

        :param val_folds:
        :param test_folds:
        :return: triple of train_dataset, val_dataset, test_dataset
        """

        if self.stratification is None:
            raise AssertionError('Dataset must have stratification field. Do dataset.stratify() first')

        train_idxs, val_idxs, test_idxs = get_train_val_test_idxs_from_stratification(
            self.stratification, val_fold_list=val_folds, test_fold_list=test_folds)

        labels_ids_left = range(self.n_labels)
        features_ids_left = range(self.n_features)

        data_subsets = []
        idxs_subsets = [train_idxs, val_idxs, test_idxs]
        name_suffixes = ['train', 'val', 'test']
        for idxs, name_suffix in zip(idxs_subsets, name_suffixes):
            if idxs is not None and idxs != []:
                x = self.feature_mtx[idxs, :]
                t = self.label_mtx[idxs, :]
                if self.interpreter is not None:
                    interpreter = deepcopy(self.interpreter)
                    interpreter.update(idxs, labels_ids_left, features_ids_left)
                else:
                    interpreter = None
                data_subsets.append(Dataset(self.name+'_{}'.format(name_suffix), label_mtx=t, feature_mtx=x,
                                            interpreter=interpreter))
            else:
                data_subsets.append(None)

        return data_subsets[0], data_subsets[1], data_subsets[2]

    def filter(self, min_label_size=2, min_feature_size=2, min_labels_per_doc=1, min_features_per_doc=1,
               verbose=0):

        docs_ids_left = np.arange(self.label_mtx.shape[0])
        labels_ids_left = np.arange(self.label_mtx.shape[1])
        features_ids_left = np.arange(self.feature_mtx.shape[1])

        filtering_main_iterations = 0
        while True:
            filtering_main_iterations += 1
            if verbose > 0:
                print('MAIN FILTERING iteration ', filtering_main_iterations)

            if verbose > 0:
                print('\tFiltering docs/labels matrix:')
            self.label_mtx, removed_rows, removed_cols, _ = filter_sparse_matrix(
                self.label_mtx,
                min_nonzeros_per_row=min_labels_per_doc,
                min_nonzeros_per_col=min_label_size,
                verbose=verbose-1)
            docs_ids_left = np.delete(docs_ids_left, removed_rows)
            labels_ids_left = np.delete(labels_ids_left, removed_cols)

            if verbose > 0:
                print('\tFiltering docs/features matrix:')
            self.feature_mtx = self.feature_mtx[np.setdiff1d(np.arange(self.feature_mtx.shape[0]), removed_rows), :]
            self.feature_mtx, removed_rows, removed_cols, _ = filter_sparse_matrix(
                self.feature_mtx,
                min_nonzeros_per_row=min_features_per_doc,
                min_nonzeros_per_col=min_feature_size,
                verbose=verbose-1)

            docs_ids_left = np.delete(docs_ids_left, removed_rows)
            features_ids_left = np.delete(features_ids_left, removed_cols)

            if len(removed_rows) == 0:
                break
            else:
                self.label_mtx = self.label_mtx[np.setdiff1d(np.arange(self.label_mtx.shape[0]), removed_rows), :]

        # update_meta_info_containers
        if self.interpreter is not None:
            self.interpreter.update(docs_ids_left, labels_ids_left, features_ids_left)

    def get_summary(self, add_mtx_stats=False, do_print_to_console=True):

        def mtx_statistics(mtx, axis):
            bins = [0, 1, 3, 10, 30, 100, 300, 1000, 3000, float('inf')]
            aggr = np.array(np.sum(mtx > 0, axis=axis))
            counts, bins = np.histogram(aggr, bins)
            result_ = ""
            for (c, bs, be) in zip(counts, bins[:-1], bins[1:]):
                result_ += "in [{}-{}): {} items\n".format(bs, be, c)
            result_ += "min={}, mean={}, max={}\n".format(np.min(aggr), np.mean(aggr), np.max(aggr))
            return result_

        result = "{} dataset summary:\n".format(self.name)

        result += "Label_mtx: "
        if self.label_mtx is not None:
            result += "type={}, shape={}, nnz={}, dtype={}\n".format(type(self.label_mtx), self.label_mtx.shape,
                                                                     self.label_mtx.nnz, self.label_mtx.dtype)
            if add_mtx_stats:
                result += "Documents per label:\n" + mtx_statistics(self.label_mtx, axis=0)
                result += "Labels per document:\n" + mtx_statistics(self.label_mtx, axis=1)
        else:
            result += "None\n"

        result += "Features_mtx: "
        if self.feature_mtx is not None:
            result += "type={}, shape={}, nnz={}, dtype={}\n".format(type(self.feature_mtx), self.feature_mtx.shape,
                                                                     self.feature_mtx.nnz, self.feature_mtx.dtype)
            if add_mtx_stats:
                result += "Documents per feature:\n" + mtx_statistics(self.feature_mtx, axis=0)
                result += "Features per document:\n" + mtx_statistics(self.feature_mtx, axis=1)

        else:
            result += "None\n"

        result += "nDocs = {}, nLabels = {}, nFeatures = {}\n".format(self.n_docs, self.n_labels, self.n_features)
        if self.interpreter is not None:
            result += "Interpreter available"
        else:
            result += "Interpreter not available"
        if self.stratification is not None:
            result += "\nStratification available"
        else:
            result += "\nStratification not available"

        if do_print_to_console:
            print(result)

        return result


def filter_train_test_datasets(train_dataset, test_dataset,
                               min_label_size=1, min_feature_size=1, verbose=0):

    n_train_docs = train_dataset.n_docs

    lm = vstack((train_dataset.label_mtx, test_dataset.label_mtx))
    fm = vstack((train_dataset.feature_mtx, test_dataset.feature_mtx))

    if train_dataset.interpreter is not None and test_dataset.interpreter is not None:
        doc_mi = MetaInfoContainer([el for el in train_dataset.interpreter.docs_meta_info] +
                                   [el for el in test_dataset.interpreter.docs_meta_info])

        joined_dataset = Dataset('joined', lm, fm,
                                 DatasetInterpreter(doc_mi,
                                                    train_dataset.interpreter.labels_meta_info,
                                                    train_dataset.interpreter.features_meta_info))
    else:
        joined_dataset = Dataset('joined', lm, fm)

    joined_dataset.filter(min_label_size, min_feature_size,
                          min_labels_per_doc=0, min_features_per_doc=0,   # rows cannot be removed
                          verbose=verbose)

    if verbose > 0:
        print('Joined TRAIN_TEST statistics:')
        print(joined_dataset.get_summary(add_mtx_stats=True))

    if joined_dataset.interpreter is not None:
        train_interpreter = DatasetInterpreter(train_dataset.interpreter.docs_meta_info,
                                               joined_dataset.interpreter.labels_meta_info,
                                               joined_dataset.interpreter.features_meta_info)

        test_interpreter = DatasetInterpreter(test_dataset.interpreter.docs_meta_info,
                                              joined_dataset.interpreter.labels_meta_info,
                                              joined_dataset.interpreter.features_meta_info)
    else:
        train_interpreter = test_interpreter = None

    new_train_dataset = Dataset(
        train_dataset.name,
        joined_dataset.label_mtx[:n_train_docs, :],
        joined_dataset.feature_mtx[:n_train_docs, :],
        interpreter=train_interpreter)

    new_test_dataset = Dataset(
        test_dataset.name,
        joined_dataset.label_mtx[n_train_docs:, :],
        joined_dataset.feature_mtx[n_train_docs:, :],
        interpreter=test_interpreter)

    return new_train_dataset, new_test_dataset
