import csv
import os
import pickle
import numpy as np
import glob
import scipy.sparse as sp
# from dataio.dataset import Dataset


def create_output_folders_structure(output_path):

    # create the 'dataset' folder
    os.makedirs(output_path, exist_ok=True)

    # create the 'train' subfolder and its 10 subfolders
    train_path = os.path.join(output_path, 'train')
    os.makedirs(train_path, exist_ok=True)
    for i in range(10):
        val_fold_path = os.path.join(train_path, f'val_fold_{i}')
        os.makedirs(val_fold_path, exist_ok=True)
        scores_path = os.path.join(val_fold_path, f'scores')
        os.makedirs(scores_path, exist_ok=True)
        knn_path = os.path.join(scores_path, f'knn')
        os.makedirs(knn_path, exist_ok=True)
        leml_path = os.path.join(scores_path, f'leml')
        os.makedirs(leml_path, exist_ok=True)

    # create the 'test' subfolder
    test_path = os.path.join(output_path, 'test')
    os.makedirs(test_path, exist_ok=True)
    scores_path = os.path.join(test_path, f'scores')
    os.makedirs(scores_path, exist_ok=True)
    knn_path = os.path.join(scores_path, f'knn')
    os.makedirs(knn_path, exist_ok=True)
    leml_path = os.path.join(scores_path, f'leml')
    os.makedirs(leml_path, exist_ok=True)


def file_pickle_to_npz(input_path, output_path):
    # Load the object from the input file
    encoding = 'latin1' if 'leml' in input_path else 'ASCII'
    with open(input_path, 'rb') as f:
        obj = pickle.load(f, encoding=encoding)

    # Check if the object is a CSR sparse matrix
    if sp.isspmatrix_csr(obj):
        # Save the matrix to the output file in NPZ format
        sp.save_npz(output_path, obj)


def pickle_to_npz(input_path, output_path):
    for i in range(10):
        print(f'Processing fold {i}')

        i_path = os.path.join(input_path, f'val_fold{i}')
        o_path = os.path.join(output_path, 'train', f'val_fold_{i}')

        # datasets
        for f_name in ['feature_mtx', 'label_mtx']:
            o_f_name = 'X' if f_name == 'feature_mtx' else 'Y'
            for subset in ['train', 'validation']:
                i_name = f_name + '_' + subset + '.pkl'
                o_name = o_f_name + '_' + subset + '.npz'
                file_pickle_to_npz(os.path.join(i_path, i_name), os.path.join(o_path, o_name))

        # scores
        for scorer in ['kadrnn', 'leml']:
            o_scorer = 'knn' if scorer == 'kadrnn' else scorer
            for subset in ['train', 'validation']:
                i_name = 'scores_' + subset + '.pkl'
                o_name = 'scores_' + subset + '.npz'
                file_pickle_to_npz(os.path.join(i_path, scorer, i_name), os.path.join(o_path, 'scores', o_scorer, o_name))


def pickle_to_npz_test(input_path, output_path):

    # datasets
    for f_name in ['feature_mtx', 'label_mtx']:
        o_f_name = 'X' if f_name == 'feature_mtx' else 'Y'
        i_name = f_name + '_test.pkl'
        o_name = o_f_name + '_test.npz'
        file_pickle_to_npz(os.path.join(input_path, i_name), os.path.join(output_path, o_name))

    # scores
    for scorer in ['kadrnn', 'leml']:
        o_scorer = 'knn' if scorer == 'kadrnn' else scorer
        i_name = 'scores_test.pkl'
        o_name = 'scores_test.npz'
        file_pickle_to_npz(os.path.join(input_path, scorer, i_name), os.path.join(output_path, 'scores', o_scorer, o_name))


def sparse_matrix_equal(a, b, format=None):
    if not sp.issparse(a) or not sp.issparse(b):
        return False

    if format is not None:
        a = a.asformat(format)
        b = b.asformat(format)

    if a.shape != b.shape:
        return False

    if a.nnz != b.nnz:
        return False

    if not np.allclose(a.data, b.data):
        return False

    if not np.array_equal(a.indices, b.indices):
        return False

    if not np.array_equal(a.indptr, b.indptr):
        return False

    return True


if __name__ == "__main__":
    # root_dir = os.path.expanduser(
    #     "~/codes/kadrnn/data/simple_wiki_lev5/filtered_min_label_size_10/fold_0_as_testset")
    # dataset_train = Dataset.load(os.path.join(root_dir, 'dataset_whole_train.pkl'))
    # dataset_train.get_summary(add_mtx_stats=True)
    input_root_dir = os.path.expanduser(f'~/codes/kadrnn/data/simple_wiki_lev5/filtered_min_label_size_10/fold_0_as_testset')
    output_root_dir = os.path.expanduser(f'~/codes/kadrnn/data/SimpleWikiLev5CatMinSize10')

    # 1.
    # create_output_folders_structure(output_root_dir)

    # 2.
    # pickle_to_npz(input_root_dir, output_root_dir)

    # 3.
    pickle_to_npz_test(os.path.join(input_root_dir, 'test_data'), os.path.join(output_root_dir, 'test'))