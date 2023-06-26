from dataio.dataset import Dataset, filter_train_test_datasets
from dataio.stratifying_data import get_train_val_test_idxs_from_stratification
from dataio.datasets_importers import import_from_matrixu_output_files, import_wiki10_from_xml_repo,\
    import_eurlex_from_xml_repo, import_amazoncat13k_from_xml_repo
import os.path
import pickle


def _test_matrixu_dataset_1():
    dataset_dir = os.path.expanduser("~/codes/kadrnn/data/simple_wiki_lev5/filtered_min_label_size_10")
    dataset_name = "kadr"

    pickled_file_path = dataset_dir+"/dataset.pkl"

    if os.path.isfile(pickled_file_path):
        print('Loading file: {}'.format(pickled_file_path))
        dataset = Dataset.load(pickled_file_path)
    else:
        print('Creating dataset from folder {}'.format(dataset_dir))
        dataset = import_from_matrixu_output_files(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            verbose=2)
        print('Saving dataset to file: {}'.format(pickled_file_path))
        dataset.save(pickled_file_path)

    print(dataset.get_summary(add_mtx_stats=True))

    doc_ids = [0, 1, 57]
    for doc_id in doc_ids:
        print("-------")
        print(dataset.interpreter.doc_string(doc_id, include_orig_id=True, include_name=True))
        print(dataset.interpreter.doc_labels(doc_id, label_mtx=dataset.label_mtx, print_original_ids=True))
        print(dataset.interpreter.doc_features(doc_id, features_mtx=dataset.feature_mtx, print_values=True))


def matrixu_from_dataset_with_stratification_to_two_datasets():
    dataset_dir = os.path.expanduser("~/codes/kadrnn/data/simple_wiki_lev5/filtered_min_label_size_10")
    pickled_file_path = dataset_dir+"/dataset.pkl"
    print('Loading file: {}'.format(pickled_file_path))
    dataset = Dataset.load(pickled_file_path)
    print(dataset.get_summary(add_mtx_stats=True))
    train_dataset, val_dataset, test_dataset = dataset.data_subsets_from_stratification(val_folds=[], test_folds=[0])
    print(train_dataset.get_summary(add_mtx_stats=True))
    print(test_dataset.get_summary(add_mtx_stats=True))

    doc_ids = [0, 1, 57]
    for dset in [dataset, train_dataset, test_dataset]:
        print("---------------------")
        for doc_id in doc_ids:
            print("-------")
            print(dset.interpreter.doc_string(doc_id, include_orig_id=True, include_name=True))
            print(dset.interpreter.doc_labels(doc_id, label_mtx=dset.label_mtx, print_original_ids=True))
            print(dset.interpreter.doc_features(doc_id, features_mtx=dset.feature_mtx, print_values=True))

    dataset_dir2 = dataset_dir+'/fold0'
    file_paths_suffixes = ['dataset_train.pkl', 'dataset_test.pkl']
    for dset, fp_suffix in zip([train_dataset, test_dataset], file_paths_suffixes):
        file_path = os.path.join(dataset_dir2, fp_suffix)
        print('Saving dataset to file: {}'.format(file_path))
        dset.save(file_path)


def train_dataset_to_k_folds(pickled_train_dataset_path: str, k_folds: int = 10) -> None:

    print('Loading file: {}'.format(pickled_train_dataset_path))
    dataset = Dataset.load(pickled_train_dataset_path)
    print(dataset.get_summary(add_mtx_stats=True))

    print('Do iterative stratificaiton')
    dataset.stratify(k_folds)

    dataset_dir = os.path.dirname(pickled_train_dataset_path)
    for fold in range(k_folds):

        fold_dir = os.path.join(dataset_dir, 'val_fold'+str(fold))
        os.makedirs(fold_dir)

        train_dataset, val_dataset, _ = dataset.data_subsets_from_stratification(
            val_folds=[fold], test_folds=[])

        for dset, fp_suffix in zip(
                [train_dataset, val_dataset], ['dataset_train.pkl', 'dataset_validation.pkl']):
            dset.save(os.path.join(fold_dir, fp_suffix))


def save_feature_mtxs_for_python2_in_all_folds(root_dir: str) -> None:

    val_fold_templ = root_dir + '/val_fold{ifold}'
    for ifold in range(0, 10):
        print('I\'m in {} fold.'.format(ifold))
        val_fold_path = val_fold_templ.format(ifold=ifold)
        for subset in ['train', 'validation']:
            dataset = Dataset.load(val_fold_path + '/dataset_{subset}.pkl'.format(subset=subset))
            dataset.save_feature_mtx(val_fold_path +'/feat_mtx_{subset}.pkl'.format(subset=subset), protocol=2)
            dataset.save_label_mtx(val_fold_path +'/lab_mtx_{subset}.pkl'.format(subset=subset), protocol=2)


def _test_eurlex_import_1():
    dataset_dir = os.path.expanduser("~/codes/kadrnn/data/Eurlex")

    pickled_file_path = dataset_dir+"/saved.pkl"

    if os.path.isfile(pickled_file_path):
        print('Loading file: {}'.format(pickled_file_path))
        with open(pickled_file_path, 'rb') as f:
            [train_dataset, test_dataset] = pickle.load(f)
    else:
        print('Creating dataset from folder {}'.format(dataset_dir))
        train_dataset, test_dataset = import_eurlex_from_xml_repo(dataset_root_path=dataset_dir)
        print('Saving dataset to file: {}'.format(pickled_file_path))
        with open(pickled_file_path, 'wb') as f:
            pickle.dump([train_dataset, test_dataset], f, pickle.HIGHEST_PROTOCOL)

    print('TRAIN DATASET')
    train_dataset.get_summary(add_mtx_stats=True)

    print('TEST DATASET')
    test_dataset.get_summary(add_mtx_stats=True)

    new_train_dataset, new_test_dataset = filter_train_test_datasets(train_dataset, test_dataset, verbose=4)

    print('FILTERED TRAIN DATASET')
    print(new_train_dataset.get_summary(add_mtx_stats=True))
    with open(os.path.join(dataset_dir, 'dataset_train.pkl'), 'wb') as f:
        pickle.dump(new_train_dataset, f, pickle.HIGHEST_PROTOCOL)

    print('FILTERED TEST DATASET')
    print(new_test_dataset.get_summary(add_mtx_stats=True))
    with open(os.path.join(dataset_dir, 'dataset_test.pkl'), 'wb') as f:
        pickle.dump(new_test_dataset, f, pickle.HIGHEST_PROTOCOL)

    pickled_file_path = dataset_dir+"/filtered.pkl"
    print('Saving filtered dataset to file: {}'.format(pickled_file_path))
    with open(pickled_file_path, 'wb') as f:
        pickle.dump([new_train_dataset, new_test_dataset], f, pickle.HIGHEST_PROTOCOL)


def _test_wiki10_dataset_1():
    dataset_dir = os.path.expanduser("~/data/xml_repo_datasets/wiki10")

    pickled_file_path = dataset_dir+"/saved.pkl"

    if os.path.isfile(pickled_file_path):
        print('Loading file: {}'.format(pickled_file_path))
        with open(pickled_file_path, 'rb') as f:
            [train_dataset, test_dataset] = pickle.load(f)
    else:
        print('Creating dataset from folder {}'.format(dataset_dir))
        train_dataset, test_dataset = import_wiki10_from_xml_repo(dataset_root_path=dataset_dir)
        print('Saving dataset to file: {}'.format(pickled_file_path))
        with open(pickled_file_path, 'wb') as f:
            pickle.dump([train_dataset, test_dataset], f, pickle.HIGHEST_PROTOCOL)

    print('TRAIN DATASET')
    print(train_dataset.get_summary(add_mtx_stats=True))

    print('TEST DATASET')
    print(test_dataset.get_summary(add_mtx_stats=True))

    doc_ids = [0, 1, 57]
    for doc_id in doc_ids:
        print("-------")
        print(train_dataset.interpreter.doc_string(doc_id, include_orig_id=True, include_name=True))
        print(train_dataset.interpreter.doc_labels(doc_id, label_mtx=train_dataset.label_mtx, print_original_ids=True))
        print(train_dataset.interpreter.doc_features(doc_id, features_mtx=train_dataset.feature_mtx, print_values=True))

    doc_ids = [0, 1, 57]
    for doc_id in doc_ids:
        print("-------")
        print(test_dataset.interpreter.doc_string(doc_id, include_orig_id=True, include_name=True))
        print(test_dataset.interpreter.doc_labels(doc_id, label_mtx=test_dataset.label_mtx, print_original_ids=True))
        print(test_dataset.interpreter.doc_features(doc_id, features_mtx=test_dataset.feature_mtx, print_values=True))


def _test_wiki10_dataset_join_filtering():
    dataset_dir = os.path.expanduser("~/codes/kadrnn/data/Wiki10-31K")

    pickled_file_path = dataset_dir+"/saved.pkl"

    if os.path.isfile(pickled_file_path):
        print('Loading file: {}'.format(pickled_file_path))
        with open(pickled_file_path, 'rb') as f:
            [train_dataset, test_dataset] = pickle.load(f)
    else:
        print('Creating dataset from folder {}'.format(dataset_dir))
        train_dataset, test_dataset = import_wiki10_from_xml_repo(dataset_root_path=dataset_dir)
        print('Saving dataset to file: {}'.format(pickled_file_path))
        with open(pickled_file_path, 'wb') as f:
            pickle.dump([train_dataset, test_dataset], f, pickle.HIGHEST_PROTOCOL)

    new_train_dataset, new_test_dataset = filter_train_test_datasets(train_dataset, test_dataset, verbose=4)

    print('TRAIN DATASET')
    print(new_train_dataset.get_summary(add_mtx_stats=True))

    print('TEST DATASET')
    print(new_test_dataset.get_summary(add_mtx_stats=True))

    pickled_file_path = dataset_dir+"/filtered.pkl"
    print('Saving dataset to file: {}'.format(pickled_file_path))
    with open(pickled_file_path, 'wb') as f:
        pickle.dump([new_train_dataset, new_test_dataset], f, pickle.HIGHEST_PROTOCOL)

    doc_ids = [0, 1, 57]
    for doc_id in doc_ids:
        print("-------")
        print(new_train_dataset.interpreter.doc_string(doc_id, include_orig_id=True, include_name=True))
        print(new_train_dataset.interpreter.doc_labels(doc_id, label_mtx=new_train_dataset.label_mtx, print_original_ids=True))
        print(new_train_dataset.interpreter.doc_features(doc_id, features_mtx=new_train_dataset.feature_mtx, print_values=True))

    doc_ids = [0, 1, 57]
    for doc_id in doc_ids:
        print("-------")
        print(new_test_dataset.interpreter.doc_string(doc_id, include_orig_id=True, include_name=True))
        print(new_test_dataset.interpreter.doc_labels(doc_id, label_mtx=new_test_dataset.label_mtx, print_original_ids=True))
        print(new_test_dataset.interpreter.doc_features(doc_id, features_mtx=new_test_dataset.feature_mtx, print_values=True))


if __name__ == "__main__":
    # root_dir = os.path.expanduser(
    #     "~/codes/kadrnn/data/simple_wiki_lev5/filtered_min_label_size_10/fold_0_as_testset")
    # dataset_train = Dataset.load(os.path.join(root_dir, 'dataset_whole_train.pkl'))
    # dataset_train.get_summary(add_mtx_stats=True)
    # pass

    # p = os.path.expanduser(
    #     "~/codes/kadrnn/data/simple_wiki_lev5/filtered_min_label_size_10/dataset.pkl")
    p = os.path.expanduser(
        "~/codes/kadrnn/data/simple_wiki_lev5/filtered_min_label_size_10/fold_0_as_testset/val_fold0/dataset_train.pkl")
    dataset_train = Dataset.load(p)
    dataset_train.get_summary(add_mtx_stats=False)
    p = os.path.expanduser(
        "~/codes/kadrnn/data/simple_wiki_lev5/filtered_min_label_size_10/fold_0_as_testset/val_fold0/dataset_validation.pkl")
    dataset_train = Dataset.load(p)
    dataset_train.get_summary(add_mtx_stats=False)
    pass

    # save_feature_mtxs_for_python2_in_all_folds(root_dir)
    # root_dir = os.path.expanduser(
    #     "~/codes/kadrnn/data/Eurlex")
    # dataset_train, dataset_test = import_eurlex_from_xml_repo(root_dir)
    # dataset_train, dataset_test = import_amazoncat13k_from_xml_repo(root_dir)   #requires more than 8GB of RAM !!!
    # dataset_train.save(os.path.join(root_dir, 'dataset_train.pkl'))
    # dataset_test.save(os.path.join(root_dir, 'dataset_test.pkl'))
    # save_feature_mtxs_for_python2_in_all_folds(root_dir)
    # dataset_path = os.path.expanduser(
    #     "~/codes/kadrnn/data/simple_wiki_lev5/filtered_min_label_size_10/fold_0_as_testset/dataset_whole_train.pkl")
    # train_dataset_to_k_folds(dataset_path)
    # matrixu_from_dataset_with_stratification_to_two_datasets()
    # _test_wiki10_dataset_1()
    # _test_wiki10_dataset_join_filtering()
    # _test_eurlex_import_1()

    # dataset_dir = get_home_dir() + "/data/21cats-add"
    # dataset_name = "21cats"
    #
    # matrixu = MatrixuGeneratedDataImporter(
    #               dataset_dir=dataset_dir, dataset_name=dataset_name, n_folds_stratification=10,
    #               verbose=2)
    # print(matrixu.features_mtx.shape)
    # print(matrixu.labels_mtx.shape)
    # matrixu.save(dataset_dir+"/matrixu_generated_data.pkl")
    #
    # matrixu = MatrixuGeneratedDataImporter.load(dataset_dir + "/matrixu_generated_data.pkl")
    # for n, i_fold in enumerate(matrixu.stratification):
    #     print('Fold {}:'.format(n))
    #     aux = matrixu.labels_mtx[i_fold, :]
    #     n_representants_per_label = np.array(np.sum(aux, axis=0)).flatten()
    #     for i_lab, r in enumerate(n_representants_per_label):
    #         print('Label {}: {} articles.'.format(i_lab, r))
