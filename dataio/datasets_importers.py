from scipy.sparse import coo_matrix
import numpy as np
from dataio.dataset import Dataset, MetaInfoContainer, DatasetInterpreter, Entity
import os.path


def import_from_matrixu_output_files(dataset_dir, dataset_name, verbose=0):

    def construct_meta_info_container(file_path, verbose=0):
        entities = []

        line_counter = 0
        freaks_counter = 0

        with open(file_path, 'r') as f:
            for line in f:
                line_counter += 1
                try:
                    name, orig_id = line.split()
                except ValueError as e:
                    s = line.split()
                    name = ''.join(s[:-1])
                    orig_id = s[-1]
                    if verbose > 0:
                        print("At line #{}, content '{}', orig. split='{}', correction='value={}, orig_id={}'".
                              format(line_counter, line, s, name, orig_id))
                    freaks_counter += 1

                entities.append(Entity(orig_id=int(orig_id), name=name))

        if verbose > 0:
            print('# freaks:', freaks_counter)

        return entities

    # construct document id - document title mapping
    file_path = "{}/{}-po_slowach-articles_dict-simple-20120507".format(dataset_dir, dataset_name)
    docs_mi = MetaInfoContainer(construct_meta_info_container(file_path, verbose=verbose - 1))

    # construct label id - label title mapping
    file_path = "{}/{}-po_slowach-cats_dict-simple-20120507".format(dataset_dir, dataset_name)
    labels_mi = MetaInfoContainer(construct_meta_info_container(file_path, verbose=verbose - 1))

    # construct feature id - feature name mapping
    file_path = "{}/{}-po_slowach-feature_dict-simple-20120507".format(dataset_dir, dataset_name)
    features_mi = MetaInfoContainer(construct_meta_info_container(file_path, verbose=verbose - 1))

    if verbose > 0:
        print('document dictionary length = ', len(docs_mi))
        print('labels dictionary length = ', len(labels_mi))
        print('features dictionary length = ', len(features_mi))

    # construct document id - list of label ids mapping
    file_path = "{}/{}-po_slowach-categories-simple-20120507".format(dataset_dir, dataset_name)
    row = []
    col = []
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                line_elements = line.split()
            except ValueError as e:
                if verbose > 0:
                    print(e)
                continue

            doc_orig_id = int(line_elements[0])
            doc_idx = docs_mi.idx_by_orig_id(doc_orig_id)
            list_of_cat_idxs = [labels_mi.idx_by_orig_id(int(el)) for el in line_elements[1:]]
            for cat_idx in list_of_cat_idxs:
                row.append(doc_idx)
                col.append(cat_idx)
                data.append(1)
    label_mtx = coo_matrix((data, (row, col)), shape=(len(docs_mi), len(labels_mi)), dtype=np.int32).tocsr()

    # construct document id - dict of features mapping
    file_path = "{}/{}-po_slowach-lista-simple-20120507".format(dataset_dir, dataset_name)
    row = []
    col = []
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                doc_orig_id, value = line.split("#")
            except ValueError as e:
                if verbose > 0:
                    print(e)
                continue
            doc_idx = docs_mi.idx_by_orig_id(int(doc_orig_id))
            features = value.split()
            for feature in features:
                try:
                    orig_feat_id, feat_value = feature.split("-", maxsplit=1)
                except ValueError as e:
                    if verbose > 0:
                        print(e)
                    continue
                row.append(doc_idx)
                col.append(features_mi.idx_by_orig_id(int(orig_feat_id)))
                data.append(float(feat_value))
    feature_mtx = coo_matrix((data, (row, col)), shape=(len(docs_mi), len(features_mi)), dtype=np.float32).tocsr()

    return Dataset(
        dataset_name,
        label_mtx,
        feature_mtx,
        DatasetInterpreter(docs_meta_info=docs_mi, labels_meta_info=labels_mi, features_meta_info=features_mi))


def import_dataset_from_xml_repo_bare(dataset_root_path: str, dataset_name: str):
    """

    :param dataset_root_path:
    :param dataset_name: e.g. 'eurlex'
    :return:
    """

    datasets = {}

    for subset in ['train', 'test']:
        file_path = os.path.join(dataset_root_path, dataset_name + '_' + subset + '.txt')
        feature_mtx, label_mtx = _read_xml_repo_data_file(file_path)
        datasets[subset] = Dataset(dataset_name + '_' + subset, label_mtx, feature_mtx)

    return datasets['train'], datasets['test']


def import_eurlex_from_xml_repo(dataset_root_path):
    return import_dataset_from_xml_repo_bare(dataset_root_path, 'eurlex')


def _construct_meta_info_container(fp):
    entities = []
    with open(fp, 'r', encoding="ISO-8859-1") as f:
        for n, line in enumerate(f):
            entities.append(Entity(orig_id=n, name=line.strip()))
    return entities


def import_dataset_from_xml_repo_with_metainfo(dataset_root_path: str, dataset_name: str):
    """

    :param dataset_root_path:
    :param dataset_name: e.g. 'Wiki10-31K', 'AmazonCat-13K'
    :return:
    """

    file_path = os.path.join(dataset_root_path, dataset_name + '_mappings', dataset_name + '_label_map.txt')
    labels_mi = MetaInfoContainer(_construct_meta_info_container(file_path))

    file_path = os.path.join(dataset_root_path, dataset_name + '_mappings', dataset_name + '_feature_map.txt')
    features_mi = MetaInfoContainer(_construct_meta_info_container(file_path))

    dataset_short_name = dataset_name.split(sep='-')[0]

    datasets = {}
    for subset in ['train', 'test']:
        fp = os.path.join(dataset_root_path, dataset_name + '_mappings', dataset_name + '_' + subset + '_map.txt')
        docs_mi_subset = MetaInfoContainer(_construct_meta_info_container(fp))

        fp = os.path.join(dataset_root_path, dataset_short_name, dataset_short_name + '_' + subset + '.txt')
        feature_mtx_train, label_mtx_train = _read_xml_repo_data_file(fp)
        datasets[subset] = Dataset(
            dataset_short_name + '_' + subset,
            label_mtx_train,
            feature_mtx_train,
            DatasetInterpreter(
                docs_meta_info=docs_mi_subset, labels_meta_info=labels_mi, features_meta_info=features_mi))

    return datasets['train'], datasets['test']


def import_lshtc3_dataset(dataset_root_path, verbose=4):
    raise NotImplementedError


def _read_xml_repo_data_file(file_path, verbose=1):
    if verbose > 0:
        print('Reading features mtx data')
    fm_row = []
    fm_col = []
    fm_data = []
    lm_row = []
    lm_col = []
    lm_data = []
    with open(file_path, 'r') as f:

        header_line_elements = next(f).split()
        n_docs = int(header_line_elements[0])
        n_features = int(header_line_elements[1])
        n_labels = int(header_line_elements[2])

        if verbose > 0:
            print('n_docs={n_docs}, n_features={n_features}, n_labels={n_labels}'.format(**locals()))

        for n, line in enumerate(f):
            if verbose > 0 and n % 100000 == 0:
                print('line {n}/{n_docs}'.format(**locals()))
            try:
                line_elements = line.split(' ')
            except ValueError as e:
                if verbose > 0:
                    print(e)
                continue
            if not line_elements[0] == '':      # in Eurlex dataset, there are train data examples without labels (why??)
                for l in line_elements[0].split(','):
                    lm_row.append(n)
                    lm_col.append(int(l))
                    lm_data.append(1)
            for kv in line_elements[1:]:
                k_v = kv.split(':')
                fm_row.append(n)
                fm_col.append(int(k_v[0]))
                fm_data.append(float(k_v[1]))

    feature_mtx = coo_matrix((fm_data, (fm_row, fm_col)), shape=(n_docs, n_features), dtype=np.float32).tocsr()
    label_mtx = coo_matrix((lm_data, (lm_row, lm_col)), shape=(n_docs, n_labels), dtype=np.int32).tocsr()

    return feature_mtx, label_mtx


def import_amazoncat13k_from_xml_repo(dataset_root_path: str):
    return import_dataset_from_xml_repo_with_metainfo(dataset_root_path, 'AmazonCat-13K')


def import_wiki10_from_xml_repo(dataset_root_path: str):
    return import_dataset_from_xml_repo_with_metainfo(dataset_root_path, 'Wiki10-31K')
