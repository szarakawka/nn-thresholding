from tensorflow import keras
import tensorflow as tf
from utils.batch_work import BatchedData
from utils.etc import pi_range
from utils.fixes import correct_absolute_paths
from utils.sparse_mtx_functions import dense_outputs_to_sparse_predictions
from utils.experiments_utils import calculate_sample_weights_for_multi_label_problem
from custom_layers.metric_ops import macro_f1score, micro_f1score
from custom_layers.losses import macro_double_soft_f1
from custom_layers.sparse_embedding_lookup import SparseEmbeddingLookup
from custom_layers.sparse_residual_add import SparseResidualAdd
from scipy.sparse import coo_matrix, vstack, csr_matrix, lil_matrix
import numpy as np
import os.path
from glob import glob
from lib.hopt import search
from lib.hopt.keras import HoptCallback
from nn_thresholding.hyperparameters_config import HP, load_hp
from phases.thresholding import scale_scores
from tensorflow_addons.metrics import F1Score
from time import sleep
import pickle


def preprocess_scores_mtx(scores_mtx, max_highest_scores_per_sample=None):
    """ Transforms scores mtx of csr_matrix type to a pair of arrays (scores_idxs, scores_values).

    :param scores_mtx: scipy.sparse.csr_matrix [n_samples, n_labels]
    :param max_highest_scores_per_sample: if None (default), then this is equal to the maximum number of non-zero
    elements per sample
    :return: (np.array() , np.array()) - scores_idxs, scores_values, both of shape [n_samples, max_scores_per_sample]
    """

    lil_scores = scores_mtx.tolil()
    orders = [np.argsort(scores)[::-1] for scores in lil_scores.data]
    scores_idxs_sorted = [list(np.array(s)[o]) for s, o in zip(lil_scores.rows, orders)]
    scores_values_sorted = [list(np.array(s)[o]) for s, o in zip(lil_scores.data, orders)]

    scores_idxs_sorted_padded = keras.preprocessing.sequence.pad_sequences(
        scores_idxs_sorted, maxlen=max_highest_scores_per_sample, padding='post', truncating='post',
        value=scores_mtx.shape[1])
    scores_values_sorted_padded = keras.preprocessing.sequence.pad_sequences(
        scores_values_sorted, maxlen=max_highest_scores_per_sample, padding='post', truncating='post',
        value=0., dtype='float32')
    return scores_idxs_sorted_padded, scores_values_sorted_padded


class ModelT2:

    def __init__(self, params: 'HP', initial_embeddings: 'np.ndarray' = None):

        self.params = params
        self.initial_embeddings = initial_embeddings
        self.model = self._build_model_t2()

    def _build_model_t2(self) -> 'Model':

        n_inputs = self.params.n_outputs

        x = keras.layers.Input((n_inputs, ), name='x_sparse', dtype='float32', sparse=True)

        eff_emb = SparseEmbeddingLookup(input_size=n_inputs, embed_size=self.params.embed_size,
                                        initial_embeddings=self.initial_embeddings,
                                        trainable_embeddings=self.params.trainable_embeddings,
                                        name='effective_embedding')(x)

        if self.params.model_version == 0:
            hidden = eff_emb
        elif self.params.model_version == 1:
            hidden = keras.layers.Dense(self.params.embed_size, activation='relu', name='hidden1')(eff_emb)
            hidden = keras.layers.Dense(self.params.embed_size, activation='relu', name='hidden2')(hidden)
            hidden = keras.layers.Dropout(self.params.dropout_rate, name='dropout')(hidden)
        else:
            raise ValueError("not known version of the model")

        logits = keras.layers.Dense(self.params.n_outputs, activation=None, name='logits')(hidden)
        predictions = keras.layers.Activation('sigmoid', name='predictions')(logits)

        model = keras.Model(inputs=[x], outputs=predictions)

        metrics = ['accuracy']
        if self.params.with_f_metrics:
            metrics += [F1Score(num_classes=self.params.n_outputs, average='macro', threshold=0.5, name='macro_f1score')]
            metrics += [F1Score(num_classes=self.params.n_outputs, average='micro', threshold=0.5, name='micro_f1score')]

        optimizer = keras.optimizers.Nadam(lr=self.params.lr, schedule_decay=self.params.lr_decay)

        if self.params.loss == 'binary_crossentropy':
            loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=self.params.label_smoothing)
        elif self.params.loss == 'macro_double_soft_f1':
            loss = macro_double_soft_f1
        else:
            raise AssertionError('Bad loss name. Must be one of [\'binary_crossentropy\', \'macro_double_soft_f1\']')

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)

        return model

    def predict_in_batches(self, scores_test, batch_size=32, verbose=1,
                           ensure_prediction_for_each_sample_flag=True):
        """

        :param scores_test: csr_matrix with scores input to model_t2 [nSamples, nLabels]
        :param batch_size:
        :param verbose:
        :param ensure_prediction_for_each_sample_flag: when this is False, sometimes all outputs from the network is
        below 0.5 threshold, so after rounding there would be no predictions at all for this sample, which is typically
        not desired. When this is True (default), then in such situations the highest score is taken regardless being
        less than 0.5.
        :return: csr_matrix with float predictions of np.float32 type. shape = [nSamples, nLabels]
        """

        if scores_test.dtype is not np.float32:
            scores_test = scores_test.astype(np.float32)

        batches_scores = BatchedData(scores_test, batch_size)

        results_on_batches = []
        n_batches = len(batches_scores)

        for i_batch, pi in pi_range(n_batches, info_frequency=1):
            if verbose > 0 and pi:
                print("\rProgress={:.2f}, elapsed time={:.2f}, ETA={:.2f}".format(
                    pi.progress, pi.elapsed_time, pi.time_remaining()), end='\n')
            scores = batches_scores[i_batch]
            float_predictions = self.model.predict_on_batch(scores)
            predictions = dense_outputs_to_sparse_predictions(float_predictions, ensure_prediction_for_each_sample_flag)
            results_on_batches.append(lil_matrix(predictions))

        return vstack(results_on_batches, format='csr', dtype=np.float32)

    @staticmethod
    def load(path_to_model_config, path_to_weights):
        params = load_hp(path_to_model_config)
        t2 = ModelT2(params=params)
        t2.model.load_weights(path_to_weights)
        return t2


class ModelT3(ModelT2):

    def __init__(self, params: 'HP', css_thresholds: 'np.ndarray', initial_embeddings: 'np.ndarray' = None):

        super().__init__(params, initial_embeddings)
        self.model_t2 = self.model
        # for layer in self.model_t2.layers:
        #     layer.set_weights([0.000001*w for w in layer.get_weights()])
            # layer.trainable = False
        self.css_thresholds = css_thresholds
        self.model = self._build_model_t3()

    def _build_model_t3(self) -> 'Model':
        x = self.model_t2.inputs[0]
        model_t2_logits = self.model_t2.get_layer('logits').output

        logits = SparseResidualAdd(-self.css_thresholds, name='t3_logits')([x, model_t2_logits])

        predictions = keras.layers.Activation('sigmoid', name='predictions')(logits)

        model = keras.Model(inputs=[x], outputs=predictions)

        metrics = ['accuracy']
        if self.params.with_f_metrics:
            metrics += [F1Score(num_classes=self.params.n_outputs, average='macro', threshold=0.5, name='macro_f1score')]
            metrics += [F1Score(num_classes=self.params.n_outputs, average='micro', threshold=0.5, name='micro_f1score')]

        optimizer = keras.optimizers.Nadam(lr=self.params.lr, schedule_decay=self.params.lr_decay)

        if self.params.loss == 'binary_crossentropy':
            loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=self.params.label_smoothing)
        elif self.params.loss == 'macro_double_soft_f1':
            loss = macro_double_soft_f1
        else:
            raise AssertionError('Bad loss name. Must be one of [\'binary_crossentropy\', \'macro_double_soft_f1\']')

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)

        return model


    @staticmethod
    def load(path_to_model_config, path_to_weights):
        params = load_hp(path_to_model_config)
        css_path = correct_absolute_paths(params.css_thresholds_path)
        with open(css_path, 'rb') as f:
            css_thresholds = pickle.load(f)
        t3 = ModelT3(params=params, css_thresholds=css_thresholds)
        t3.model.load_weights(path_to_weights)
        return t3


class ModelFactory:

    @staticmethod
    def create(params):
        if params.version.startswith('t2'):
            m = ModelT2(initial_embeddings=None, params=params)
        elif params.version.startswith('t3'):
            css_path = correct_absolute_paths(params.css_thresholds_path)
            with open(css_path, 'rb') as f:
                css_thresholds = pickle.load(f)
            m = ModelT3(params=params, css_thresholds=css_thresholds, initial_embeddings=None)
        else:
            raise AssertionError('Unsupported thresholding model version.')
        return m

    @staticmethod
    def load(path_to_model_config, path_to_weights):
        params = load_hp(path_to_model_config)
        if params.version.startswith('t2'):
            m = ModelT2.load(path_to_model_config, path_to_weights)
        elif params.version.startswith('t3'):
            m = ModelT3.load(path_to_model_config, path_to_weights)
        else:
            raise AssertionError('Unsupported thresholding model version.')
        return m


def predict_using_thresnet_from_hopt(scores, path_to_model_config, path_to_weights, batch_size=32,
                                     ensure_prediction_for_each_sample_flag=True, verbose=1):
    """

    :param scores: csr_matrix [nSamples, nLabels]
    :param path_to_model_config: dict with model specification
    :param path_to_weights: path to hdf5 files
    :param batch_size:
    :param ensure_prediction_for_each_sample_flag: when this is False, sometimes all outputs from the network is below
    0.5 threshold, so after rounding there would be no predictions at all for this sample, which is typically not
    desired. When this is True (default), then in such situations the highest score is taken regardless being less than
    0.5.
    :param verbose:
    :return: csr_matrix [nSamples, nLabels] - with predictions dtype=np.int32
    """

    m = ModelFactory.load(path_to_model_config, path_to_weights)

    params = load_hp(path_to_model_config)
    if params.scaled_scores:
        scores = scale_scores(scores)

    predictions = m.predict_in_batches(
        scores_test=scores,
        batch_size=batch_size,
        verbose=verbose,
        ensure_prediction_for_each_sample_flag=ensure_prediction_for_each_sample_flag)

    return predictions


def get_probabilities_using_thresnet_from_hopt(scores, path_to_model_config, path_to_weights, batch_size=32, verbose=1):
    """
    :param scores: csr_matrix [nSamples, nLabels]
    :param path_to_model_config: dict with model specification
    :param path_to_weights: path to hdf5 files
    :param batch_size:
    :param verbose:
    :return: csr_matrix [nSamples, nLabels] - with predictions dtype=np.int32
    """

    if scores.dtype is not np.float32:
        scores = scores.astype(np.float32)

    m = ModelFactory.load(path_to_model_config, path_to_weights)

    return m.model.predict(scores, batch_size=batch_size, verbose=verbose)


def get_dataset(scores: 'csr_matrix', labels: 'csr_matrix', params: 'HP', validation: bool = False):

    if params.scaled_scores:
        scores = scale_scores(scores)

    scores_coo = coo_matrix(scores, dtype=np.float32)
    st_scores = tf.sparse.reorder(tf.SparseTensor(
        np.column_stack((scores_coo.row, scores_coo.col)), scores_coo.data, scores_coo.shape))

    labels_coo = coo_matrix(labels)
    st_labels = tf.sparse.reorder(tf.SparseTensor(
        np.column_stack((labels_coo.row, labels_coo.col)), labels_coo.data, labels_coo.shape))

    dataset_input = tf.data.Dataset.from_tensor_slices(st_scores)
    dataset_output = tf.data.Dataset.from_tensor_slices(st_labels)

    if params.sample_weighting_mode is None:
        dataset = tf.data.Dataset.zip((dataset_input, dataset_output)).map(
            lambda ein, eout: (ein, tf.sparse.to_dense(eout)))
    else:
        dataset_sample_weights = tf.data.Dataset.from_tensor_slices(
            calculate_sample_weights_for_multi_label_problem(labels, params.sample_weighting_mode))
        dataset = tf.data.Dataset.zip((dataset_input, dataset_output, dataset_sample_weights)).map(
            lambda ein, eout, esw: (ein, tf.sparse.to_dense(eout), esw))

    if not validation:
        shuffle_buffer_size = min(5000, 10*params.batch_size)
        dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)

    dataset = dataset.batch(params.batch_size)

    return dataset


def train(train_data, eval_data, params):
    # Your training function
    val_data, test_data = eval_data

    # Callbacks

    if params.with_f_metrics and params.model_version != 'nce':
        earlystop = keras.callbacks.EarlyStopping(patience=params.early_stop_patience, monitor='val_macro_f1score', mode='max')
        hopt_callback = HoptCallback(model_prefix='{val_macro_f1score:06.5f}_{epoch:03d}', model_lower_better=False,
                                     model_monitor='val_macro_f1score', save_tf_graphs=False,
                                     test_generators=test_data, workers=8, mode='max')
    else:
        earlystop = keras.callbacks.EarlyStopping(patience=params.early_stop_patience, monitor='val_loss', mode='min')
        hopt_callback = HoptCallback(model_prefix='{val_loss:08.4f}_{epoch:03d}', model_lower_better=True,
                                     model_monitor='val_loss', save_tf_graphs=False,
                                     test_generators=test_data, workers=8)

    callbacks = [earlystop, hopt_callback]

    # Model
    m = ModelFactory.create(params)

    m.model.fit(
        x=train_data,
        validation_data=val_data,
        initial_epoch=0,
        epochs=params.max_epochs,
        shuffle=False,
        callbacks=callbacks
    )


def hopt_optimize_thresnet_phase(out_dir, params, scores_train, labels_train,
                                 scores_validation=None, labels_validation=None,
                                 scores_test=None, labels_test=None):
    """
    Meant to be called from main kadrnn pipeline.
    :param out_dir:
    :param params:
    :param scores_train:
    :param labels_train:
    :param scores_validation:
    :param labels_validation:
    :param scores_test:
    :param labels_test:
    :return:
    """

    if scores_validation is None or labels_validation is None:
        # If val data not given, we need to do train-val splitting here
        val_ratio = 0.1
        val_size = int(val_ratio * scores_train.shape[0])
        val_idxs = np.array(val_size*[True] + (scores_train.shape[0] - val_size)*[False])
        np.random.shuffle(val_idxs)
        train_idxs = ~val_idxs
        scores_validation = scores_train[val_idxs, :]
        labels_validation = labels_train[val_idxs, :]
        scores_train = scores_train[train_idxs, :]
        labels_train = labels_train[train_idxs, :]

    dataset_train = get_dataset(scores_train, labels_train, params)
    dataset_val = get_dataset(scores_validation, labels_validation, params, validation=True)

    if scores_test is not None and labels_test is not None:
        dataset_test = get_dataset(scores_test, labels_test, params, validation=True)
    else:
        dataset_test = None

    # hyperparameter random search
    search(train, params, out_dir, [params.iterations], [params.max_epochs], dataset_train, [dataset_val, dataset_test])


def find_paths_of_best_hyperparams_found_by_hopt(run_dir, mode='max'):
    models_dir = os.path.join(run_dir, 'models')
    hyperparams_dir = os.path.join(run_dir, 'hyperparams')

    models_files = glob(models_dir+'/*')
    values = []
    cfg_names = []
    for fn in models_files:
        f = os.path.basename(fn)
        value, _, data, time = f[:-5].split('_')
        values.append(float(value))
        cfg_names.append(data+'_'+time+'.json')

    best_idx = np.argmax(values) if mode == 'max' else np.argmin(values)

    best_hyperparameters_config_path = os.path.join(hyperparams_dir, cfg_names[int(best_idx)])
    best_model_weights_path = models_files[int(best_idx)]

    return best_hyperparameters_config_path, best_model_weights_path
