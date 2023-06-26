from tensorflow.python.keras import backend as K
import tensorflow as tf


# the metrics here are not suitable for use in mini-batches !! (they are not stateful!!!)

@tf.function
def micro_precision(y_true, y_pred):
    """Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    mi_precision = true_positives / (predicted_positives + K.epsilon())
    return mi_precision


@tf.function
def _precisions_per_label(y_true, y_pred):
    """Calculates precisions per each label
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-2)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=-2)
    precisions = true_positives / (predicted_positives + K.epsilon())
    return precisions


@tf.function
def macro_precision(y_true, y_pred):
    """Calculates the macro precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    precisions = _precisions_per_label(y_true, y_pred)
    ma_precision = K.mean(precisions)
    return ma_precision


@tf.function
def micro_recall(y_true, y_pred):
    """Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    mi_recall = true_positives / (possible_positives + K.epsilon())
    return mi_recall


@tf.function
def _recalls_per_label(y_true, y_pred):
    """Calculates recalls per each label
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-2)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=-2)
    recalls = true_positives / (possible_positives + K.epsilon())
    return recalls


@tf.function
def macro_recall(y_true, y_pred):
    """Calculates the macro recall, a metric for multi-label classification of
    how many selected items are relevant.
    """
    recalls = _recalls_per_label(y_true, y_pred)
    ma_recall = K.mean(recalls)
    return ma_recall


@tf.function
def micro_f1score(y_true, y_pred):
    """Calculates the F score, the weighted harmonic mean of precision and recall.
    """

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0.

    p = micro_precision(y_true, y_pred)
    r = micro_recall(y_true, y_pred)
    mi_f1score = 2. * (p * r) / (p + r + K.epsilon())
    return mi_f1score


@tf.function
def _f1scores_per_label(y_true, y_pred):
    precisions = _precisions_per_label(y_true, y_pred)
    recalls = _recalls_per_label(y_true, y_pred)
    f1scores = 2. * (precisions * recalls) / (precisions + recalls + K.epsilon())
    return f1scores


@tf.function
def macro_f1score(y_true, y_pred):
    """Calculates the F score, the weighted harmonic mean of precision and recall.
    """

    f1scores = _f1scores_per_label(y_true, y_pred)
    ma_f1score = K.mean(f1scores)
    return ma_f1score
