from lib.hopt import Parameters, randoms
import json


class HP(Parameters):

    # Model
    n_outputs = None                    # set later
    embed_size = randoms.Int(100, 200)
    # embed_size = randoms.Int(30, 50)
    dropout_rate = randoms.Float(0.1, 0.3)
    trainable_embeddings = True
    model_version = None     # now get_hp(version) decides        randoms.IntChoice([0, 1])    # or 'nce'
    with_f_metrics = True
    css_thresholds_path = None          # used for ModelT3

    # Training
    # lr = randoms.FloatExp(base=10, pow_a=-4, pow_b=-3)
    lr = randoms.FloatExp(base=10, pow_a=-3, pow_b=-2)
    lr_decay = randoms.Float(0.0015, 0.0055)
    early_stop_patience = 20
    batch_size = 256
    sample_weighting_mode = None    # now get_hp(version) decides           # [None, 'max', 'mean', 'sum']
    # neg_samples_per_batch = 3*batch_size
    loss = 'binary_crossentropy'        # [binary_crossentropy, macro_double_soft_f1]
    label_smoothing = 0

    # Testing

    # HP search
    iterations = 3    # 5
    max_epochs = 1000   # 1000

    # Data
    scaled_scores = True
    # test_folds = [0]
    # train_aug = None    # augmentation subdir name; None selects the directory itself instead of any of the subdirs
    # val_aug = None      # augmentation subdir name; None selects the directory itself instead of any of the subdirs
    # train_epoch_ratio = 1.0     # Reduces size of training set (results in more frequent validation)
    # val_epoch_ratio = 1.0       # Reduces size of validation set (1.0 = full set)

    version = ''


def load_hp(json_path):
    with open(json_path, 'r') as f:
        params_dict = json.load(f)
    hp = HP()
    for k, v in params_dict.items():
        if k in dir(hp):
            setattr(hp, k, v)
    return hp


def get_hp(version, **kwargs):
    params = HP

    params.version = version
    # if version == 't2max':
    #     params.sample_weighting_mode = 'max'
    #     params.model_version = 0
    # elif version == 't2none':
    #     params.sample_weighting_mode = None
    #     params.model_version = 0
    # elif version == 't2max_v1':
    #     params.sample_weighting_mode = 'max'
    #     params.model_version = 1
    # elif version == 't2none_v1_sf1':
    #     params.sample_weighting_mode = None
    #     params.model_version = 1
    #     params.loss = 'macro_double_soft_f1'
    if version.startswith('t2'):
        params.version = version
        params.sample_weighting_mode = None
        params.model_version = 1
    elif version.startswith('t3'):
        params.version = version
        params.sample_weighting_mode = None
        params.model_version = 1
        params.css_thresholds_path = kwargs['css_thresholds_path']
    else:
        raise AssertionError('bad version')

    if version.endswith('ls'):
        params.label_smoothing = 0.01

    return params
