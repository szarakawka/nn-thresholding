from lib.hopt import Parameters, randoms


class HP(Parameters):

    # Model
    embed_size = randoms.Int(64, 256)
    # hidden_size = randoms.Int(32, 64)
    # dropout_rate = randoms.Float(0.2, 0.8)
    trainable_embeddings = True
    max_num_of_scores_per_sample = 10
    model_version = 0           # randoms.IntChoice([0, 1])    # or 'nce'
    with_f_metrics = True

    # Training
    lr = randoms.FloatExp(base=10, pow_a=-4, pow_b=-2)
    lr_decay = randoms.Float(0.0015, 0.0055)
    early_stop_patience = 25
    batch_size = 64
    sample_weighting_mode = 'max'    # [None, 'max', 'mean', 'sum']
    neg_samples_per_batch = 3*batch_size

    # Testing

    # HP search
    iterations = 3
    max_epochs = 1000

    # Data
    test_folds = [0]
    train_aug = None    # augmentation subdir name; None selects the directory itself instead of any of the subdirs
    val_aug = None      # augmentation subdir name; None selects the directory itself instead of any of the subdirs
    train_epoch_ratio = 1.0     # Reduces size of training set (results in more frequent validation)
    val_epoch_ratio = 1.0       # Reduces size of validation set (1.0 = full set)

    version = ''


def get_hp(version):
    params = HP

    params.version = version
    if version == 't2max':
        params.sample_weighting_mode = 'max'

    if version == 't2none':
        params.sample_weighting_mode = None

    return params

