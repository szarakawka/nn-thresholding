import numpy as np


class S3G3DataGenerator:
    def __init__(self):
        self.n_labels = 6
        self.lab_id_to_name = ['S1G3', 'S2G1', 'S3G2', 'S1G1', 'S2G2', 'S3G3']
        self.labels = {'S1G3': 0, 'S2G1': 1, 'S3G2': 2, 'S1G1': 3, 'S2G2': 4, 'S3G3': 5}
        self.label_groups = {'G1': [1, 3], 'G2': [2, 4], 'G3': [0, 5]}

        self.in_group_label_probability = 0.6
        self.out_of_group_label_probability = 0.1


class ToyDataSet1:
    def __init__(self, noise_level_prob=0.1, crucial_prob=0.95, crucial_absense_prob=0.02, fluctuations=0.1):
        self.n_features = 10
        self.n_labels = 6
        self.eye = np.eye(self.n_labels, dtype=np.float32)
        self.nl = noise_level_prob  # noise level
        self.crucial_prob = crucial_prob
        self.crucial_absense_prob = crucial_absense_prob
        self.fluctuations = fluctuations
        self.features_distribution_in_classes = self._make_class_distributions(crucial_prob, crucial_absense_prob, noise_level_prob)

    @staticmethod
    def _make_class_distributions(high_prob, low_prob, whatever_prob):
        hp = high_prob
        lp = low_prob
        we = whatever_prob
        return {0: [hp, we, we, we, we, we, we, we, we, we, we, we, we, hp],
                1: [we, hp, hp, we, we, we, we, we, we, we, we, we, we, hp],
                2: [we, hp, lp, we, we, we, we, we, we, we, we, we, we, hp],
                3: [we, we, we, we, hp, we, we, we, we, we, we, we, we, hp],
                4: [we, we, we, we, we, we, we, we, we, we, hp, hp, 0.5, 0.5], # the last feature here is negation of the penultimate one
                5: [we, we, we, we, we, we, we, we, we, we, hp, hp, hp, hp]}

    def generate_sample(self, which_class):

        sample = np.random.rand(self.n_features) < self.features_distribution_in_classes[which_class]
        sample = sample.astype(np.float32)

        # in class 4 the last feature is negation of penultimate one
        if which_class == 5:
            sample[-1] = 1. - sample[-2]

        if self.fluctuations > 0:
            sample *= (1. + self.fluctuations * np.random.randn(self.n_features))
            sample[sample < 0.] = 0.

        return sample, self.eye[which_class, :]

    def make_data_set(self, n_samples):

        x = np.zeros((n_samples, self.n_features), dtype=np.float32)
        t = np.zeros((n_samples, self.n_labels))

        which_label = [np.random.randint(self.n_labels) for _ in range(n_samples)]

        for n, l in enumerate(which_label):
            x[n, :], t[n, :] = self.generate_sample(l)

        return x, t

    # TODO
    def make_imbalanced_data_set(self, n_samples):
        raise NotImplementedError("Implement this function!")


class ToyDataSetCurseOfDimensionality2Labels:
    def __init__(self, n_features, labels_prob=(0.5, 0.5), noise_level_prob=0.5, crucial_prob=0.95,
                 crucial_absense_prob=0.05, fluctuations=0.1):
        assert n_features >= 2
        self.n_features = n_features
        self.labels_prob = labels_prob
        self.n_labels = len(labels_prob)
        self.eye = np.eye(self.n_labels, dtype=np.float32)
        self.nl = noise_level_prob  # noise level
        self.crucial_prob = crucial_prob
        self.crucial_absense_prob = crucial_absense_prob
        self.fluctuations = fluctuations
        self.features_distribution_in_classes = self._make_class_distributions(crucial_prob, crucial_absense_prob,
                                                                               noise_level_prob)

    def _make_class_distributions(self, high_prob, low_prob, noise_prob):
        hp = high_prob
        lp = low_prob
        tail = [noise_prob for _ in range(self.n_features-2)]
        return {0: [hp, lp] + tail,
                1: [lp, hp] + tail}

    def generate_sample(self, which_class):

        sample = np.random.rand(self.n_features) < self.features_distribution_in_classes[which_class]
        sample = sample.astype(np.float32)

        if self.fluctuations > 0:
            sample *= (1. + self.fluctuations * np.random.randn(self.n_features))
            sample[sample < 0.] = 0.

        return sample, self.eye[which_class, :]

    def make_data_set(self, n_samples):

        x = np.zeros((n_samples, self.n_features), dtype=np.float32)
        t = np.zeros((n_samples, self.n_labels))

        which_label = np.random.choice([0, 1], n_samples, p=list(self.labels_prob))

        for n, l in enumerate(which_label):
            x[n, :], t[n, :] = self.generate_sample(l)

        return x, t


# # TODO
# class ToyDataSetXor2Labels:
#     raise NotImplementedError
