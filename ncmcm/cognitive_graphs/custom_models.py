from itertools import combinations
import numpy as np
from sklearn import clone


class CustomEnsembleModel:
    """
    This ensemble takes a model and creates binary predictors for each label-combination.
    As a prediction for each instance it gives the most abundant prediction from its sub-models.
    """

    def __init__(self,
                 base_model):
        """
        :param base_model: a model from which the binary classifiers will be built (e.g. Logistic Regression). It needs
        to have the method "fit", "predict" and "predict_proba".
        """
        self.base_model = base_model
        self.combinatorics = []
        self.ensemble_models = []

    def fit(self,
            neuron_traces,
            labels):

        self.ensemble_models = []
        self.combinatorics = list(combinations(np.unique(labels), 2))
        for idx, class_mapping in enumerate(self.combinatorics):
            b_model = clone(self.base_model)
            mapped_labels = np.array([label if label in class_mapping else -1 for label in labels])
            mask = mapped_labels != -1
            # apply mask to the dataset and only use instances of 'A' or 'B' to train
            neuron_traces_filtered = neuron_traces[mask]
            mapped_labels_filtered = mapped_labels[mask]
            b_model.fit(neuron_traces_filtered, mapped_labels_filtered)
            self.ensemble_models.append(b_model)
        return self

    def predict(self,
                neuron_traces):

        results = np.zeros((neuron_traces.shape[0], len(self.combinatorics))).astype(int)
        for idx, b_model in enumerate(self.ensemble_models):
            results[:, idx] = b_model.predict(neuron_traces)
        return [np.bincount(results[row, :]).argmax() for row in range(results.shape[0])]

    def predict_proba(self,
                      neuron_traces):

        y_prob_map = np.zeros((neuron_traces.shape[0], len(self.combinatorics)))
        for idx, model in enumerate(self.ensemble_models):
            prob = model.predict_proba(neuron_traces)[:, 0]
            y_prob_map[:, idx] = prob
        return y_prob_map

    def classify(self, inputs):
        return np.sign(self.predict(inputs))

    def get_params(self, deep=False):
        return {'base_model': self.base_model}
