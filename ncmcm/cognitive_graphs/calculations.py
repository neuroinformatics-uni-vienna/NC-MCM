import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from ncmcm.cognitive_graphs.custom_models import CustomEnsembleModel


def adj_matrix_ncmcm(C, B):
    """
        Calculate the adjacency matrix and list of cognitive-behavioral states.

        Parameters:
        - B: behavioral timeseries data
        - C: cognitive state timeseries data

        Returns:
        - C_B_states: List of all cognitive-behavioral states (coded as: CCBB).
        - T: Adjacency matrix for the cognitive-behavioral states.
    """

    b = np.unique(B)
    c = np.unique(C)
    T = np.zeros((len(c) * len(b), len(c) * len(b)))
    C_B_states = np.asarray([str(cs + 1) + '-' + str(bs) for cs in c for bs in b])

    for m in range(len(B) - 1):
        cur_sample = m
        next_sample = m + 1
        cur_state = np.where(str(C[cur_sample] + 1) + '-' + str(B[cur_sample]) == C_B_states)[0][0]
        next_state = np.where(str(C[next_sample] + 1) + '-' + str(B[next_sample]) == C_B_states)[0][0]
        T[next_state, cur_state] += 1

    # normalize T
    T = T / (len(B) - 1)
    T = T.T

    return T, C_B_states


def fit_model(neuron_traces,
              B,
              base_model,
              ensemble=True,
              cv_folds=0):
    """
    Allows to fit a model which is used to predict behaviors from the neuron traces (accuracy is printed). Its
    probabilities are used for an eventual clustering.

    Parameters:

        neuron_traces (np.ndarray, list): Neuronal activity timeseries (shape = (neurons, activity-timeseries))

        B (np.ndarray, list): Behavioral timeseries data

        base_model: Classifier to project into probability space (needs .predict_proba method)
        ensemble: A boolean indicating if the CustomEnsembleModel should be created.
        cv_folds: If cross validation should be applied, this signals the amount of cross-folds.

    Returns:

        Boolean success indicator
    """
    B = np.asarray(B)
    neuron_traces = np.asarray(neuron_traces)

    if not hasattr(base_model, 'fit'):
        print('Model has no method \'fit\'.')
        return False

    if ensemble:
        pred_model = CustomEnsembleModel(base_model)
    else:
        pred_model = base_model

    if cv_folds and type(cv_folds) is int:
        cv_scores = cross_val_score(pred_model, neuron_traces.T, B, cv=cv_folds,
                                    scoring='accuracy')  # 5-fold cross-validation
        print(f'Mean cross-validation results for {cv_folds} folds:\n'
              f'\tMean: {np.mean(cv_scores)}\n'
              f'The full scores can be accessed by \'self.cv_scores\'')

    pred_model.fit(neuron_traces.T, B)
    B_pred = np.asarray(pred_model.predict(neuron_traces.T))
    print("Accuracy for full training data:", accuracy_score(B, B_pred))
    yp_map = pred_model.predict_proba(neuron_traces.T)

    return yp_map, pred_model
