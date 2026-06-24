import sys
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from ncmcm.cognitive_graphs.calculations import *
from ncmcm.cognitive_graphs.custom_models import *
from ncmcm.cognitive_graphs.helpers import *
from pyvis.network import Network
from ncmcm.cognitive_graphs.cognitive_graphs import behavioral_state_diagram
from ncmcm.cognitive_graphs.cognitive_graphs import cluster_neural_activity
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.show = lambda *args, **kwargs: None

def test_behavioral_state_diagram():
    cognitive_states = [np.random.randint(0, 2) for _ in range(100)]
    actions = ['sit', 'stand', 'walk', 'run']
    behaviors = [np.random.choice(actions) for _ in range(100)]

    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      interactive=None,
                                      test_run=True,
                                      bins=10)
    print(result)
    assert result is True
    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      behaviors=actions,
                                      interactive=None,
                                      test_run=True,
                                      bins=10)
    assert result is True
    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      interactive=None,
                                      weights_hist=True,
                                      test_run=True,
                                      bins=10)
    assert result is True
    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      behaviors=actions,
                                      interactive=None,
                                      weights_hist=True,
                                      test_run=True,
                                      bins=10)
    assert result is True
    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      interactive=None,
                                      adj_matrix=True,
                                      test_run=True,
                                      bins=10)
    assert result is True
    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      behaviors=actions,
                                      interactive=None,
                                      adj_matrix=True,
                                      test_run=True,
                                      bins=10)
    assert result is True


@patch('builtins.input', return_value='plot_test')
def test_interactive(return_value):
    cognitive_states = [np.random.randint(0, 2) for _ in range(100)]
    actions = ['sit', 'stand', 'walk', 'run']
    behaviors = [np.random.choice(actions) for _ in range(100)]

    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      behaviors=actions,
                                      interactive="plot_test",
                                      adj_matrix=True,
                                      test_run=True,
                                      bins=10)
    assert result is True

    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      interactive="plot_test",
                                      adj_matrix=False,
                                      test_run=True,
                                      bins=10)
    assert result is True


def test_cluster_neural_activity():
    n = np.random.uniform(low=0.0, high=3.0, size=(15, 100))
    actions = ['sit', 'stand', 'walk', 'run']
    behaviors = [np.random.choice(actions) for _ in range(100)]

    res = cluster_neural_activity(n, behaviors,
                                  n_clusters=3,
                                  nrep=2,
                                  test_stationary_property=True,
                                  model=RandomForestClassifier(n_estimators=3,
                                                               random_state=42))

    assert len(res) == 2
    assert len(res[0]) == 4
    assert len(res[0][0]) == 100
    assert len(np.unique(res[0][0])) == 3
    assert 0 <= res[0][1] <= 1
    assert 0 <= res[0][2] <= 1
    assert 0 <= res[0][3] <= 1

    res = cluster_neural_activity(n, behaviors,
                                  n_clusters=4,
                                  nrep=3,
                                  test_stationary_property=False,
                                  model=LogisticRegression())

    assert len(res) == 3
    assert len(res[0]) == 2
    assert len(res[0][0]) == 100
    assert len(np.unique(res[0][0])) == 4
    assert 0 <= res[0][1] <= 1
