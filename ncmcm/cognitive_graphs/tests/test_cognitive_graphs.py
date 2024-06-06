from sklearn.ensemble import RandomForestClassifier

from ncmcm.cognitive_graphs.cognitive_graphs import *
from unittest.mock import patch


def test_behavioral_state_diagram():
    cognitive_states = [np.random.randint(0, 2) for _ in range(100)]
    actions = ['sit', 'stand', 'walk', 'run']
    behaviors = [np.random.choice(actions) for _ in range(100)]

    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      interactive=False,
                                      test=True,
                                      bins=10)
    assert result is True
    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      behaviors=actions,
                                      interactive=False,
                                      test=True,
                                      bins=10)
    assert result is True
    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      interactive=False,
                                      weights_hist=True,
                                      test=True,
                                      bins=10)
    assert result is True
    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      behaviors=actions,
                                      interactive=False,
                                      weights_hist=True,
                                      test=True,
                                      bins=10)
    assert result is True
    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      interactive=False,
                                      adj_matrix=True,
                                      test=True,
                                      bins=10)
    assert result is True
    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      behaviors=actions,
                                      interactive=False,
                                      adj_matrix=True,
                                      test=True,
                                      bins=10)
    assert result is True


@patch('builtins.input', return_value='plot_test')
def test_interactive(return_value):
    cognitive_states = [np.random.randint(0, 2) for _ in range(100)]
    actions = ['sit', 'stand', 'walk', 'run']
    behaviors = [np.random.choice(actions) for _ in range(100)]

    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      behaviors=actions,
                                      interactive=True,
                                      adj_matrix=True,
                                      test=True,
                                      bins=10)
    assert result is True

    result = behavioral_state_diagram(cognitive_states, behaviors,
                                      interactive=True,
                                      adj_matrix=False,
                                      test=True,
                                      bins=10)
    assert result is True


def test_cluster_neural_activity():
    n = np.random.uniform(low=0.0, high=3.0, size=(15, 1000))
    actions = ['sit', 'stand', 'walk', 'run']
    behaviors = [np.random.choice(actions) for _ in range(1000)]

    res = cluster_neural_activity(n, behaviors,
                                  n_clusters=3,
                                  nrep=2,
                                  stationary=True,
                                  model=RandomForestClassifier(n_estimators=3,
                                                               random_state=42))

    assert len(res) == 2
    assert len(res[0]) == 3
    assert len(res[0][0]) == 1000
    assert len(np.unique(res[0][0])) == 3
    assert 0 <= res[0][1] <= 1
    assert 0 <= res[0][2] <= 1

    res = cluster_neural_activity(n, behaviors,
                                  n_clusters=4,
                                  nrep=3,
                                  stationary=False,
                                  model=LogisticRegression())

    assert len(res) == 3
    assert len(res[0]) == 2
    assert len(res[0][0]) == 1000
    assert len(np.unique(res[0][0])) == 4
    assert 0 <= res[0][1] <= 1
