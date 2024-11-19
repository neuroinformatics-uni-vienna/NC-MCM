import colorsys
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as smt

from ncmcm.statistical_testing.markov import *


def test_markov_property_test():
    states = [1, 5, 10]
    for s in states:
        sequence = np.random.randint(0,
                                     s,
                                     size=300)
        p, P1 = markov_property_test(sequence,
                                     simulations=300)
        assert 0 <= p <= 1
        assert len(P1) == s


def test_compute_transition_matrix_lag2():
    states = [1, 5, 10]
    for s in states:
        sequence = np.random.randint(0,
                                     s,
                                     size=300)
        P, S, M, N = estimate_joint_probability_matrix_lag2(sequence)

        assert len(S) == s
        assert M == 300
        assert N == s
        assert P.shape == (s, s, s)


def test_stationary_property_test():
    states = [1, 5, 10]
    for s in states:
        sequence = np.random.randint(0,
                                     s,
                                     size=300)
        p1, _ = stationary_property_test(sequence=sequence, simulations=10,
                                         chunks_num=3, test_mode='ks')
        assert 0 <= p1 <= 1
        p1, _ = stationary_property_test(sequence=sequence, simulations='optimal',
                                         chunks_num=3, test_mode='ks')
        assert 0 <= p1 <= 1
        p1, _ = stationary_property_test(sequence=sequence, simulations='optimal',
                                         chunks_num=None, test_mode='ks', verbose=1)
        assert 0 <= p1 <= 1

    states = [2, 3, 4]
    for s in states:
        sequence = np.random.randint(0,
                                     s,
                                     size=500)
        p1, _, p2, _ = stationary_property_test(sequence=sequence, simulations=10, chunks_num=None, test_mode='both')
        assert 0 <= p1 <= 1
        assert 0 <= p2 <= 1
        p1, _ = stationary_property_test(sequence=sequence, simulations=10, chunks_num=None, test_mode='ttest')
        assert 0 <= p1 <= 1
        p1, _ = stationary_property_test(sequence=sequence, simulations=10, chunks_num=None, test_mode='something else')
        assert 0 <= p1 <= 1


def test_simulate_markov_sequence():
    length = 100
    states = 3
    rand_P = np.random.rand(states, states)
    rand_P /= np.sum(rand_P,
                     axis=1,
                     keepdims=True)

    z, P = simulate_markov_sequence(length,
                                    order=10,
                                    N=10,
                                    P=rand_P)

    assert length == len(z)
    assert rand_P.shape == P.shape

    order = 3

    z, P = simulate_markov_sequence(length,
                                    order=order,
                                    N=states)

    assert length == len(z)
    assert len(P) == order
    assert P.shape == (states,) * (order + 1)


def test_make_random_adj_matrices():
    states = 3
    matrices = make_random_adj_matrices(num_matrices=30,
                                        matrix_shape=(states, states),
                                        sparse=False)
    assert len(matrices) == 30
    assert matrices[0].shape == (states, states)

    matrices = make_random_adj_matrices(num_matrices=20,
                                        matrix_shape=(states, states),
                                        sparse=True)
    assert len(matrices) == 20
    assert matrices[0].shape == (states, states)


def test_non_stationary_process():
    states = [1, 5, 10]
    for s in states:
        sequence = non_stationary_process(100,
                                          N=s)
        assert len(sequence) == 100
        assert len(np.unique(sequence)) <= s


def test_non_stationary_process2():
    states = [1, 5, 10]
    for s in states:
        sequence = non_stationary_process2(100,
                                           N=s,
                                           epsilon=0.01*s)
        assert len(sequence) == 100
        assert len(np.unique(sequence)) <= s


def test_simulate_random_sequence():
    states = [1, 5, 10]
    for s in states:
        sequence = simulate_random_sequence(M=100,
                                            N=s)

        assert len(sequence) == 100
        assert len(np.unique(sequence)) <= s
