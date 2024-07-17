import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from ncmcm.visualisers.latent_space import *


def test_comparison():
    logreg = LogisticRegression()
    y = np.random.rand(500, 3)
    b = np.zeros(500, dtype=int)
    b_names = ['sit', 'stand', 'walk', 'run']

    for i in range(1, 500):
        y[i] += np.random.normal(scale=0.05, size=3)
        y[i] = np.asarray(np.clip(y[i], 0, 1))
    for i in range(500):
        x, f, z = y[i]
        if x + f + z < 1.5:
            b[i] = 0
        elif x + f + z < 2:
            b[i] = 1
        elif x + f + z < 2.5:
            b[i] = 2
        else:
            b[i] = 3

    lsv = LatentSpaceVisualiser(y, b, b_names=b_names, legend=True)
    res1 = lsv.comparison_model(model=logreg, show_fig=False)
    lsv = LatentSpaceVisualiser(y, b, b_names=b_names, legend=False)
    res2 = lsv.comparison_model(model=logreg, show_fig=False)

    assert res1 == True
    assert res2 == True


def test_y_original():
    logreg = RandomForestClassifier()
    y = np.random.rand(500, 3)
    b = np.zeros(500, dtype=int)
    b_names = ['sit', 'stand', 'walk', 'run']

    for i in range(1, 500):
        y[i] += np.random.normal(scale=0.05, size=3)
        y[i] = np.asarray(np.clip(y[i], 0, 1))
    for i in range(500):
        x, f, z = y[i]
        if x + f + z < 1.5:
            b[i] = 0
        elif x + f + z < 2:
            b[i] = 1
        elif x + f + z < 2.5:
            b[i] = 2
        else:
            b[i] = 3
    y_true = np.concatenate((np.random.rand(500, 20), y), axis=1)

    lsv = LatentSpaceVisualiser(y, b, b_names=b_names, legend=False)
    res1 = lsv.comparison_model(model=logreg, y_original=y_true, show_fig=False)
    lsv = LatentSpaceVisualiser(y, b, b_names=b_names, legend=True)
    res2 = lsv.comparison_model(model=logreg, y_original=y_true, show_fig=False)
    lsv = LatentSpaceVisualiser(y, b, b_names=b_names, legend=False)
    res3 = lsv.comparison_model(model=logreg, y_original=y_true[:-1, :], show_fig=False)

    assert res1 == True
    assert res2 == True
    assert res3 == False