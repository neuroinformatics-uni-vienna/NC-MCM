import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from ncmcm.visualisers.latent_space import *


def test_comparison():
    logreg = LogisticRegression()
    y = np.random.rand(100, 3)
    b = np.random.randint(low=0, high=4, size=100, dtype=int)
    b_names = ['sit', 'stand', 'walk', 'run']

    lsv = LatentSpaceVisualiser(y, b, b_names=b_names, legend=True)
    res1 = lsv.comparison_model(model=logreg, show_fig=False)
    lsv = LatentSpaceVisualiser(y, b, b_names=b_names, legend=False)
    res2 = lsv.comparison_model(model=logreg, show_fig=False)

    assert res1 == True
    assert res2 == True


def test_y_original():
    logreg = RandomForestClassifier()
    y = np.random.rand(500, 3)
    y_add = np.random.rand(500, 20)
    b = np.random.randint(low=0, high=4, size=500, dtype=int)
    b_names = ['sit', 'stand', 'walk', 'run']
    y_true = np.concatenate((y_add, y), axis=1)

    lsv = LatentSpaceVisualiser(y, b, b_names=b_names, legend=False)
    res1 = lsv.comparison_model(model=logreg, original_data=y_true, show_fig=False)
    lsv = LatentSpaceVisualiser(y, b, b_names=b_names, legend=True)
    res2 = lsv.comparison_model(model=logreg, original_data=y_true, show_fig=False)
    lsv = LatentSpaceVisualiser(y, b, b_names=b_names, legend=False)
    res3 = lsv.comparison_model(model=logreg, original_data=y_true[:-1, :], show_fig=False)

    assert res1 == True
    assert res2 == True
    assert res3 == False

def test_movie_maker():
    y = np.random.rand(50, 3)
    b = np.random.randint(low=0, high=4, size=50, dtype=int)
    b_names = ['sit', 'stand', 'walk', 'run']

    lsv = LatentSpaceVisualiser(y, b, b_names=b_names, legend=True)
    res1 = lsv.make_movie(show_fig=False)
    res2 = lsv.make_movie(show_fig=False, fps=1000, initial_alpha=0.01)
    res3 = lsv.make_movie(show_fig=False, fade=10, alpha=0.45)
    res4 = lsv.make_movie(show_fig=False, fade=200, alpha=1, initial_alpha=0.01)

    assert res1 == True
    assert res2 == True
    assert res3 == True
    assert res4 == True
