import numpy as np
import pandas as pd


def per_user_means(scores):    
    return scores.sum(axis=1) / (scores > 0).sum(axis=1)


def per_user_stds(scores):
    means = per_user_means(scores)
    nonzeros = (scores > 0).sum(axis=1)
    sums = scores.sum(axis=1)
    sq_sums = (scores**2).sum(axis=1)
    return np.sqrt((sq_sums + nonzeros*means**2 - 2*means*sums) / nonzeros)


def read_small(data_path=None):
    if data_path is None:
        data_path = 'ml-latest-small/'
    movies = pd.read_csv(data_path + 'movies.csv')
    ratings = pd.read_csv(data_path + 'ratings.csv')
    min_rate = .5
    return movies, ratings, .5


def read_10M(data_path=None):
    if data_path is None:
        data_path = 'ml-10M100K/'
    movies = pd.read_csv(data_path + 'movies.dat', delimiter='::', engine='python')
    ratings = pd.read_csv(data_path + 'ratings.dat', delimiter='::', engine='python')
    min_rate = .5
    return movies, ratings, .5


def read_1M(data_path=None):
    if data_path is None:
        data_path = 'ml-1m/'
    movies = pd.read_csv(data_path + 'movies.dat', delimiter='::', engine='python')
    ratings = pd.read_csv(data_path + 'ratings.dat', delimiter='::', engine='python') 
    min_rate = 1.
    return movies, ratings, 1.