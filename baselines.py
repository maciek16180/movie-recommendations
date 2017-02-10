import numpy as np


def user_baselines(train, query=None, mean_score=None):
    if query is None:
        query = train
    if mean_score is None:
        mean_score = train.sum() / np.count_nonzero(train)
    baselines = query.sum(axis=1) / (query > 0).sum(axis=1) - mean_score
    return np.nan_to_num(baselines)


def item_baselines(train, usr_baselines=None, mean_score=None):
    if mean_score is None:
        mean_score = train.sum() / np.count_nonzero(train)
    if usr_baselines is None:
        usr_baselines = user_baselines(train, mean_score=mean_score)
    nonzeros = (train > 0).sum(axis=0)
    usr_baselines = usr_baselines[:, np.newaxis] * (train > 0)
    preds = (train.sum(axis=0) - usr_baselines.sum(axis=0)) / nonzeros - mean_score
    return np.nan_to_num(preds)


# ta funkcja zwraca szacowania nawet istniejacych ocen

def baseline_predictors(train, query=None, mean_score=None):
    if query is None:
        query = train
    if mean_score is None:
        mean_score = train.sum() / np.count_nonzero(train)
    usr_baselines = user_baselines(train, query, mean_score)
    itm_baselines = item_baselines(train, mean_score=mean_score)
    return mean_score + usr_baselines[:, np.newaxis] + itm_baselines