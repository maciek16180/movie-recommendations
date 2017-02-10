import numpy as np
from collaborative_filtering import scores_cf_uu, calculate_similarities_cf_ii, scores_cf_ii
from slope_one import calculate_similarities_SO, scores_SO_basic, scores_SO_weighted, \
                      calculate_similarities_SO_bipolar, scores_SO_bipolar
from baselines import baseline_predictors


def cross_valid(scores, folds=10, method='baseline', min_rate=None, seed=1234, verbose=True, cf_n=None, 
                uucf_metric='cosine', uucf_threshold=50, iicf_sub_base=False):
    assert (method not in ['uucf', 'iicf']) or cf_n is not None
    assert min_rate is not None
    np.random.seed(seed)
    order = np.random.permutation(scores.shape[0])
    per_fold = np.ceil(scores.shape[0] / float(folds)).astype(np.int64)
    
    results = []
    
    idx = 0
    while idx < scores.shape[0]:
        if verbose:
            print 'Started fold {} out of {}'.format(idx / per_fold + 1, folds)
        
        test = scores[order[idx : idx + per_fold]]
        train = scores[np.delete(order, np.arange(idx, idx + per_fold))]
        query, target = query_target_split(test, set_seed=False)
        
        if method == 'baseline':
            S = baseline_predictors(train, query)
        elif method == 'uucf':
            S = scores_cf_uu(train, query, target > 0, cf_n, metric=uucf_metric, threshold=uucf_threshold, min_rate=min_rate)[0]
        elif method == 'iicf':
            S = scores_cf_ii(train, query, target > 0, cf_n, 
                             calculate_similarities_cf_ii(train, subtract_baselines=iicf_sub_base), min_rate=min_rate)
        elif method == 'sobasic':
            S = scores_SO_basic(train, query, target > 0, calculate_similarities_SO(train), min_rate=min_rate)
        elif method == 'soweighted':
            S = scores_SO_weighted(train, query, target > 0, calculate_similarities_SO(train), min_rate=min_rate)
        elif method == 'sobipolar':
            S = scores_SO_bipolar(train, query, target > 0, *calculate_similarities_SO_bipolar(train), min_rate=min_rate)
        
        results.append((evaluate_ratings(S, target, method='mae'),
                        evaluate_ratings(S, target, method='rmse')))
        idx += per_fold
        
    return results


def analyze_cross_valid_res(cv_result):
    return (np.mean(cv_result, axis=0), np.std(cv_result, axis=0))


def train_test_split(scores, seed=1234, set_seed=True):
    N = scores.shape[0]
    test_size = N / 10
    if set_seed:
        np.random.seed(seed)
    test_idx = np.random.choice(N, test_size, replace=False)
    test_set = scores[test_idx]
    train_set = np.delete(scores, test_idx, axis=0)
    return train_set, test_set


def query_target_split(test_set, seed=1234, set_seed=True):
    query_set = test_set.copy()
    target_set = test_set.copy()
    if set_seed:
        np.random.seed(seed)
    for u in xrange(test_set.shape[0]):
        rated = np.where(test_set[u] > 0)[0]
        target = np.random.choice(rated, rated.size / 2, replace=False)
        query_set[u,target] = 0
    target_set[np.where(query_set > 0)] = 0
    return query_set, target_set


def evaluate_ratings(scores, target, method='mae', min_rate=.5):
    mask = target > 0
    abs_dists = np.abs(scores * mask - target)
    if method == 'mae':
        return abs_dists.sum() / mask.sum()
    elif method == 'nmae':
        return abs_dists.sum() / mask.sum() / (5. - min_rate)
    elif method == 'rmse':
        return np.sqrt((abs_dists**2).sum() / mask.sum())
    else:
        raise Exception('Invalid argument value: method')