import numpy as np
from scipy.spatial.distance import cdist
from utils import per_user_means, per_user_stds
from baselines import baseline_predictors


'''
User-User CF
'''

def scores_cf_uu(train, query, target_inds, n, subtract_mean=True, div_by_std=True, metric='pearson',
                 threshold=50, min_rate=None):
    assert not (div_by_std and not subtract_mean)
    
    ratings = []
    num_ratings = []
    
    train_rated = train > 0
    train_means = per_user_means(train)
    train_stds = per_user_stds(train)
    
    train_centered = (train - train_means[:, np.newaxis]) * train_rated
    
    query_means = per_user_means(query)
    query_stds = per_user_stds(query)
    
    baselines = baseline_predictors(train, query)
    
    query_rated = query > 0
    query_centered = (query - query_means[:, np.newaxis]) * query_rated
    
    if metric == 'cosine':
        dists = cdist(query_centered, train_centered, metric='cosine')
    elif metric == 'pearson':
        numerator = query_centered.dot(train_centered.T)
        denom1 = (query_centered**2).dot(train_rated.T)
        denom2 = query_rated.dot((train_centered**2).T)
        dists = (numerator / np.sqrt(denom1 * denom2))
        if threshold > 0:
            common_ratings = query_rated.dot(train_rated.T.astype(np.float64))
            multiplier = common_ratings / threshold
            multiplier[multiplier > 1] = 1
            dists *= multiplier
        dists = 1 - dists
    
    for u_idx in xrange(query.shape[0]):

        user_ratings = query[u_idx]
        user_rated = user_ratings > 0
        user_mean = query_means[u_idx]
        user_std = query_stds[u_idx]
        user_baseline = baselines[u_idx]
        
        targets = np.where(target_inds[u_idx])[0]

        user_dists = dists[u_idx][np.newaxis]

        dists_zero_idx = np.where((user_dists == 0).T * train_rated[:, targets])

        per_item_dists = user_dists.T * train_rated[:, targets]
        per_item_dists[per_item_dists == 0] = np.nan
        per_item_dists[dists_zero_idx] = 0
        
        if per_item_dists.shape[0] > n:
            order = per_item_dists.argpartition(axis=0, kth=n-1)[:n]
        else:
            order = np.indices((per_item_dists.shape[0], targets.size))[0]
        
        weights = 1 - per_item_dists[order, np.arange(targets.size)]

        ratings_gathered = (~np.isnan(weights)).sum(axis=0)

        weights = np.nan_to_num(weights)
        
        scores = train[:, targets][order, np.arange(targets.size)]
        if subtract_mean:
            multiplier = train_stds[order] if div_by_std else 1
            scores = (scores - train_means[order]) / multiplier

        scores = (scores * weights).sum(axis=0) / (np.abs(weights)).sum(axis=0)
        
        estimated_ratings = np.zeros(query.shape[1])
        estimated_ratings[targets] = scores
        
        if subtract_mean:
            multiplier = user_std if div_by_std else 1.        
            estimated_ratings = user_mean + multiplier * estimated_ratings
            
        estimated_ratings[user_rated] = user_ratings[user_rated]

        # dla ocen nan dajemy baseline predictor

        estimated_ratings[np.isnan(estimated_ratings)] = user_baseline[np.isnan(estimated_ratings)]
        if min_rate:
            estimated_ratings[estimated_ratings > 5] = 5
            estimated_ratings[estimated_ratings < min_rate] = min_rate
        
        ratings.append(estimated_ratings[np.newaxis])
        
        user_num_ratings = np.zeros(query.shape[1])
        user_num_ratings[targets] = ratings_gathered
        num_ratings.append(user_num_ratings[np.newaxis])
        
    ratings = np.vstack(ratings)
    num_ratings = np.vstack(num_ratings)
        
    return ratings, num_ratings


def recommend_cf_uu(userId, n, scores, min_n=0, subtract_mean=True, div_by_std=True, metric='pearson'):
    
    estimated_ratings, ratings_gathered = scores_cf_uu(scores, scores[userId-1][np.newaxis], 
                                                       [np.ones(scores.shape[1])], n, subtract_mean, 
                                                       div_by_std, metric)
    estimated_ratings, ratings_gathered = estimated_ratings[0], ratings_gathered[0]
    estimated_ratings[scores[userId-1] > 0] = np.nan
    estimated_ratings[ratings_gathered < min_n] = np.nan
    
    movie_order = (-estimated_ratings).argsort()
    
    return movie_order, estimated_ratings[movie_order], ratings_gathered[movie_order]


'''
Item-Item CF
'''

def calculate_similarities_cf_ii(scores, nan_to_1=True, subtract_baselines=False):
    baselines = baseline_predictors(scores)
    scores_local = scores
    if subtract_baselines:
        scores_local = scores.copy()
        scores_local = (scores_local - baselines) * (scores_local > 0)
    dists = cdist(scores_local.T, scores_local.T, metric='cosine')
    if nan_to_1:
        dists[np.isnan(dists)] = 1
    return dists


def scores_cf_ii(train, query, target_inds, n, dists=None, min_rate=None):
    if dists is None:
        raise NotImplementedError('distances have to be precomputed')
    
    ratings = []
    
    baselines = baseline_predictors(train, query)
    
    for u_idx in xrange(query.shape[0]):

        user_ratings = query[u_idx]
        user_rated = user_ratings > 0
        user_baseline = baselines[u_idx]
        
        targets = np.where(target_inds[u_idx])[0]        

        user_dists = dists[targets][:, user_rated]
        if user_dists.shape[1] > n:
            order = user_dists.argpartition(axis=1, kth=n-1)[:, :n]
        else:
            order = np.indices((targets.size, user_dists.shape[1]))[1]
        weights = 1 - user_dists[np.arange(targets.size)[:, np.newaxis], order]

        scores = user_ratings[user_rated][order] - user_baseline[targets, np.newaxis]
        scores = (scores * weights).sum(axis=1) / (np.abs(weights)).sum(axis=1) + user_baseline[targets]
            
        estimated_ratings = np.zeros(query.shape[1])
        estimated_ratings[targets] = scores
        estimated_ratings[user_rated] = user_ratings[user_rated]
        
        estimated_ratings[np.isnan(estimated_ratings)] = user_baseline[np.isnan(estimated_ratings)]
        if min_rate:
            estimated_ratings[estimated_ratings > 5] = 5
            estimated_ratings[estimated_ratings < min_rate] = min_rate
        
        ratings.append(estimated_ratings[np.newaxis])

    return np.vstack(ratings)


def recommend_cf_ii(userId, n, scores, dists=None):
    if dists is None:
        raise NotImplementedError('distances have to be precomputed')

    estimated_ratings = scores_cf_ii(scores, scores[userId-1][np.newaxis], [np.ones(scores.shape[1])], 
                                     n, dists=dists)[0]
    estimated_ratings[scores[userId-1] > 0] = np.nan
    
    movie_order = (-estimated_ratings).argsort()
    
    return movie_order, estimated_ratings[movie_order]