import numpy as np
from utils import per_user_means, per_user_stds
from baselines import baseline_predictors


'''
Basic Slope One
'''

def calculate_similarities_SO(scores, nan_to_0=True):
    rated = scores > 0
    dists = scores.T.dot(rated)
    dists = dists - dists.T
    cards = rated.T.dot(rated.astype(np.float))
    return np.nan_to_num(dists / cards), cards


def scores_SO_basic(train, query, target_inds, dists_cards=None, min_rate=None):
    if dists_cards is None:
        raise NotImplementedError('distances and cardinalities have to be precomputed')
        
    dists, cards = dists_cards
    baselines = baseline_predictors(train, query)
    cards_nonzero = cards > 0
        
    ratings = []    
        
    for u_idx in xrange(query.shape[0]):
        user_ratings = query[u_idx]
        user_rated = user_ratings > 0
        
        targets = np.where(target_inds[u_idx])[0]        

        weights = cards_nonzero[targets][:, user_rated]
        scores = dists[targets][:, user_rated] + user_ratings[user_rated]
        scores = (scores * weights).sum(axis=1) / weights.sum(axis=1)
        
        estimated_ratings = np.zeros(query.shape[1])
        estimated_ratings[targets] = scores
        estimated_ratings[user_rated] = user_ratings[user_rated]
        
        estimated_ratings[np.isnan(estimated_ratings)] = baselines[u_idx][np.isnan(estimated_ratings)]
        if min_rate:
            estimated_ratings[estimated_ratings > 5] = 5
            estimated_ratings[estimated_ratings < min_rate] = min_rate
        
        ratings.append(estimated_ratings[np.newaxis])
        
    return np.vstack(ratings)


def recommend_SO(userId, scores, mode='basic', dists_cards=None):
    if dists_cards is None:
        raise NotImplementedError('distances have to be precomputed')
    
    user_scores = scores[userId-1][np.newaxis]
    targets = [np.ones(scores.shape[1])]
    
    if mode == 'basic':
        estimated_ratings = scores_SO_basic(scores, user_scores, targets, dists_cards)[0]
    elif mode == 'weighted':
        estimated_ratings = scores_SO_weighted(scores, user_scores, targets, dists_cards)[0]
    elif mode == 'bipolar':
        dists_cards_l, dists_cards_d = dists_cards
        estimated_ratings = scores_SO_bipolar(scores, user_scores, targets, dists_cards_l, dists_cards_d)[0]
    estimated_ratings[scores[userId-1] > 0] = np.nan
    
    movie_order = (-estimated_ratings).argsort()
    
    return movie_order, estimated_ratings[movie_order]


'''
Weighted Slope One
'''

def scores_SO_weighted(train, query, target_inds, dists_cards=None, min_rate=None):
    if dists_cards is None:
        raise NotImplementedError('distances and cardinalities have to be precomputed')
        
    dists, cards = dists_cards
    baselines = baseline_predictors(train, query)
        
    ratings = []    
        
    for u_idx in xrange(query.shape[0]):
        user_ratings = query[u_idx]
        user_rated = user_ratings > 0
        
        targets = np.where(target_inds[u_idx])[0]        
        
        weights = cards[targets][:, user_rated]
        scores = dists[targets][:, user_rated] + user_ratings[user_rated]
        scores = (scores * weights).sum(axis=1) / weights.sum(axis=1)
        
        estimated_ratings = np.zeros(query.shape[1])
        estimated_ratings[targets] = scores
        estimated_ratings[user_rated] = user_ratings[user_rated]
        
        estimated_ratings[np.isnan(estimated_ratings)] = baselines[u_idx][np.isnan(estimated_ratings)]
        if min_rate:
            estimated_ratings[estimated_ratings > 5] = 5
            estimated_ratings[estimated_ratings < min_rate] = min_rate
        
        ratings.append(estimated_ratings[np.newaxis])
        
    return np.vstack(ratings)


'''
Bi-polar Slope One
'''

def calculate_similarities_SO_bipolar(scores, nan_to_0=True):
    means = per_user_means(scores)[:, np.newaxis]
    liked = scores >= means
    disliked = (scores < means) * (scores > 0)
    
    def calculate_dists(mask):
        dists = (scores * mask).T.dot(mask)
        dists = dists - dists.T
        cards = mask.T.dot(mask.astype(np.float))
        return np.nan_to_num(dists / cards), cards
    
    dists_like = calculate_dists(liked)
    dists_dislike = calculate_dists(disliked)
    
    return dists_like, dists_dislike


def scores_SO_bipolar(train, query, target_inds, dists_cards_liked=None, dists_cards_disliked=None, min_rate=None):
    if dists_cards_liked is None or dists_cards_disliked is None:
        raise NotImplementedError('distances and cardinalities have to be precomputed')
        
    dists_l, cards_l = dists_cards_liked
    dists_d, cards_d = dists_cards_disliked
    baselines = baseline_predictors(train, query)
        
    ratings = []    
        
    for u_idx in xrange(query.shape[0]):
        user_ratings = query[u_idx]
        user_rated = user_ratings > 0
        user_mean = user_ratings.sum() / user_rated.sum()
        
        user_l = user_ratings >= user_mean
        user_d = (user_ratings < user_mean) * (user_ratings > 0)
        
        targets = np.where(target_inds[u_idx])[0]        
        
        weights_l = cards_l[targets][:, user_l]
        weights_d = cards_d[targets][:, user_d]
        
        scores_l = ((dists_l[targets][:, user_l] + user_ratings[user_l]) * weights_l).sum(axis=1)
        scores_d = ((dists_d[targets][:, user_d] + user_ratings[user_d]) * weights_d).sum(axis=1)
        
        scores = (scores_l + scores_d) / (weights_l.sum(axis=1) + weights_d.sum(axis=1))
        
        estimated_ratings = np.zeros(query.shape[1])
        estimated_ratings[targets] = scores
        estimated_ratings[user_rated] = user_ratings[user_rated]
        
        estimated_ratings[np.isnan(estimated_ratings)] = baselines[u_idx][np.isnan(estimated_ratings)]
        if min_rate:
            estimated_ratings[estimated_ratings > 5] = 5
            estimated_ratings[estimated_ratings < min_rate] = min_rate
        
        ratings.append(estimated_ratings[np.newaxis])
        
    return np.vstack(ratings)
