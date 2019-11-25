import numpy as np
import warnings
from pdb import set_trace
from sklearn.cluster import KMeans

epsilon = 1e-7

def entropy(pk, qk=None):
    pk = 1.0 * pk / np.sum(pk, axis=0) # Normalizing qk
    pk[pk == 0] = epsilon
    if qk is not None:
        qk = 1.0 * qk / np.sum(qk, axis=0) # Normalizing qk
        qk[qk == 0] = epsilon
        S = pk * np.log(pk / qk)
        # S = S[np.isnan(S) != True].sum(axis = 0)
        return S.sum(axis=0)
    else:
        S = pk * np.log(pk)
        # S = S[np.isnan(S) != True].sum(axis = 0)
        return -S.sum(axis=0)


def uncertainity_sampling(X, model, measure):
    threshold = 5
    lc_thresh=0.6; m_thresh=0.2; h_thresh=0.8

    y_prob = model.predict_proba(X)
    y_prob_sorted = - np.sort(-y_prob, axis = 1)
    
    if measure == "lc":
        if X.shape[0] == 1:
            return (y_prob_sorted[0, 0] < lc_thresh)
        return np.argsort(y_prob_sorted[:, 0])[:threshold]
    
    elif measure == "m":
        if X.shape[0] == 1:
            return (y_prob_sorted[:, 0] - y_prob_sorted[:, 1]) < m_thresh
        return np.argsort(y_prob_sorted[:, 0] - y_prob_sorted[:, 1])[:threshold]
    
    elif measure == "h":
        entropies = entropy(y_prob.T) 
        if X.shape[0] == 1:
            return (entropies[0] > h_thresh)
        return np.argsort(entropies)[:5]
    
    else:
        warnings.warn("uncertainity measure '"+ measure + "' is not applicable")
        return None
    
def query_by_committee(X, committee, measure):
    threshold = 5
    kl_thresh=0.4
    C = len(committee)
    no_of_classes = None
    results = []
    num_samples = X.shape[0]
    
    if measure == "kl divergence":
        probabilities = np.array([model.predict_proba(X) for model in committee])
        consensus_probabilites = np.sum(probabilities, axis=0) / len(committee)
        kl_divergence = np.zeros((num_samples, C))
        for i in range(num_samples):
            for m in range(C):
                kl_divergence[i, m] = entropy(probabilities[m, i], qk=consensus_probabilites[i])
        max_disagreements = np.max(kl_divergence, axis=1)
        if num_samples == 1:
            return max_disagreements[0] > kl_thresh
        else:
            return np.argsort(max_disagreements)[:threshold]

    for model in committee:
        y_prob = model.predict_proba(X)
        no_of_classes = y_prob.shape[1]
        y_class = np.argmax(y_prob, axis = 1)
        results.append(y_class)

    mat = np.transpose(np.asarray(results))
        
    if measure == "vote_entropy":
        votes = np.zeros((X.shape[0], no_of_classes))
        for pred in results:
            votes[:, pred] += 1 # Counting number of votes for each class
        votes = votes / C # This gets the fractional distribution of each vote
        vote_entropy = entropy(votes.T)
        # pdb.set_trace()
        if vote_entropy.shape[0] == 1: # For stream based learning
            return vote_entropy[0] > 0.60 # Value found while experimenting

        return np.argsort(vote_entropy)[:threshold]
        # votes = np.zeros(len(X))
        # for i in range(no_of_classes):
        #     vote = np.zeros(len(X))
        #     for j, row in enumerate(mat):
        #         unique, counts = np.unique(row, return_counts=True)
        #         count_dict = dict(zip(unique, counts))
        #         try:
        #             vote[j] = count_dict[i]
        #         except:
        #             vote[j] = epsilon
        #     vote /= C
        #     vote = vote * np.log(vote)
        #     votes += vote
        # return np.argsort(votes)[:5]
    
    else:
        warnings.warn("committee measure '"+ measure + "' is not applicable")
        return None

def diversity_sampling(X, model, num_classes=3, threshold=5):
    kmeans = KMeans(n_clusters=num_classes, random_state=42).fit(X)
    index = np.arange(X.shape[0])
    # Separate X into different clusters based on classes and keep track of indices
    clusters = [X[kmeans.labels_ == i] for i in range(num_classes)]
    indices = [index[kmeans.labels_ == i] for i in range(num_classes)]
    # get predicted class probabilities for all clusters
    preds = [ np.max(model.predict_proba(clusters[i]), axis=1 ) for i in range(num_classes) ]
    # sort and get indices
    pred_sorted = [ np.argsort(preds[i]) for i in range(num_classes)]
    # Re-shuffle original indices
    indices_sorted = [indices[i][pred_sorted[i]] for i in range(num_classes)]
    ranking = np.zeros(threshold)
    
    i = 0
    rank_index = 0
    # set_trace()
    while rank_index < threshold:
        # Take the least certain sample indices from each cluster one by one
        for j in range(num_classes):
            if rank_index == threshold:
                break
            elif len(indices_sorted[j]) > i:
                ranking[rank_index] = indices_sorted[j][i]
                rank_index += 1
        i += 1
    # set_trace()
    return ranking.astype(np.int32)


