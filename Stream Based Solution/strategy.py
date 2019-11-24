import numpy as np
import warnings
from scipy.stats import entropy

def uncertainty_sampling(X, model, measure, threshold=5, lc_thresh=0.6, m_thresh=0.2, h_thresh=0.8):
    if len(X.shape) == 1:
        X = X[None, :]
    y_prob = model.predict_proba(X) # Returns a (samples x classes) array of class probabilities
    y_prob_sorted = - np.sort(-y_prob, axis = 1) # Sorts probabilities in decreasing order for each sample
    # X.length

    if measure == "lc":

        # FYI: argsort returns indices for a list sorted in increasing order
        if X.shape[0] == 1:
            return (y_prob_sorted[0, 0] < lc_thresh)
        return np.argsort(y_prob_sorted[:, 0])[:threshold] # Changed [: 1] to [:, 0]
    
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
    
def query_by_committee(X, committee, measure, kl_thresh=0.4):
    threshold = 5
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

    if measure == "vote entropy":
        votes = np.zeros((X.shape[0], no_of_classes))
        for pred in results:
            votes[:, pred] += 1 # Counting number of votes for each class
        votes = votes / C # This gets the fractional distribution of each vote
        vote_entropy = entropy(votes.T)

        if vote_entropy.shape[0] == 1: # For stream based learning
            return vote_entropy[0] > 0.60 # Value found while experimenting

        return np.argsort(votes)[:threshold]
    else:
        warnings.warn("committee measure '"+ measure + "' is not applicable")
        return None
