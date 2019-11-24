import numpy as np
import warnings

epsilon = 1e-7

def uncertainity_sampling(X, model, measure):
    threshold = 5
    y_prob = model.predict_proba(X)
    y_prob_sorted = - np.sort(-y_prob, axis = 1)
    
    if measure == "lc":
        return np.argsort(y_prob_sorted[:, 0])[:threshold]
    
    elif measure == "m":
        return np.argsort(y_prob_sorted[:, 0] - y_prob_sorted[:, 1])[:threshold]
    
    elif measure == "h":
        votes = np.zeros(len(X))
        no_of_classes = y_prob.shape[1]
        for i in range(no_of_classes):
            votes += y_prob[:, i] * np.log(y_prob[:, i] + epsilon) 
        return np.argsort(votes)[:5]
    
    else:
        warnings.warn("uncertainity measure '"+ measure + "' is not applicable")
        return None
    
def query_by_committee(X, committee, measure):
    threshold = 5
    C = len(committee)
    no_of_classes = None
    results = []
    
    for model in committee:
        y_prob = model.predict_proba(X)
        no_of_classes = y_prob.shape[1]
        y_class = np.argmax(y_prob, axis = 1)
        results.append(y_class)

    mat = np.transpose(np.asarray(results))
        
    if measure == "vote_entropy":
        votes = np.zeros(len(X))
        for i in range(no_of_classes):
            vote = np.zeros(len(X))
            for j, row in enumerate(mat):
                unique, counts = np.unique(row, return_counts=True)
                count_dict = dict(zip(unique, counts))
                try:
                    vote[j] = count_dict[i]
                except:
                    vote[j] = epsilon
            vote /= C
            vote = vote * np.log(vote)
            votes += vote
        return np.argsort(votes)[:5]
    
    else:
        warnings.warn("committee measure '"+ measure + "' is not applicable")
        return None
