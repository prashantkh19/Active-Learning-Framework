import numpy as np
import warnings

def uncertainity_sampling(X, model, measure):
    threshold = 5
    y_prob = model.predict_proba(X)
    y_prob_sorted = - np.sort(-y_prob, axis = 1)
    
    if measure == "lc":
        return np.argsort(y_prob_sorted[:, 1])[:threshold]
    
    elif measure == "m":
        return np.argsort(y_prob_sorted[:, 1] - y_prob_sorted[:, 2])[:threshold]
    
    elif measure == "h":
        votes = np.zeros(len(X))
        no_of_classes = y_prob.shape[1]
        for i in range(no_of_classes):
            votes += y_prob[:, i] * np.log(y_prob[:, i]) 
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

    if measure == "vote_entropy":
        votes = np.zeros(len(X))
        for i in range(no_of_classes):
            vote = np.zeros(len(X))
            for model_op in results:
                vote += (np.equal(model_op, np.full(model_op.shape, i)) + 0.5)
                vote /= C
                vote = vote * np.log(vote)
            votes += vote
        return np.argsort(votes)[:5]
    
    else:
        warnings.warn("committee measure '"+ measure + "' is not applicable")
        return None
