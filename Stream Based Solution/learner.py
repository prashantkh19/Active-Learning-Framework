import numpy as np
from strategy import *
from tqdm import tqdm_notebook as tqdm
    
class ActiveLearner():
    
    def __init__(self, models, strategy = "uncertainity", uncertainity_measure = None, committee_measure = "vote entropy"):
        self.models = models
        self.committee_measure = committee_measure
        self.uncertainity_measure = uncertainity_measure
        self.X = None
        self.y = None
        self.strategy = strategy

        if len(models) == 0:
            raise Exception('Please provide a list of models with length at least 1')

        if(strategy == "uncertainity" and uncertainity_measure is None):
            raise Exception('uncertainity_measure is not provided')
        
        if(strategy == "committee" and committee_measure is None):
            raise Exception('committee_measure is not provided')

        if(strategy == "uncertainity" and len(models) > 1):
            warnings.warn("Multiple models are not required in case of uncertainity sampling, only 1st model will be used")

        if(strategy == "committee" and len(models) == 1):
            warnings.warn("Multiple models are recommended in case of committee strategy")
        
        
    
    def query(self, X_pool):
        if self.strategy == "uncertainity":
            return uncertainty_sampling(X_pool, self.models[0], self.uncertainity_measure)

        elif self.strategy == "committee":
            return query_by_committee(X_pool, self.models, self.committee_measure)

        else:
            return None
    
    def score(self, X, y):
        if self.strategy == "committee":
            # TODO:// Improve this
            return  max([model.score(X, y) for model in self.models])
        #     score = 0
        #     for model in self.models:
        #         score += model.score(X, y)
        #     return score/len(self.models)
        else: 
            return self.models[0].score(X, y) 
        
    def teach(self, X, y):
        if self.X is None:
            self.X = X
            self.y = y

        else:
            self.X = np.concatenate((self.X, X))
            self.y = np.concatenate((self.y, y))

        for model in self.models:
            model.fit(self.X, self.y)
