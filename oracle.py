
class MyOracle():
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def get_label(self, idxs):
        return self.y[idxs]