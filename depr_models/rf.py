import pandas as pd
from os.path import join
from sklearn.ensemble import RandomForestRegressor

class RF:
    def __init__(self, n_estimators, seed):
        self.n_estimators = n_estimators
        self.seed = seed
        
    def fit(self, X, y):
        model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.seed)
        model.fit(X, y)
        return model
        
    def save_model(self, model, save_dir):
        pass
        
    def predict(self, model, X, save_dir, X_train=None, y_train=None):
        return self.model.predict(X)