#coding=utf-8

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.linear_model import LogisticRegression


# Параметрами с которыми вы хотите обучать деревья
TREE_PARAMS_DICT = {'max_depth': 7, 'max_features': 3}
# Параметр tau (learning_rate) для вашего GB
TAU = 0.08051


class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau
        
    def fit(self, X_data, y_data):
        self.base_algo = LogisticRegression(C=0.00001).fit(X_data, y_data)
        self.estimators = []

        curr_pred = - np.log(1. / self.base_algo.predict_proba(X_data)[:, 1] - 1)
        
        for iter_num in range(self.iters):
            yp = 1. / (1 + np.exp(-curr_pred))
            # Нужно посчитать градиент функции потерь
            grad = -yp * (1. - yp) * (y_data / yp - (1. - y_data) / (1. - yp))
            # Нужно обучить DecisionTreeRegressor предсказывать антиградиент
            # Не забудьте про self.tree_params_dict
            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, - grad)

            self.estimators.append(algo)
            # Обновите предсказания в каждой точке
            curr_pred += self.tau * algo.predict(X_data)
        return self        
    
    def predict(self, X_data):
        # Предсказание на данных
        res = -np.log(1. / self.base_algo.predict_proba(X_data)[:, 1] - 1)
        for estimator in self.estimators:
            res += self.tau * estimator.predict(X_data)
        # Задача классификации, поэтому надо отдавать 0 и 1
        return res > 0.10
