import numpy as np
from utils import *
from sklearn.linear_model import LinearRegression as ScikitLinearRegression
from scipy.interpolate import UnivariateSpline
from sklego.linear_model import LowessRegression

class LinearRegression:

    def __init__(self):
        self.lr = None

    def fit(self, x, y):
        self.lr = ScikitLinearRegression()
        self.lr.fit(x.reshape(-1, 1), y)

    def predict(self, x):
        if self.lr is None:
            raise Exception("Tried to predict before training model.")
        else:
            return self.lr.predict(x.reshape(-1, 1))
    


class SmoothingSpline:

    def __init__(self, smooth_param):
        self.spline = None
        self.s = smooth_param

    def fit(self, x, y):
        x, y = remove_duplicates(x, y)
        self.spline = UnivariateSpline(x, y, s=self.s, k=3)

    def predict(self, x):
        if self.spline is None:
            raise Exception("Tried to predict before training model.")
        else:
            return self.spline(x)

    
"""
class LocalLinearRegression:

    def __init__(self, sigma):
        self.lr = None
        self.sigma = sigma

    def fit(self, x, y):
        self.lr = LowessRegression(sigma=self.sigma)
        self.lr.fit(x.reshape(-1, 1), y)

    def predict(self, x):
        if self.lr is None:
            raise Exception("Tried to predict before training model.")
        else:
            return self.lr.predict(x.reshape(-1, 1))
"""

class LocalLinearRegression:
  def __init__(self, bandwidth):
    self.h = bandwidth

  def fit(self, x, y):
    self.x = x
    self.y = y
    self.n = len(x)

  def _get_ells(self, xi):
    u = np.array([xj - xi for xj in self.x])
    weights = np.array([k(xi, xj, self.h) for xj in self.x])
    w_dot_u_squared = np.dot(weights, u**2)
    w_dot_u = np.dot(weights, u)
    b = np.array([weights[j] * (w_dot_u_squared - u[j] * w_dot_u)  for j in range(self.n)])
    ell = b / np.sum(b)
    return ell

  def _predict_one(self, xi):
    u = np.array([xj - xi for xj in self.x])
    weights = np.array([k(xi, xj, self.h) for xj in self.x])
    w_dot_u_squared = np.dot(weights, u**2)
    w_dot_u = np.dot(weights, u)
    b = np.array([weights[j] * (w_dot_u_squared - u[j] * w_dot_u)  for j in range(self.n)])
    ell = b / np.sum(b)
    return np.dot(ell, self.y)

  def predict(self, x):
    return np.array([self._predict_one(xi) for xi in x])

  def compute_cv(self):
    cv = np.sum([(self.y[i] - self._predict_one(self.x[i]))**2 /
                    (1 - self._get_ells(self.x[i])[i])**2
                    for i in range(self.n)])
    return cv
