#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:48:47 2017

@author: Hao Wang
@email: wangronin@gmail.com
"""

import numpy as np
from numpy import newaxis, zeros, tile, eye, c_, ones


class trend(object):
    
    def __init__(self, n_feature, beta=None):
        self.n_feature = n_feature
        self.beta = beta
        self.n_basis = None
            
    def set_beta(self, beta):
        if beta is not None:
            beta = np.atleast_1d(beta)
            if len(beta) != self.n_basis:
                raise Exception('beta does not have the right size!')
        self.beta = beta
    
    def __call__(self, X):
        if self.beta is None:
            raise Exception('beta is not set!')
        return self.F(X).dot(self.beta)
    
    def F(self, X):
        raise NotImplementedError
    
    def check_input(self, X):
        # Check input shapes
        X = np.atleast_2d(X)
        if X.shape[1] != self.n_feature:
            X = X.T
        if X.shape[1] != self.n_feature:
            raise Exception('x does not have the right size!')
        return X
        
    def jacobian(self, X):
        raise NotImplementedError        
    
    def __eq__(self, trend_b):
        pass
    
    
class constant_trend(trend):
    """
    Zero order polynomial (constant, p = 1) regression model.

    x --> f(x) = 1
    
    """
    def __init__(self, n_feature, beta=None):
        super(constant_trend, self).__init__(n_feature, beta)
        self.n_basis = n_feature
        
    def F(self, X):
        X = self.check_input(X)
        n_eval = X.shape[0]
        return ones((n_eval, 1))
    
    def jacobian(self, X):
        X = self.check_input(X)
        n_eval = X.shape[0]
        return zeros((n_eval, self.n_feature, 1))
    
    
class linear_trend(trend):
    """
    First order polynomial (linear, p = n+1) regression model.

    x --> f(x) = [ 1, x_1, ..., x_n ].T
    """
    def __init__(self, n_feature, beta=None):
        super(linear_trend, self).__init__(n_feature, beta)
        self.n_basis = n_feature + 1
        
    def F(self, X):
        X = self.check_input(X)
        n_eval = X.shape[0]
        return c_[ones(n_eval), X]
    
    def jacobian(self, X):
        X = self.check_input(X)
        n_eval = X.shape[0]
        __ = c_[zeros(self.n_feature), eye(self.n_feature)]
        return tile(__[newaxis, ...], (n_eval, 1, 1))


class quadratic_trend(trend):
    """
    Second order polynomial (quadratic, p = n*(n-1)/2+n+1) regression model.

    x --> f(x) = [ 1, { x_i, i = 1,...,n }, { x_i * x_j,  (i,j) = 1,...,n } ].T
                                                          i > j
    """
    def __init__(self, n_feature, beta=None):
        super(linear_trend, self).__init__(n_feature, beta)
        self.n_basis = (n_feature + 1) * (n_feature + 2) / 2
        
    def F(self, X):
        X = self.check_input(X)
        n_eval = X.shape[0]
        f = c_[ones(n_eval), X]
        for k in range(self.n_feature):
            f = c_[f, X[:, k, np.newaxis] * X[:, k:]]
        return f
    
    def jacobian(self, X):
        raise NotImplementedError
        
        
class nonparametric_trend(trend):
    def __init__(self):
        pass
    
        
if __name__ == '__main__':
    T = linear_trend(2, beta=(1, 2, 10))
    
    X = np.random.randn(1, 2)
    print T(X)
    print T.jacobian(X)


# legacy functions    
def constant(x):
   
    """
    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    """
    x = np.asarray(x, dtype=np.float64)
    n_eval = x.shape[0]
    f = np.ones([n_eval, 1])
    return f


def linear(x):
    """
    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    """
    x = np.asarray(x, dtype=np.float64)
    n_eval = x.shape[0]
    f = np.hstack([np.ones([n_eval, 1]), x])
    return f


def quadratic(x):
    """
    Second order polynomial (quadratic, p = n*(n-1)/2+n+1) regression model.

    x --> f(x) = [ 1, { x_i, i = 1,...,n }, { x_i * x_j,  (i,j) = 1,...,n } ].T
                                                          i > j

    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    """

    x = np.asarray(x, dtype=np.float64)
    n_eval, n_features = x.shape
    f = np.hstack([np.ones([n_eval, 1]), x])
    for k in range(n_features):
        f = np.hstack([f, x[:, k, np.newaxis] * x[:, k:]])

    return f