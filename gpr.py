# -*- coding: utf-8 -*-

# Author: Hao Wang <wangronin@gmail.com>

# from __future__ import print_function

import pdb
import warnings

import numpy as np
from numpy.random import uniform
from numpy import log, pi, log10

from scipy import linalg
from scipy.linalg import cho_solve
from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.utils import check_random_state, check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

from .cma_es import cma_es
from .kernel import *
from .trend import constant_trend, linear_trend, quadratic_trend

MACHINE_EPSILON = np.finfo(np.double).eps


def l1_cross_distances(X):
    """
    Computes the nonzero componentwise L1 cross-distances between the vectors
    in X.

    Parameters
    ----------

    X: array_like
        An array with shape (n_samples, n_features)

    Returns
    -------

    D: array with shape (n_samples * (n_samples - 1) / 2, n_features)
        The array of componentwise L1 cross-distances.

    ij: arrays with shape (n_samples * (n_samples - 1) / 2, 2)
        The indices i and j of the vectors in X associated to the cross-
        distances in D: D[k] = np.abs(X[ij[k, 0]] - Y[ij[k, 1]]).
    """
    X = check_array(X)
    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0
    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = np.abs(X[k] - X[(k + 1):n_samples])

    return D, ij


# TODO: remove the dependences from sklearn
class GaussianProcess(BaseEstimator, RegressorMixin):
    """The Gaussian Process model class.

    Read more in the :ref:`User Guide <gaussian_process>`.

    Parameters
    ----------
    regr : string or callable, optional
        A regression function returning an array of outputs of the linear
        regression functional basis. The number of observations n_samples
        should be greater than the size p of this basis.
        Default assumes a simple constant regression trend.
        Available built-in regression models are::

            'constant', 'linear', 'quadratic'

    corr : string or callable, optional
        A stationary autocorrelation function returning the autocorrelation
        between two points x and x'.
        Default assumes a squared-exponential autocorrelation model.
        Built-in correlation models are::

            'absolute_exponential', 'squared_exponential',
            'generalized_exponential', 'cubic', 'linear', 'matern'

    beta0 : double array_like, optional
        The regression weight vector to perform Ordinary Kriging (OK).
        Default assumes Universal Kriging (UK) so that the vector beta of
        regression weights is estimated using the maximum likelihood
        principle.

    storage_mode : string, optional
        A string specifying whether the Cholesky decomposition of the
        correlation matrix should be stored in the class (storage_mode =
        'full') or not (storage_mode = 'light').
        Default assumes storage_mode = 'full', so that the
        Cholesky decomposition of the correlation matrix is stored.
        This might be a useful parameter when one is not interested in the
        MSE and only plan to estimate the BLUP, for which the correlation
        matrix is not required.

    verbose : boolean, optional
        A boolean specifying the verbose level.
        Default is verbose = False.

    theta0 : double array_like, optional
        An array with shape (n_features, ) or (1, ).
        The parameters in the autocorrelation model.
        If thetaL and thetaU are also specified, theta0 is considered as
        the starting point for the maximum likelihood estimation of the
        best set of parameters.
        Default assumes isotropic autocorrelation model with theta0 = 1e-1.

    thetaL : double array_like, optional
        An array with shape matching theta0's.
        Lower bound on the autocorrelation parameters for maximum
        likelihood estimation.
        Default is None, so that it skips maximum likelihood estimation and
        it uses theta0.

    thetaU : double array_like, optional
        An array with shape matching theta0's.
        Upper bound on the autocorrelation parameters for maximum
        likelihood estimation.
        Default is None, so that it skips maximum likelihood estimation and
        it uses theta0.

    normalize : boolean, optional
        Input X and observations y are centered and reduced wrt
        means and standard deviations estimated from the n_samples
        observations provided.
        Default is normalize = True so that data is normalized to ease
        maximum likelihood estimation.

    TODO: the nugget behaves differently than what is described here...
    nugget : double or ndarray, optional
        Introduce a nugget effect to allow smooth predictions from noisy
        data.  If nugget is an ndarray, it must be the same length as the
        number of data points used for the fit.
        The nugget is added to the diagonal of the assumed training covariance;
        in this way it acts as a Tikhonov regularization in the problem.  In
        the special case of the squared exponential correlation function, the
        nugget mathematically represents the variance of the input values.
        Default assumes a nugget close to machine precision for the sake of
        robustness (nugget = 10. * MACHINE_EPSILON).

    optimizer : string, optional
        A string specifying the optimization algorithm to be used.
        Default uses 'fmin_cobyla' algorithm from scipy.optimize.
        Available optimizers are::

            'fmin_cobyla', 'Welch'

        'Welch' optimizer is dued to Welch et al., see reference [WBSWM1992]_.
        It consists in iterating over several one-dimensional optimizations
        instead of running one single multi-dimensional optimization.

    random_start : int, optional
        The number of times the Maximum Likelihood Estimation should be
        performed from a random starting point.
        The first MLE always uses the specified starting point (theta0),
        the next starting points are picked at random according to an
        exponential distribution (log-uniform on [thetaL, thetaU]).
        Default does not use random starting point (random_start = 1).

    random_state: integer or numpy.RandomState, optional
        The generator used to shuffle the sequence of coordinates of theta in
        the Welch optimizer. If an integer is given, it fixes the seed.
        Defaults to the global numpy random number generator.


    Attributes
    ----------
    theta_ : array
        Specified theta OR the best set of autocorrelation parameters (the \
        sought maximizer of the reduced likelihood function).

    log_likelihood_ : array
        The optimal reduced likelihood function value.

    Examples
    --------

    References
    ----------

    """

    _regression_types = {
        'constant': constant_trend,
        'linear': linear,
        'quadratic': quadratic}

    _correlation_types = {
        'absolute_exponential': absolute_exponential,
        'squared_exponential': squared_exponential,
        'generalized_exponential': generalized_exponential,
        'cubic': cubic,
        'matern': matern,
        'linear': linear}

    _optimizer_types = [
        'BFGS',
        'CMA']

    # 10. * MACHINE_EPSILON
    def __init__(self, regr='constant', corr='squared_exponential', beta0=None, 
                 theta0=1e-1, thetaL=None, thetaU=None, sigma2=None, optimizer='BFGS', 
                 normalize=False, random_start=1, nugget=None, nugget_estim=False, 
                 wait_iter=5, eval_budget=None, random_state=None, verbose=False):

        self.regr = regr
        self.corr = corr
        self.beta0 = beta0
        self.sigma2 = sigma2
        self.verbose = verbose
        
        # hyperparameters: kernel function
        self.theta0 = np.array(theta0).flatten()
        self.thetaL = np.array(thetaL).flatten()
        self.thetaU = np.array(thetaU).flatten()
        
        if not (np.isfinite(self.thetaL).all() and np.isfinite(self.thetaU).all()):
            raise ValueError("all bounds are required finite.")
        
        # TODO: remove normalize in the future
        self.normalize = normalize
        
        # model optimization parameters
        self.optimizer = optimizer
        self.random_start = random_start
        self.random_state = random_state
        self.wait_iter = wait_iter
        self.eval_budget = eval_budget

        self.noise_var = np.atleast_1d(nugget) if nugget is not None else None
        self.nugget_estim = True if nugget_estim else False
        self.noisy = True if (self.noise_var is not None) or \
            self.nugget_estim else False

        # three cases to compute the log-likelihood function
        # TODO: verify: it seems the noisy case is the most useful one
        if not self.noisy:
            self.llf_mode = 'noiseless'
        elif self.nugget_estim:
            self.llf_mode = 'nugget_estim'
        else:
            self.llf_mode = 'noisy'

    def _process_data(self, X, y):
        # Force data to 2D numpy.array
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self.y_ndim_ = y.ndim
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # Check shapes of DOE & observations
        n_samples, n_features = X.shape
        _, n_targets = y.shape

        # Run input checks
        self._check_params(n_samples)

        # Normalize data or don't
#        if self.normalize:
#            X_mean = np.mean(X, axis=0)
#            X_std = np.std(X, axis=0)
#            y_mean = np.mean(y, axis=0)
#            y_std = np.std(y, axis=0)
#            X_std[X_std == 0.] = 1.
#            y_std[y_std == 0.] = 1.
#            # center and scale X if necessary
#            X = (X - X_mean) / X_std
#            y = (y - y_mean) / y_std
#        else:
        X_mean = np.zeros(1)
        X_std = np.ones(1)
        y_mean = np.zeros(1)
        y_std = np.ones(1)

        # Calculate matrix of distances D between samples
        D, ij = l1_cross_distances(X)
        if (np.min(np.sum(D, axis=1)) == 0. and self.corr != pure_nugget):
            raise Exception("Multiple input features cannot have the same"
                            " target value.")

        # Regression matrix and parameters
        F = self.regr.F(X)
        n_samples_F = F.shape[0]
        if F.ndim > 1:
            p = F.shape[1]
        else:
            p = 1
        if n_samples_F != n_samples:
            raise Exception("Number of rows in F and X do not match. Most "
                            "likely something is going wrong with the "
                            "regression model.")
        if p > n_samples_F:
            raise Exception(("Ordinary least squares problem is undetermined "
                             "n_samples=%d must be greater than the "
                             "regression model size p=%d.") % (n_samples, p))
        if self.beta0 is not None:
            if self.beta0.shape[0] != p:
                raise Exception("Shapes of beta0 and F do not match.")

        # Set attributes
        self.X = X
        self.y = y
        self.D = D
        self.ij = ij
        self.F = F
        self.X_mean, self.X_std = X_mean, X_std
        self.y_mean, self.y_std = y_mean, y_std

    def fit(self, X, y):
        """
        The Gaussian Process model fitting method.

        Parameters
        ----------
        X : double array_like
            An array with shape (n_samples, n_features) with the input at which
            observations were made.

        y : double array_like
            An array with shape (n_samples, ) or shape (n_samples, n_targets)
            with the observations of the output to be predicted.

        Returns
        -------
        gp : self
            A fitted Gaussian Process model object awaiting data to perform
            predictions.
        """
        # Run input checks
        self._check_params()
        self.random_state = check_random_state(self.random_state)
        self._process_data(X, y)

        # Determine Gaussian Process model parameters
        if self.thetaL is not None and self.thetaU is not None:
            # Maximum Likelihood Estimation of the parameters
            if self.verbose:
                print ("Maximum Likelihood Estimation of the hyperparameters...")
            self.theta_, self.log_likelihood_, par = self._arg_max_log_likelihood_function()
            
            if np.isinf(self.log_likelihood_):
                raise Exception("Bad parameter region. Try increasing upper bound")
        else:
            # Given parameters
            if self.verbose:
                print "Given hyperparameters"
                
            par = {}
            self.theta_ = self.theta0
            self.log_likelihood_ = self.log_likelihood_function(np.r_[self.theta_.flatten(), 
                                                                      self.sigma2], par)
            if np.isinf(self.log_likelihood_):
                raise Exception("Bad point. Try increasing theta0.")

        self.noise_var = par['noise_var']
        self.sigma2 = par['sigma2']
        self.rho = par['rho']
        self.Yt = par['Yt']
        self.C = par['C']
        self.Ft = par['Ft']
        self.G = par['G']
        self.Q = par['Q']

        # compute for beta and gamma
        self.compute_beta_gamma()

        return self

    def update(self, X, y):
        """
        update the model's data set without re-estimation of parameters
        """
        # TODO: implement incremental training 
        self.fit(X, y)

        return self

    def predict(self, X, eval_MSE=False, batch_size=None):
        """
        This function evaluates the Gaussian Process model at x.

        Parameters
        ----------
        X : array_like
            An array with shape (n_eval, n_features) giving the point(s) at
            which the prediction(s) should be made.

        eval_MSE : boolean, optional
            A boolean specifying whether the Mean Squared Error should be
            evaluated or not.
            Default assumes evalMSE = False and evaluates only the BLUP (mean
            prediction).

        batch_size : integer, optional
            An integer giving the maximum number of points that can be
            evaluated simultaneously (depending on the available memory).
            Default is None so that all given points are evaluated at the same
            time.

        Returns
        -------
        y : array_like, shape (n_samples, ) or (n_samples, n_targets)
            An array with shape (n_eval, ) if the Gaussian Process was trained
            on an array of shape (n_samples, ) or an array with shape
            (n_eval, n_targets) if the Gaussian Process was trained on an array
            of shape (n_samples, n_targets) with the Best Linear Unbiased
            Prediction at x.

        MSE : array_like, optional (if eval_MSE == True)
            An array with shape (n_eval, ) or (n_eval, n_targets) as with y,
            with the Mean Squared Error at x.
        """
        check_is_fitted(self, "X")

        # Check input shapes
        # TODO: remove the support for multiple independent outputs
        X = check_array(X)
        n_eval, _ = X.shape
        n_samples, n_features = self.X.shape
        n_samples_y, n_targets = self.y.shape

        # Run input checks
        self._check_params(n_samples)

        if X.shape[1] != n_features:
            raise ValueError(("The number of features in X (X.shape[1] = %d) "
                              "should match the number of features used "
                              "for fit() "
                              "which is %d.") % (X.shape[1], n_features))

        if batch_size is None:
            # No memory management
            # (evaluates all given points in a single batch run)

            # Normalize input
            X = (X - self.X_mean) / self.X_std

            # Initialize output
            y = np.zeros(n_eval)
            if eval_MSE:
                MSE = np.zeros(n_eval)

            # Get pairwise componentwise L1-distances to the input training set
            dx = manhattan_distances(X, Y=self.X, sum_over_features=False)
            # Get regression function and correlation
            f = self.regr.F(X)
            r = self.corr(self.theta_, dx).reshape(n_eval, n_samples)

            # Scaled predictor
            y_ = np.dot(f, self.beta) + np.dot(r, self.gamma)

            # Predictor
            y = (self.y_mean + self.y_std * y_).reshape(n_eval, n_targets)

            if self.y_ndim_ == 1:
                y = y.ravel()

            # Mean Squared Error
            if eval_MSE:
                rt = linalg.solve_triangular(self.C, r.T, lower=True)

                if self.beta0 is None:
                    # Universal Kriging
                    u = linalg.solve_triangular(self.G.T,
                                                np.dot(self.Ft.T, rt) - f.T,
                                                lower=True)
                else:
                    # Ordinary Kriging
                    u = np.zeros((n_targets, n_eval))

                MSE = np.dot(self.sigma2.reshape(n_targets, 1),
                             (1. - (rt ** 2.).sum(axis=0) + 
                             (u ** 2.).sum(axis=0))[np.newaxis, :])
                MSE = np.sqrt((MSE ** 2.).sum(axis=0) / n_targets)

                # Mean Squared Error might be slightly negative depending on
                # machine precision: force to zero!
                MSE[MSE < 0.] = 0.

                if self.y_ndim_ == 1:
                    MSE = MSE.ravel()

                return y, MSE

            else:

                return y

        else:
            # Memory management
            if type(batch_size) is not int or batch_size <= 0:
                raise Exception("batch_size must be a positive integer")

            if eval_MSE:

                y, MSE = np.zeros(n_eval), np.zeros(n_eval)
                for k in range(max(1, n_eval / batch_size)):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size + 1, n_eval + 1])
                    y[batch_from:batch_to], MSE[batch_from:batch_to] = \
                        self.predict(X[batch_from:batch_to],
                                     eval_MSE=eval_MSE, batch_size=None)

                return y, MSE

            else:

                y = np.zeros(n_eval)
                for k in range(max(1, n_eval / batch_size)):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size + 1, n_eval + 1])
                    y[batch_from:batch_to] = \
                        self.predict(X[batch_from:batch_to],
                                     eval_MSE=eval_MSE, batch_size=None)

                return y

    def corr_grad_theta(self, theta, X, R, nu=1.5):
        # Check input shapes
        X = np.atleast_2d(X)
        n_eval, _ = X.shape
        n_features = self.X.shape[1]

        if _ != n_features:
            raise Exception('x does not have the right size!')

        diff = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2.

        if self.corr_type == 'squared_exponential':
            grad = -diff * R[..., np.newaxis]

        elif self.corr_type == 'matern':
            c = np.sqrt(3)
            D = np.sqrt(np.sum(theta * diff, axis=-1))

            if nu == 0.5:
                grad = - diff * theta / D * R
            elif nu == 1.5:
                grad = -3 * np.exp(-c * D)[..., np.newaxis] * diff / 2.
            elif nu == 2.5:
                pass

        elif self.corr_type == 'absolute_exponential':
            grad = -np.sqrt(diff) * R[..., np.newaxis]
        elif self.corr_type == 'generalized_exponential':
            pass
        elif self.corr_type == 'cubic':
            pass
        elif self.corr_type == 'linear':
            pass

        return grad

    def correlation_matrix(self, theta, X=None):
        D = self.D
        ij = self.ij

        n_samples = self.X.shape[0]

        # Set up R
        r = self.corr(theta, D)

        R = np.eye(n_samples)
        R[ij[:, 0], ij[:, 1]] = r
        R[ij[:, 1], ij[:, 0]] = r

        return R

    def compute_beta_gamma(self):
        if self.beta0 is None:
            # Universal Kriging
            self.beta = linalg.solve_triangular(self.G, 
                                                np.dot(self.Q.T, self.Yt))
        else:
            # Ordinary Kriging
            self.beta = np.array(self.beta0)
        self.gamma = linalg.solve_triangular(self.C.T, self.rho).reshape(-1, 1)

    def _compute_aux_var(self, R):
        # Cholesky decomposition of R: Note that this matrix R can be singular
        # change notation from 'C' to 'L': 'L'ower triagular component...
        try:
            L = linalg.cholesky(R, lower=True)
        except linalg.LinAlgError as e:
            raise e

        # Get generalized least squares solution
        Ft = linalg.solve_triangular(L, self.F, lower=True)
        Yt = linalg.solve_triangular(L, self.y, lower=True)

        # compute rho
        Q, G = linalg.qr(Ft, mode='economic')
        rho = Yt - np.dot(Q.dot(Q.T), Yt)

        return L, Ft, Yt, Q, G, rho

    def log_likelihood_function(self, hyper_par, par_out=None, eval_grad=False):
        """
        TODO: rewrite the documentation here
        TODO: maybe eval_hessian in the future?...
        This function determines the BLUP parameters and evaluates the reduced
        likelihood function for the given autocorrelation parameters theta.

        Parameters
        ----------
        theta : array_like, optional
            An array containing the autocorrelation parameters at which the
            Gaussian Process model parameters should be determined.
            Default uses the built-in autocorrelation parameters
            (ie ``theta = self.theta_``).

        Returns
        -------
        log_likelihood_function_value : double
            The value of the reduced likelihood function associated to the
            given autocorrelation parameters theta.

        par : dict
            A dictionary containing the requested Gaussian Process model
            parameters:

                sigma2
                        Gaussian Process variance.
                beta
                        Generalized least-squares regression weights for
                        Universal Kriging or given beta0 for Ordinary
                        Kriging.
                gamma
                        Gaussian Process weights.
                C
                        Cholesky decomposition of the correlation matrix [R].
                Ft
                        Solution of the linear equation system : [R] x Ft = F
                G
                        QR decomposition of the matrix Ft.
        """
        check_is_fitted(self, "X")

        # Log-likelihood
        log_likelihood = -np.inf

        # Retrieve data
        n_samples, n_features = self.X.shape
        n_par = len(hyper_par)

        if self.llf_mode == 'noiseless':
            theta = hyper_par
            noise_var = 0

            R0 = self.correlation_matrix(theta)
            
            try:
                L, Ft, Yt, Q, G, rho = self._compute_aux_var(R0)
            except linalg.LinAlgError:
                if eval_grad:
                    return (log_likelihood, np.zeros((n_par, 1)))
                else:
                    return log_likelihood
                    
            sigma2 = (rho ** 2.).sum(axis=0) / n_samples
            log_likelihood = -0.5 * (n_samples * log(2. * pi * sigma2) + 
                                     2. * np.log(np.diag(L)).sum() + n_samples)

        elif self.llf_mode == 'nugget_estim':
            theta, alpha = hyper_par[:-1], hyper_par[-1]
            R0 = self.correlation_matrix(theta)
            R = alpha * R0 + (1 - alpha) * np.eye(n_samples)

            try:
                L, Ft, Yt, Q, G, rho = self._compute_aux_var(R)
            except linalg.LinAlgError:
                if eval_grad:
                    return (log_likelihood, np.zeros(n_par, 1))
                else:
                    return log_likelihood

            sigma2_total = (rho ** 2.).sum(axis=0) / n_samples
            sigma2, noise_var = alpha * sigma2_total, (1 - alpha) * sigma2_total
            log_likelihood = -0.5 * (n_samples * log(2. * pi * sigma2_total) + 
                                     2. * np.log(np.diag(L)).sum() + n_samples)

        elif self.llf_mode == 'noisy':
            theta, sigma2 = hyper_par[:-1], hyper_par[-1]
            noise_var = self.noise_var
            sigma2_total = sigma2 + noise_var

            R0 = self.correlation_matrix(theta)
            C = sigma2 * R0 + noise_var * np.eye(n_samples)
            R = C / sigma2_total

            try:
                L, Ft, Yt, Q, G, rho = self._compute_aux_var(R)
            except linalg.LinAlgError:
                if eval_grad:
                    return (log_likelihood, np.zeros(n_par, 1)) 
                else: 
                    return log_likelihood

            log_likelihood = -0.5 * (n_samples * log(2. * pi * sigma2_total) + 
                                     2. * np.log(np.diag(L)).sum() + 
                                     np.dot(rho.T, rho) / sigma2_total)

        if par_out is not None:
            par_out['sigma2'] = sigma2
            par_out['noise_var'] = noise_var
            par_out['rho'] = rho
            par_out['Yt'] = Yt
            # TODO: change variable 'C' --> 'L'
            par_out['C'] = L
            par_out['Ft'] = Ft
            par_out['G'] = G
            par_out['Q'] = Q

        # for verificationn
        # TODO: remove this in the future
        if np.exp(log_likelihood) > 1:
            return -np.inf, np.zeros((n_par, 1)) if eval_grad else -np.inf
            
        if not eval_grad:
            return log_likelihood[0]

        # gradient calculation of the log-likelihood
        gamma = linalg.solve_triangular(L.T, rho).reshape(-1, 1)
        
        Rinv = cho_solve((L, True), np.eye(n_samples))
        Rinv_upper = Rinv[np.triu_indices(n_samples, 1)]
        _upper = gamma.dot(gamma.T)[np.triu_indices(n_samples, 1)]

        llf_grad = np.zeros((n_par, 1))

        if self.llf_mode == 'noiseless':
            # The grad tensor of R w.r.t. theta
            R_grad_tensor = self.corr_grad_theta(theta, self.X, R0)

            for i in range(n_par):
                R_grad_upper = R_grad_tensor[:, :, i][np.triu_indices(n_samples, 1)]

                llf_grad[i] = np.sum(_upper * R_grad_upper) / sigma2 \
                    - np.sum(Rinv_upper * R_grad_upper)

        elif self.llf_mode == 'nugget_estim':
            # The grad tensor of R w.r.t. theta: note that the additional v below
            R_grad_tensor = alpha * self.corr_grad_theta(theta, self.X, R0)

            # partial derivatives w.r.t theta's
            for i in range(n_par - 1):
                R_grad_upper = R_grad_tensor[:, :, i][np.triu_indices(n_samples, 1)]

                # Note that sigma2_total is used here
                llf_grad[i] = np.sum(_upper * R_grad_upper) / sigma2_total \
                    - np.sum(Rinv_upper * R_grad_upper)

            # partial derivatives w.r.t 'v'
            R_dv = R0 - np.eye(n_samples)
            llf_grad[n_par - 1] = -0.5 * (np.sum(Rinv * R_dv) - 
                                          np.dot(gamma.T, R_dv.dot(gamma)) / sigma2_total)

        elif self.llf_mode == 'noisy':
            gamma_ = gamma / sigma2_total
            Cinv = Rinv / sigma2_total
            # Covariance: partial derivatives w.r.t. theta
            C_grad_tensor = sigma2_total * self.corr_grad_theta(theta, self.X, R0)
            
            # Covariance: partial derivatives w.r.t. sigma2
            C_grad_tensor = np.concatenate([C_grad_tensor, R0[..., np.newaxis]], axis=2)

            for i in range(n_par):
                C_grad = C_grad_tensor[:, :, i]
                llf_grad[i] = -0.5 * (np.sum(Cinv * C_grad) - 
                                      np.dot(gamma_.T, C_grad).dot(gamma_))
                
        return log_likelihood[0], llf_grad

    def _arg_max_log_likelihood_function(self):
        """
        This function estimates the autocorrelation parameters theta as the
        maximizer of the reduced likelihood function.
        (Minimization of the opposite reduced likelihood function is used for
        infoenience)

        Returns
        -------
        optimal_theta : array_like
            The best set of autocorrelation parameters (the sought maximizer of
            the reduced likelihood function).

        optimal_reduced_likelihood_function_value : double
            The optimal reduced likelihood function value.

        optimal_par : dict
            The BLUP parameters associated to thetaOpt.
        """

        if self.verbose:
            print "The chosen optimizer is: " + str(self.optimizer)
            print 'Log-likelihood mode: {}'.format(self.llf_mode)
            if self.random_start > 1:
                print "{} random restarts are specified.".format(self.random_start)
                      
        # setup the log10 search bounds
        bounds = np.c_[self.thetaL, self.thetaU]
        if self.llf_mode == 'nugget_estim':
            alpha_bound = np.atleast_2d([1e-10, 1.0 - 1e-10])
            bounds = np.r_[bounds, alpha_bound]

        elif self.llf_mode == 'noisy':
            # TODO: better estimation the upper and lowe bound of sigma2
            # TODO: implement optimization for the heterogenous case
            sigma2_upper = self.y.std() ** 2. - self.noise_var
            sigma2_bound = np.atleast_2d([1e-5, sigma2_upper])
            bounds = np.r_[bounds, sigma2_bound]

        log10bounds = log10(bounds)
        
        # if the model has been optimized before as the starting point,
        # then use the last optimized hyperparameters
        # supposed to be good for updating the model incrementally
        # TODO: validate this
        if hasattr(self, 'theta_'):
            log10theta0 = log10(self.theta_)
        else:
            log10theta0 = log10(self.theta0) if self.theta0 is not None else \
                np.random.uniform(log10(bounds)[:, 0], log10(bounds)[:, 1])

        log10param = log10theta0 if self.llf_mode not in ['nugget_estim', 'noisy'] else \
            np.r_[log10theta0, uniform(log10bounds[-1, 0], log10bounds[-1, 1])]
            
        optimal_par = {}
        n_par = len(log10param)
        # TODO: how to set this properly?
        eval_budget = 200 * n_par if self.eval_budget is None else self.eval_budget
        llf_opt = np.inf
            
        # a restarting L-BFGS method based on analytical gradient
        # TODO: maybe adopt an ILS-like restarting heuristic?
        if self.optimizer == 'BFGS':
            def obj_func(log10param):
                param = 10. ** np.array(log10param)
                __ = self.log_likelihood_function(param, eval_grad=True)
                return -__[0], -__[1] * param.reshape(-1, 1)
            
            wait_count = 0  # stagnation counter
            for iteration in range(self.random_start):
                if iteration != 0:
                    log10param = np.random.uniform(log10bounds[:, 0], log10bounds[:, 1])

                param_opt_, llf_opt_, info = fmin_l_bfgs_b(obj_func, log10param, 
                                                           bounds=log10bounds,
                                                           maxfun=eval_budget)
                
                diff = (llf_opt - llf_opt_) / max(abs(llf_opt_), abs(llf_opt), 1)
                if iteration == 0:
                    param_opt = param_opt_
                    llf_opt = llf_opt_    
                # TODO: verify this rule to determine the marginal improvement 
                elif diff >= 1e7 * MACHINE_EPSILON:
                    param_opt, llf_opt = param_opt_, llf_opt_
                    wait_count = 0
                else:
                    wait_count += 1

                if self.verbose:
                    print 'restart {} {} evals, best log likekihood value: {}'.format(iteration + \
                        1, info['funcalls'], -llf_opt)
                    if info["warnflag"] != 0:
                        warnings.warn("fmin_l_bfgs_b terminated abnormally with "
                                      "the state: {}".format(info))

                eval_budget -= info['funcalls']
                if eval_budget <= 0 or wait_count >= self.wait_iter:
                    break
                
        elif self.optimizer == 'CMA':   # IPOP-CMA-ES 
            def obj_func(log10param):
                param = 10. ** np.array(log10param)
                __ = self.log_likelihood_function(param)
                return __
            
            opt = {'sigma_init': 0.25 * np.max(log10bounds[:, 1] - log10bounds[:, 0]),
                   'eval_budget': eval_budget,
                   'f_target': np.inf,
                   'lb': log10bounds[:, 1],
                   'ub': log10bounds[:, 0],
                   'restart_budget': self.random_start}

            optimizer = cma_es(n_par, log10param, obj_func, opt, is_minimize=False,
                               restart='IPOP')
            param_opt, llf_opt, evalcount, info = optimizer.optimize()
            param_opt = param_opt.flatten()
            
            if self.verbose:
                print '{} evals, best log likekihood value: {}'.format(evalcount, -llf_opt)
                
        optimal_param = 10. ** param_opt
        optimal_llf_value = self.log_likelihood_function(optimal_param, optimal_par)

        if self.llf_mode in ['nugget_estim', 'noisy']:
            optimal_theta = optimal_param[:-1]
        else:
            optimal_theta = optimal_param

        return optimal_theta, optimal_llf_value, optimal_par

    def _check_params(self, n_samples=None):

        # Check regression model
#        if not callable(self.regr):
#            if self.regr in self._regression_types:
#                self.regr = self._regression_types[self.regr]
#            else:
#                raise ValueError("regr should be one of %s or callable, "
#                                 "%s was given."
#                                 % (self._regression_types.keys(), self.regr))

        # Check regression weights if given (Ordinary Kriging)
        if self.beta0 is not None:
            self.beta0 = np.atleast_2d(self.beta0)
            if self.beta0.shape[1] != 1:
                # Force to column vector
                self.beta0 = self.beta0.T

        # Check correlation model
        if not callable(self.corr):
            if self.corr in self._correlation_types:
                self.corr = self._correlation_types[self.corr]
            else:
                raise ValueError("corr should be one of %s or callable, "
                                 "%s was given." % (self._correlation_types.keys(), self.corr))

        # Check correlation parameters
        # self.theta0 = np.atleast_2d(self.theta0)
        lth = self.theta0.size

        if self.thetaL is not None and self.thetaU is not None:
            if self.thetaL.size != lth or self.thetaU.size != lth:
                raise ValueError("theta0, thetaL and thetaU must have the "
                                 "same length.")
            if np.any(self.thetaL <= 0) or np.any(self.thetaU < self.thetaL):
                raise ValueError("The bounds must satisfy O < thetaL <= "
                                 "thetaU.")

        elif self.thetaL is None and self.thetaU is None:
            if np.any(self.theta0 <= 0):
                raise ValueError("theta0 must be strictly positive.")

        elif self.thetaL is None or self.thetaU is None:
            raise ValueError("thetaL and thetaU should either be both or "
                             "neither specified.")

        # Force verbose type to bool
        self.verbose = bool(self.verbose)

        # Force normalize type to bool
        self.normalize = bool(self.normalize)

        # Check nugget value
        # self.nugget = np.asarray(self.nugget)
        # if np.any(self.nugget) < 0.:
        #     raise ValueError("nugget must be positive or zero.")
        # if (n_samples is not None
        #         and self.nugget.shape not in [(), (n_samples,)]):
        #     raise ValueError("nugget must be either a scalar "
        #                      "or array of length n_samples.")

        # Check optimizer
        if self.optimizer not in self._optimizer_types:
            raise ValueError("optimizer should be one of %s"
                             % self._optimizer_types)

        # Force random_start type to int
        self.random_start = int(self.random_start)
