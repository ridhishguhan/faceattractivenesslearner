import numpy as np

from Utils import logsumexp
EPS = np.finfo(float).eps


def log_multivariate_normal_density(X, means, covars, covariance_type='full'):
    log_multivariate_normal_density_dict = {
        'diag': _log_multivariate_normal_density_diag,
        'full': _log_multivariate_normal_density_full}
    return log_multivariate_normal_density_dict[covariance_type](
        X, means, covars)

class GMM():
    def __init__(self, n_components=1, covariance_type='diag',
                 thresh=1e-2, min_covar=1e-3,
                 n_iter=100, n_init=1, params='wmc', init_params='wmc'):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.thresh = thresh
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params

        if not covariance_type in ['diag', 'full']:
            raise ValueError('Invalid value for covariance_type: %s' %
                            covariance_type)

        if n_init < 1:
            raise ValueError('GMM estimation requires at least one run')

        self.weights_ = np.ones(self.n_components) / self.n_components

        # flag to indicate exit status of fit() method: converged (True) or
        # n_iter reached (False)
        self.converged_ = False

    def _get_covars(self):
        """Covariance parameters for each mixture component.
        The shape depends on `cvtype`::
            (`n_states`, `n_features`)                if 'diag',
            (`n_states`, `n_features`, `n_features`)  if 'full'
            """
        if self.covariance_type == 'full':
            return self.covars_
        elif self.covariance_type == 'diag':
            return [np.diag(cov) for cov in self.covars_]

    def _set_covars(self, covars):
        """Provide values for covariance"""
        covars = np.asarray(covars)
        _validate_covars(covars, self.covariance_type, self.n_components)
        self.covars_ = covars

    def eval(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError('the shape of X  is not compatible with self')

        lpr = (log_multivariate_normal_density(
                X, self.means_, self.covars_, self.covariance_type)
               + np.log(self.weights_))
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities

    def score(self, X):
        logprob, _ = self.eval(X)
        return logprob

    def predict(self, X):
        logprob, responsibilities = self.eval(X)
        return responsibilities.argmax(axis=1)

    def predict_proba(self, X):
        logprob, responsibilities = self.eval(X)
        return responsibilities


    def fit(self, X):
        """Estimate model parameters with the expectation-maximization
        algorithm.

         Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        """
        ## initialization step
        X = np.asarray(X, dtype=np.float)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.shape[0] < self.n_components:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                (self.n_components, X.shape[0]))

        max_log_prob = -np.infty
        count = 0

        for _ in range(self.n_init):

            # EM algorithms
            log_likelihood = []
            # reset self.converged_ to False
            self.converged_ = False
            for i in xrange(self.n_iter):
                # Expectation step
                print "EM Algorithm Iteration : ", i+1," / ",self.n_iter
                curr_log_likelihood, responsibilities = self.eval(X)
                log_likelihood.append(curr_log_likelihood.sum())

                # Check for convergence.
                if i > 0 and abs(log_likelihood[-1] - log_likelihood[-2]) < \
                        self.thresh:
                    count += 1
                    if count > 10:
                        self.converged_ = True
                        break
                else:
                    count = 0

                # Maximization step
                self._do_mstep(X, responsibilities, self.params,
                        self.min_covar)

            # if the results are better, keep it
            if self.n_iter:
                if log_likelihood[-1] > max_log_prob:
                    max_log_prob = log_likelihood[-1]
                    best_params = {'weights': self.weights_,
                                   'means': self.means_,
                                   'covars': self.covars_}
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "or check for degenerate data.")
        if self.n_iter:
            self.max_log_prob = max_log_prob
            self.covars_ = best_params['covars']
            self.means_ = best_params['means']
            self.weights_ = best_params['weights']
        return self

    def _do_mstep(self, X, responsibilities, params, min_covar=0):
        #print "MSTEP"
        """ Perform the Mstep of the EM algorithm and return the class weights.
        """
        weights = responsibilities.sum(axis=0)
        weighted_X_sum = np.dot(responsibilities.T, X)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

        if 'w' in params:
            self.weights_ = (weights / (weights.sum() + 10 * EPS) + EPS)
        if 'm' in params:
            self.means_ = weighted_X_sum * inverse_weights
        if 'c' in params:
            covar_mstep_func = _covar_mstep_funcs[self.covariance_type]
            self.covars_ = covar_mstep_func(
                self, X, responsibilities, weighted_X_sum, inverse_weights,
                min_covar)
        return weights

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        ndim = self.means_.shape[1]
        if self.covariance_type == 'full':
            cov_params = self.n_components * ndim * (ndim + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * ndim
        elif self.covariance_type == 'tied':
            cov_params = ndim * (ndim + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = ndim * self.n_components
        return  int(cov_params + mean_params + self.n_components - 1)

def _log_multivariate_normal_density_diag(X, means=0.0, covars=1.0):
    """Compute Gaussian log-density at X for a diagonal model"""
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
    return lpr


def _log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-7):
    from scipy import linalg
    import itertools
    if hasattr(linalg, 'solve_triangular'):
        solve_triangular = linalg.solve_triangular
    else:
        solve_triangular = linalg.solve
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(itertools.izip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:

            cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                      lower=True)
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) + \
                                     n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob


def _validate_covars(covars, covariance_type, n_components):
    """Do basic checks on matrix covariance sizes and values
    """
    from scipy import linalg
    if covariance_type == 'diag':
        if len(covars.shape) != 2:
            raise ValueError("'diag' covars must have shape"
                             "(n_components, n_dim)")
        elif np.any(covars <= 0):
            raise ValueError("'diag' covars must be non-negative")
    elif covariance_type == 'full':
        if len(covars.shape) != 3:
            raise ValueError("'full' covars must have shape "
                             "(n_components, n_dim, n_dim)")
        elif covars.shape[1] != covars.shape[2]:
            raise ValueError("'full' covars must have shape "
                             "(n_components, n_dim, n_dim)")
        for n, cv in enumerate(covars):
            if (not np.allclose(cv, cv.T)
                or np.any(linalg.eigvalsh(cv) <= 0)):
                raise ValueError("component %d of 'full' covars must be "
                                 "symmetric, positive-definite" % n)
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")


def _covar_mstep_diag(gmm, X, responsibilities, weighted_X_sum, norm,
                      min_covar):
    """Performing the covariance M step for diagonal cases"""
    avg_X2 = np.dot(responsibilities.T, X * X) * norm
    avg_means2 = gmm.means_ ** 2
    avg_X_means = gmm.means_ * weighted_X_sum * norm
    return avg_X2 - 2 * avg_X_means + avg_means2 + min_covar


def _covar_mstep_full(gmm, X, responsibilities, weighted_X_sum, norm,
                      min_covar):
    """Performing the covariance M step for full cases"""
    n_features = X.shape[1]
    cv = np.empty((gmm.n_components, n_features, n_features))
    for c in xrange(gmm.n_components):
        post = responsibilities[:, c]
        np.seterr(under='ignore')
        avg_cv = np.dot(post * X.T, X) / (post.sum() + 10 * EPS)
        mu = gmm.means_[c][np.newaxis]
        cv[c] = (avg_cv - np.dot(mu.T, mu) + min_covar * np.eye(n_features))
    return cv


_covar_mstep_funcs = {'diag': _covar_mstep_diag,
                      'full': _covar_mstep_full,
                      }