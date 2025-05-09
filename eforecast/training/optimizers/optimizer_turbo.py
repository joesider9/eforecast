from copy import deepcopy

import numpy as np
import scipy.stats as ss
from eforecast.training.optimizers.turbo import Turbo1
from eforecast.training.optimizers.turbo.utils import from_unit_cube, latin_hypercube, to_unit_cube


def order_stats(X):
    _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)  # Need to do it this way due to ties
    o_stats = obs[idx]
    return o_stats


def copula_standardize(X):
    X = np.nan_to_num(np.asarray(X))  # Replace inf by something large
    assert X.ndim == 1 and np.all(np.isfinite(X))
    o_stats = order_stats(X)
    quantile = np.true_divide(o_stats, len(X) + 1)
    X_ss = ss.norm.ppf(quantile)
    return X_ss


class TurboOptimizer:
    primary_import = "Turbo"

    def __init__(self, api_config, **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """

        self.space_x = api_config
        self.bounds = np.array([self.space_x[key]['range'] for key in sorted(self.space_x.keys())])
        self.lb, self.ub = self.bounds[:, 0], self.bounds[:, 1]
        if not np.all(self.ub >= self.lb):
            print('HELPPPPPP')
        self.dim = len(self.bounds)
        self.max_evals = np.iinfo(np.int32).max  # NOTE: Largest possible int
        self.batch_size = None
        self.history = []

        self.turbo = Turbo1(
            f=None,
            lb=self.bounds[:, 0],
            ub=self.bounds[:, 1],
            n_init=2 * self.dim + 1,
            max_evals=self.max_evals,
            batch_size=1,  # We need to update this later
            verbose=False,
        )

    def restart(self, random_state):
        self.turbo._restart()
        self.turbo._X = np.zeros((0, self.turbo.dim))
        self.turbo._fX = np.zeros((0, 1))
        X_init = latin_hypercube(self.turbo.n_init, self.dim, random_state)
        self.X_init = from_unit_cube(X_init, self.lb, self.ub)

    def suggest(self, n_suggestions=1, random_state=None, warming=4):
        if self.batch_size is None:  # Remember the batch size on the first call to suggest
            self.batch_size = warming
            self.turbo.batch_size = warming
            self.turbo.failtol = np.ceil(np.max([4.0 / self.batch_size, self.dim / self.batch_size]))
            self.turbo.n_init = max([self.turbo.n_init, self.batch_size])
            self.restart(random_state)

        X_next = np.zeros((n_suggestions, self.dim))

        # Pick from the initial points
        n_init = min(len(self.X_init), n_suggestions)
        if n_init > 0:
            X_next[:n_init] = deepcopy(self.X_init[:n_init, :])
            self.X_init = self.X_init[n_init:, :]  # Remove these pending points

        # Get remaining points from TuRBO
        n_adapt = n_suggestions - n_init
        if n_adapt > 0:
            if len(self.turbo._X) > 0:  # Use random points if we can't fit a GP
                X = to_unit_cube(deepcopy(self.turbo._X), self.lb, self.ub)
                fX = copula_standardize(deepcopy(self.turbo._fX).ravel())  # Use Copula
                X_cand, y_cand, _ = self.turbo._create_candidates(
                    X, fX, length=self.turbo.length, n_training_steps=100, hypers={}
                )
                X_next[-n_adapt:, :] = self.turbo._select_candidates(X_cand, y_cand)[:n_adapt, :]
                X_next[-n_adapt:, :] = from_unit_cube(X_next[-n_adapt:, :], self.lb, self.ub)

        # Unwarp the suggestions
        suggestions = self.unwarp(X_next)
        return suggestions

    def unwarp(self, X_next):
        s = []
        for n in range(X_next.shape[0]):
            x_ = X_next[n]
            trial = dict()
            i = 0
            for key in sorted(self.space_x.keys()):
                trial[key] = x_[i]
                i+=1
            s.append(trial)
        return s

    def observe(self, X, y, random_state=None):
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        if y.shape[0] > 0:
            valid_id = np.where(np.isfinite(y))[0].tolist()
            X       = X.iloc[valid_id]
            y       = y[valid_id].reshape(-1, 1)
        assert len(X) == len(y)
        cols = sorted(X.columns)
        XX, yy = X[cols].values, np.array(y)[:, None]

        if len(self.turbo._fX) >= self.turbo.n_init:
            self.turbo._adjust_length(yy)

        self.turbo.n_evals += X.shape[0]

        self.turbo._X = deepcopy(XX)
        self.turbo._fX = deepcopy(yy)
        self.turbo.X = deepcopy(XX)
        self.turbo.fX = deepcopy(yy)

        # Check for a restart
        if self.turbo.length < self.turbo.length_min:
            self.restart(random_state)


