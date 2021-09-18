import numpy as np
from tqdm import tqdm

################################################################################
### cached_property() - computed, cached as attribute, reset manually        ###
################################################################################

_NOT_FOUND = object()
class cached_property:
    def __init__(self, func, field_name="_cache"):
        self.func = func
        self.attrname = None
        self.field_name = field_name
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it.")
        try:
            instance_dict = instance.__dict__
        except AttributeError:  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None

        if self.field_name not in instance_dict:
            instance_dict[self.field_name] = dict()
        cache = instance_dict[self.field_name]
        
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            val = self.func(instance)
            try:
                cache[self.attrname] = val
            except TypeError:
                msg = (
                    f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                    f"does not support item assignment for caching {self.attrname!r} property."
                )
                raise TypeError(msg) from None
        return val


class ChioceModelEngine(object):
    def __init__(self, data, theta, weights=None):
        self.data_cooker(data, weights)
        self.new_theta(theta)

    def new_theta(self, value):
        self.theta_setter(value)
        self.cache_invalidator()
        
    def cache_invalidator(self):
        """Dropping all cached data if the model parameter changes"""
        self._cache = dict()
        
    def theta_setter(self, theta):
        if len(theta) != 3:
            raise ValueError("params argument should have 3 entries")

        assert theta[0].shape == (self.J, self.k_x)
        assert theta[1].shape == (1, self.k_z)
        assert theta[2].shape == (self.J, self.k_z)
        
        self.beta = theta[0]  # (J, k_x)
        self.gamma = theta[1]  # (1, k_z)
        self.d_gamma = theta[2]  # (J, k_z)
        

    def data_cooker(self, data, weights):
        if len(data) != 3:
            raise ValueError("data argument should have 3 entries")
    
        N, J, k_z = data[1].shape
        assert len(data[1].shape) == 3
        assert len(data[0].shape) == 3
        assert data[0].shape[0] == N
        assert data[0].shape[1] == 1
        assert data[2].shape == (N, J)            
        
        self.X = data[0]  # (N, 1, k_x)
        self.Z = data[1]  # (N, J, k_z)
        self.Y = data[2]  # (N, J)
        
        # Precomputing two products to save time later
        self.XX = np.expand_dims(self.X, axis=2) * np.expand_dims(self.X, axis=3)  # (N, 1, k_x, k_x)
        self.ZZ = np.expand_dims(self.Z, axis=2) * np.expand_dims(self.Z, axis=3)  # (N, J, k_z, k_z)
        
        self.N, self.J, self.k_z = data[1].shape
        self.k_x = data[0].shape[2]
        
        if weights is None:
            self.weights = None
        else:
            assert weights.shape == (N, 1)
            self.weights = weights/np.mean(weights)
            # TODO: COMPLETE THE IMPLEMENTATION OF WEIGHTS
        
    
    @cached_property
    def probabilities(self):
        """Estimated choice probabilities"""
        q_hat = (
            np.einsum("ijx,jx->ij", self.X, self.beta)
            + np.einsum("ijz,jz->ij", self.Z, self.gamma + self.d_gamma)
        )
        q_hat -= np.amax(q_hat, axis=1, keepdims=True)
        q_hat = np.exp(q_hat)
        return q_hat/np.sum(q_hat, axis=1, keepdims=True)

    @cached_property
    def score(self):
        """Log-likelihood of the model"""
        return np.mean(self.Y*np.log(self.probabilities)) * self.J


    @cached_property
    def comm_mult_first(self):
        """A multiplier that shows up in all first derivatives"""
        return np.expand_dims(self.Y-self.probabilities, axis=2)
    
    @cached_property
    def comm_mult_second(self):
        """A multiplier that shows up in all second derivatives"""
        return np.expand_dims(self.probabilities * (self.probabilities - 1), axis=(2, 3))

    
    @cached_property
    def beta_first(self):
        return np.mean(self.X * self.comm_mult_first, axis=0)
    
    @cached_property
    def beta_second(self):
        return np.mean(self.XX * self.comm_mult_second, axis=0)


    @cached_property
    def gamma_first(self):
        return np.sum(self.d_gamma_first, axis=0, keepdims=True)

    @cached_property
    def gamma_second(self):
        return np.sum(self.d_gamma_second, axis=0, keepdims=True)


    @cached_property
    def d_gamma_first(self):
        return np.mean(self.Z * self.comm_mult_first, axis=0)

    @cached_property
    def d_gamma_second(self):
        return np.mean(self.ZZ * self.comm_mult_second, axis=0)
    
    @cached_property
    def first_derivatives(self):
        """Returning all first derivatives"""
        return self.beta_first, self.gamma_first, self.d_gamma_first

    @cached_property
    def second_derivatives(self):
        """Returning all second derivatives"""
        return self.beta_second, self.gamma_second, self.d_gamma_second


class ChoiceModelEstimator(object):
    # TODO: indicate which stopping criterion worked.
    # TODO: pull out the variance of the estimates
    # TODO: punch a hole for weights for estimation.

    def __init__(
        self,
        data,
        initial_guess=None,
        learning_rate=0.1,
        L2_regularization=0.,
        stopping_criteria=None,
        display_tqdm=True,
    ):
        self.initialize_engine(data, initial_guess)
        
        if stopping_criteria is None:
            stopping_criteria = dict()
        self.stopping_criteria = {
            "max_iterations": 500,
            "relative_score_change": 1e-5,
            "score_derivative_norm_L1": 1e-4,  # L1-norm
        }
        self.stopping_criteria.update(stopping_criteria)
        
        self.learning_rate = learning_rate
        self.L2_reg = L2_regularization
        self._display_tqdm = display_tqdm
        
        self.optimization_log = list()


    @property
    def current_theta(self) -> tuple:
        return self._current_theta
    
    @current_theta.setter
    def current_theta(self, value):
        self._current_theta = value
        self.engine.new_theta(value)
        
        
    def initialize_engine(self, data, start) -> None:
        # All checks of data are performed on the engine's side
        _, J, k_z = data[1].shape
        k_x = data[0].shape[2]
        param_dims = [(J, k_x), (1, k_z), (J, k_z)]
        self.param_count = (J * (k_x+k_z)) + k_z
        
        if start is None:
            self.initial_guess = tuple(np.zeros(dim) for dim in param_dims)
        else:
            assert len(start) == 3
            assert all([s.shape == dim for s, dim in zip(start, param_dims)])
            assert np.allclose(np.mean(start[0], axis=0), np.zeros((k_x,)))
            assert np.allclose(np.mean(start[2], axis=0), np.zeros((k_z,)))
            self.initial_guess = tuple(start)
        
        self.engine = ChioceModelEngine(data, self.initial_guess)
        self.current_theta = self.initial_guess
        self.k_x, self.k_z, self.J = k_x, k_z, J
        

    def current_model_stats(self) -> dict:
        return {
            "score": self.engine.score,
            "beta_deriv_C1_norm": np.max(np.abs(self.engine.beta_first)),
            "gamma_deriv_C1_norm": np.max(np.abs(self.engine.gamma_first)),
            "d_gamma_deriv_C1_norm": np.max(np.abs(self.engine.d_gamma_first)),
            "theta": self.current_theta,
        }
        
    def add_log_line(self) -> None:
        self.optimization_log.append(self.current_model_stats().copy())

    def newton_step(self) -> None:
        newton_step = [
            np.einsum(
                "jk,jkz->jz",  # This is a layer-by-layer multipliction of colums by matrixes
                first_derivative - (2*self.L2_reg*param),
                np.linalg.inv(
                    second_derivative
                    - (2*self.L2_reg*np.expand_dims(np.eye(k), axis=0))
                )
            )
            for first_derivative, second_derivative, param, k in zip(
                (self.engine.beta_first, self.engine.gamma_first, self.engine.d_gamma_first),
                (self.engine.beta_second, self.engine.gamma_second, self.engine.d_gamma_second),
                self.current_theta,
                (self.k_x, self.k_z, self.k_z),
            )
        ]
        # Forcing renormalization for parameters that require it
        newton_step[0] -= np.mean(newton_step[0], axis=0, keepdims=True)
        newton_step[2] -= np.mean(newton_step[2], axis=0, keepdims=True)

        self.current_theta = tuple(
            param - (self.learning_rate*step)
            for param, step in zip(self.current_theta, newton_step)
        )
    
    def get_stopping_stats(self) -> dict:
        prev_score = self.optimization_log[-2]["score"]
        curr_score = self.engine.score
        
        first_derivatives = (
            self.engine.beta_first,
            self.engine.gamma_first,
            self.engine.d_gamma_first,
        )
        
        stopping_stats = dict()
        stopping_stats["relative_score_change"] = 2*(curr_score-prev_score)/abs(curr_score+prev_score)
        stopping_stats["score_derivative_norm_C1"] = max(map(lambda x: np.max(np.abs(x)), first_derivatives))
        stopping_stats["score_derivative_norm_L1"] = sum(
            map(lambda x: np.sum(np.abs(x)), first_derivatives)
        ) / self.param_count
        stopping_stats["score_derivative_norm_L2"] = np.sqrt(sum(
            map(lambda x: np.sum(np.square(x)), first_derivatives)
        ) / self.param_count)
        return stopping_stats
        
    def check_stopping(self) -> bool:
        stop = False
        stopping_stats = self.get_stopping_stats()
        for crit, thresh in self.stopping_criteria.items():
            if crit in stopping_stats:
                scaled_crit = stopping_stats[crit] / thresh
                stopping_stats[f"{crit}__scaled"] = scaled_crit
                stop = stop or (scaled_crit < 1.)
        self.optimization_log[-1].update(stopping_stats)
        return stop

    
    def get_optimization_iterator(self):
        iterator = range(self.stopping_criteria["max_iterations"])
        if self._display_tqdm:
            iterator = tqdm(iterator)
        return iterator

    def fit(self):
        self.add_log_line()
        for _ in self.get_optimization_iterator():
            self.newton_step()
            self.add_log_line()
            if self.check_stopping():
                break
