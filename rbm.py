"""Boltzmann Base Module"""

import numpy as np
from scipy.special import expit


class BinaryRestrictedBoltzmannMachine(object):
    """ RestrictedBoltzmannInterface base class. """
    def __init__(
            self,
            n_components=256,
            learning_rate=0.1,
            n_iter=10,
            gibbs_steps=1,
            verbose=False,
            random_state=10
                 ):
        """Restricted Boltzmann Machines (RBM) implementation for Binary Data.
           The RBM is optimized by contrastive divergence, following the work
           of Hinton et al. "A Practical Guide to Training Restricted Boltzmann
           Machines"

        :param n_components: Number of binary hidden units, defaults to 256.
        :type n_components: int, optional.
        :param learning_rate: The learning rate for weight updates,
            defaults to 0.1.
        :type learning_rate: float, optional.
        :param n_iter: Number of iterations for training, defaults to 10.
        :type n_iter: int, optional.
        :param gibbs_steps: Number of gibbs steps to estimate the model data.
        :type gibbs_steps: int, optional.
        :param verbose: The verbosity level when printing in fit,
            default to False.
        :type verbose: bool.
        :param random_state: Random state for reproducibility.
        :type random_state: float.
        """
        # check consistency
        self._check_consistency(n_components, int, 'n_components')
        self._check_consistency(learning_rate, float, 'learning_rate')
        self._check_consistency(n_iter, int, 'n_iter')
        self._check_consistency(gibbs_steps, int, 'n_iter')
        self._check_consistency(verbose, bool, 'verbose')

        # assigning variables
        self._n_components = n_components
        self._learning_rate = learning_rate
        self._n_iter = n_iter
        self._gibbs_steps = gibbs_steps
        self._verbose = verbose
        
        # assign random state
        self._rng = np.random.RandomState(random_state)

    def fit(self, X):
        """Fit the model to the data X.

        :param X: Training data.
        :type X: np.ndarray.
        """
        # extract input features
        n_samples, dim = X.shape

        # initialize weights
        self.W = np.zeros(shape=(dim, self._n_components))
        self.a = np.zeros(shape=(dim, ))
        self.b = np.zeros(shape=(self._n_components, ))

        # perform n_iter steps
        for iter_ in range(self._n_iter):
            cd = self._fit_step(X)
            if self._verbose:
                print(f'Iteration {iter_ + 1}, CD = {cd}')

    def transform(self, X, probs=True):
        """Transform the model on the give data.

        :param X: Data to be transformed.
        :type X: np.ndarray
        :param probs: Returning probabilities instead of samples. This
            procedure is less prone to noise, defaults to True.
        :type probs: bool, optional
        :return: Transformed data.
        :rtype: np.ndarray
        """
        X_t = []
        # for each visible state
        for visible in X:
            # sample a hidden state
            latent = self._prob_h_given_v(visible)
            if not probs:
                latent = self._sample_binary_state(latent)
            X_t.append(latent)
        return np.stack(X_t)

    def inverse_transform(self, X, probs=True):
        """Inverse transform the model on the give data.

        :param X: Data to be transformed.
        :type X: np.ndarray
        :param probs: Returning probabilities instead of samples. This
            procedure is less prone to noise, defaults to True.
        :type probs: bool, optional
        :return: Transformed data.
        :rtype: np.ndarray
        """
        X_t = []
        # for each hidden state
        for hidden in X: 
            # sample a visible state
            visible = self._prob_v_given_h(hidden)
            if not probs:
                visible = self._sample_binary_state(visible)
            X_t.append(visible)
        return np.stack(X_t)

    def fit_transform(self, X):
        """Fit the model and then transform it.

        :param X: Training data.
        :type X: np.ndarray.
        :return: Transformed data.
        :rtype: np.ndarray
        """
        # fit the model
        self.fit(X)
        # transform the data
        X_t = self.transform(X)
        return X_t

    def _fit_step(self, visibles):
        """Fit step for the RBM.

        :param visibles: Visible variables.
        :type visibles: np.ndarray
        :return: Contrastive divergence.
        :rtype: float
        """
        # start the step
        energies = []
        for probs_v_pos in visibles:
            # sampling positive states (visible + hiddens)
            v_pos = self._sample_binary_state(probs_v_pos)
            probs_h_pos = self._prob_h_given_v(v_pos)
            h_pos = self._sample_binary_state(probs_h_pos)
            # sampling negative states (visible + hiddens)
            h_neg, v_neg, _, _ = self._gibbs_sampling(v_pos, self._gibbs_steps)
            # delta weigths
            dW = np.outer(v_pos, h_pos) - np.outer(v_neg, h_neg)
            da = v_pos - v_neg
            db = h_pos - h_neg
            # update weigths
            self.W += self.learning_rate*dW
            self.b += self.learning_rate*db
            self.a += self.learning_rate*da
            # calculate energies
            energy_pos = self.energy(v_pos, h_pos)
            energy_neg = self.energy(v_neg, h_neg) 
            energies.append((energy_pos, energy_neg))
        # averaging energies
        avg_pos_en, avg_neg_en = list(map(lambda y: sum(y) / len(y), zip(*energies)))
        return - avg_pos_en + avg_neg_en

    def _gibbs_sampling(self, visibles, gibbs_steps):
        """Perfoming Gibbs sampling.

        :param visibles: Visible unit variable
        :type visibles: np.ndarray
        """
        for _ in range(gibbs_steps):
            # getting probs hiddens
            probs_h = self._prob_h_given_v(visibles)
            # sample hiddens
            hiddens = self._sample_binary_state(probs_h)
            # getting probs visibles
            probs_v = self._prob_v_given_h(hiddens)
            # sample visibles
            visibles = self._sample_binary_state(probs_v)
        return hiddens, visibles, probs_h, visibles

    def _sample_binary_state(self, probs):
        """Sampling binary units according to specific probabilities.

        :param probs: Probabilities to sample 1.
        :type probs: np.ndarray
        :return: Binary units samples, with same shape as probabilities.
        :rtype: np.ndarray
        """
        return self._rng.binomial(n=1, p=probs)

    def _prob_h_given_v(self, visible):
        """Returns the probability of hidden vectors given visible vectors.

        :param visible: Visible vector input.
        :type visible: np.ndarray
        :return: Probabilities of hidden vectors given visible vectors.
        :rtype: np.ndarray
        """
        return expit(self.b + np.dot(visible, self.W))

    def _prob_v_given_h(self, hidden):
        """Returns the probability of hidden vectors given visible vectors.

        :param hidden: Visible vector input.
        :type hidden: np.ndarray

        :return: Probabilities of hidden vectors given visible vectors.
        :rtype: np.ndarray
        """
        return expit(self.a + np.dot(self.W, hidden))

    def energy(self, visible, hiddens):
        """Energy function of the model.

        :param visibles: visible units of the model.
        :type visibles: numpy.ndarray
        :param hiddens: hidden units of the model.
        :type hiddens: numpy.ndarray
        """
        return -np.dot(self.a, visible) -np.dot(self.b, hiddens) -np.dot(visible.T, np.dot(self.W, hiddens))

    def _check_consistency(self, value, object_type, name):
        """Simple function to check consistency. """
        if not isinstance(value, object_type):
            raise ValueError(f"{name} must be {object_type}")
        
    @property
    def n_components(self):
        """Property method n_components.

        :return: property object
        :rtype: float
        """
        return self._n_components

    @property
    def learning_rate(self):
        """Property method learning_rate.

        :return: property object
        :rtype: float
        """
        return self._learning_rate

    @property
    def n_iter(self):
        """Property method n_iter.

        :return: property object
        :rtype: int
        """
        return self._n_iter

    @property
    def gibbs_steps(self):
        """Property method gibbs_steps.

        :return: property object
        :rtype: int
        """
        return self._gibbs_steps
