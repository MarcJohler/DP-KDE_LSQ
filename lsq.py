# Code for the paper "Fast Private Kernel Density Estimation via Locality Sensitive Quantization"
# By T. Wagner, Y. Naamad, N. Mishra
# Published in ICML 2023

# The code below is for the Gaussian kernel with fixed bandwidth, k(x,y) := exp(-||x-y||_2^2).
# The bandwidth can be changed by scaling the input point coordinates.

### Import and function defintions
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import gaussian_kde
import scipy.integrate as integrate
from scipy.optimize import minimize_scalar
import math

import differential_privacy

### Auxiliary : computing squared-distance matrix ###

def get_sqdistance_matrix(M1, M2):
    allsqnorms = np.linalg.norm(np.vstack([M1,M2]), axis=1).reshape(-1, 1)**2
    M1sqnorms = allsqnorms[:M1.shape[0],:]
    M2sqnorms = allsqnorms[M1.shape[0]:,:].reshape(1, -1)
    dm = M1sqnorms + M2sqnorms - 2.0 * np.dot(M1, M2.T)
    dm[dm < 0.0] = 0.0
    return dm


### Exact KDE ###
def GaussianKDE(dataset, queries):
    exp_sq_dist_matrix = np.exp(-1 * get_sqdistance_matrix(dataset, queries))
    return np.mean(exp_sq_dist_matrix, axis=0).T


### LSQ with Random Fourier Features ###

class LSQ_RFF:
    def __init__(self, dataset, dimension, reps):
        self.n = None
        self.d = dimension
        self.reps = reps

        # Sample random fourier features
        self.rff = np.sqrt(2) * np.random.normal(0, 1, (self.d, self.reps))
        self.rff_shift = np.random.uniform(0, 2*np.pi, self.reps).reshape(1, -1)

        self.rff_kde = None
        self.sanitized_rff_kde = None

        self.sketch_dataset(dataset)

    def sketch_dataset(self, dataset):
        self.n = dataset.shape[0]
        self.rff_kde = np.mean(self.apply_rff(dataset), axis=0)

    def apply_rff(self, m):
        return np.sqrt(2) * np.cos(np.dot(m, self.rff) + self.rff_shift)

    def sanitize(self, epsilon):
        self.sanitized_rff_kde = self.rff_kde + \
                                 np.random.laplace(0, np.sqrt(2) * self.reps * 1. / (epsilon * self.n), self.reps)

    def non_private_kde(self, queries):
        return (1./self.reps) * np.dot(self.rff_kde, self.apply_rff(queries).T)

    def private_kde(self, queries):
        return (1./self.reps) * np.dot(self.sanitized_rff_kde, self.apply_rff(queries).T)

    def one_number_kde(self, number, sanitized: bool):
        if sanitized:
            return self.private_kde(np.array(number))
        return self.non_private_kde(np.array(number))
    
    def kde_for_queries(self, queries, sanitized):
        if sanitized:
            return self.private_kde(queries)
        else:
            return self.non_private_kde(queries)

### LSQ with Fast Gauss Transform ###

class LSQ_FGT:
    def __init__(self, dataset, dimension, coordinate_range, rho):
        self.n = None
        self.d = dimension
        self.coordinate_range = coordinate_range
        self.small_radius_squared = rho
        self.rho = rho

        # Sketch
        self.sketch = np.zeros((self.coordinate_range ** self.d, self.rho ** self.d))
        self.sanitized_sketch = None

        # Sketch indexing auxiliaries
        self.aux_dim0_powers = np.flip(np.array([self.coordinate_range ** i for i in range(self.d)]))
        self.aux_dim0_tuples = np.indices(tuple([self.coordinate_range] * self.d)).reshape(self.d, -1).T
        self.aux_dim1_tuples = np.indices(tuple([self.rho] * self.d)).reshape(self.d, -1).T
        self.noise_scale = (2 * (1 - 0.5 ** self.rho)) ** self.d

        # Hermite polynomials
        self.hermite_polynomials = [sp.special.hermite(j) for j in np.arange(self.rho)]

        self.sketch_dataset(dataset)

    def sketch_dataset(self, dataset):
        self.n = dataset.shape[0]

        if self.rho == 0:
            return

        # Partition dataset into hypercubes
        rounded_dataset = np.rint(dataset)
        # Compute the index of the cell containing each data point
        cell_indices = rounded_dataset.dot(self.aux_dim0_powers).astype(np.dtype(int))

        if self.rho == 1:
            np.add.at(self.sketch[:, 0], cell_indices, np.ones(self.n))
        elif self.d == 2:
            residual_dataset = dataset - rounded_dataset
            all_powers = np.einsum('nk,nl->nkl',
                                   np.vstack([residual_dataset[:, 0]**j for j in range(self.rho)]).T,
                                   np.vstack([residual_dataset[:, 1]**j for j in range(self.rho)]).T
                                   ).reshape(self.n, -1)
            np.add.at(self.sketch, cell_indices, all_powers)
        else:
            residual_dataset = dataset - rounded_dataset
            for idx, idx_tuple in enumerate(self.aux_dim1_tuples):
                np.add.at(self.sketch[:, idx], cell_indices, np.prod(np.power(residual_dataset, idx_tuple), axis=1))

        self.sketch *= 1. / self.n

    def sanitize(self, epsilon):
        self.sanitized_sketch = self.sketch + \
                                np.random.laplace(0, self.noise_scale * 1. / (epsilon * self.n), self.sketch.shape)

    def g(self, query):

        # Compute the cells which are close enough to matter
        cell_distances = get_sqdistance_matrix(query.reshape(1, -1), self.aux_dim0_tuples)
        relevant_cells = np.where(cell_distances.ravel() <= self.small_radius_squared)[0]

        # Compute normalized hermite functions of all query residual coordinates in relevant cells
        # (Denominator turns hermite polynomial to hermite function)
        q_residuals = query - self.aux_dim0_tuples[relevant_cells, :]
        denominator = np.exp(q_residuals ** 2)
        hermite_evaluations = [(1. / np.math.factorial(j)) *
                               self.hermite_polynomials[j](q_residuals) / denominator for j in np.arange(self.rho)]

        # Compute g-coordinates of query
        q_sketch = np.zeros((len(relevant_cells), self.rho ** self.d))
        for cell_id in range(len(relevant_cells)):
            for idx, idx_tuple in enumerate(self.aux_dim1_tuples):
                q_sketch[cell_id, idx] = np.prod([hermite_evaluations[idx_tuple[i]][cell_id, i] for i in range(self.d)])

        return q_sketch, relevant_cells

    def one_query_kde(self, query, sanitized):
        if sanitized:
            dataset_sketch = self.sanitized_sketch
        else:
            dataset_sketch = self.sketch
        q_sketch, relevant_cells = self.g(query)
        return dataset_sketch[relevant_cells, :].ravel().dot(q_sketch.ravel())
    
    def one_number_kde(self, number: float, sanitized: bool):
        return self.one_query_kde(np.array(number), sanitized)

    def non_private_kde(self, queries):
        return np.array([self.one_query_kde(query, False) for query in queries])

    def private_kde(self, queries):
        return np.array([self.one_query_kde(query, True) for query in queries])
    
    def kde_for_queries(self, queries, sanitized):
        if sanitized:
            return self.private_kde(queries)
        else:
            return self.non_private_kde(queries)
        

class Mechanism:
    def __init__(self, mechanism_name = 'SCIPY', sanitized: bool = False, epsilon: float = 1.0,
                 domain_boundaries: tuple = (None, None), mechanism_parameters = None):
        # input checks
        assert isinstance(epsilon, float)
        assert isinstance(domain_boundaries, tuple)
        assert len(domain_boundaries) == 2
        # save mechanism info
        self.mechanism_name = mechanism_name
        if mechanism_name == 'SCIPY':
            self.mechanism_class = gaussian_kde
        elif mechanism_name == 'LSQ_RFF':
            self.mechanism_class = LSQ_RFF
        elif mechanism_name == 'LSQ_FGT':
            self.mechanism_class = LSQ_FGT
        self.sanitized = sanitized
        self.epsilon = epsilon
        self._x_lower_bound = domain_boundaries[0]
        self._x_upper_bound = domain_boundaries[1]
        self.mechanism_parameters = mechanism_parameters if mechanism_parameters is not None else {}
        self._pdf_multiplicator = 1.0
    
    def setup_mechanism(self, dataset, epsilon = None, domain_boundaries: tuple = None):
        # check if input is a pandas series
        if isinstance(dataset, pd.Series):
           dataset = np.array(dataset).reshape((len(dataset), 1)) 
        # save the dataset
        self._params = {
            'dataset': dataset,
        }
        # check if epsilon has been overwritten
        if epsilon:
            self.sanitized = True
            self.epsilon = epsilon
        # Reset the variables in case they were set before 
        self._x_lower_bound = None
        self._x_upper_bound = None
        self._x_range = None
        self._pdf_multiplicator = 1.0
        # compute the differentially private boundaries if the method shall apply sanitization
        if self.sanitized:
            # compute the differentially private min and max
            private_min, private_max = differential_privacy.compute_private_extremal_points(dataset, epsilon)   
            # define lower and upper bound of the dataset
            if domain_boundaries is not None:
                assert isinstance(domain_boundaries, tuple)
                assert len(domain_boundaries) == 2 
                if domain_boundaries[0] is None or domain_boundaries[1] is None:
                    std = differential_privacy.compute_private_std(dataset, (private_min, private_max), epsilon)
                self._x_lower_bound = private_min if domain_boundaries[0] is None else domain_boundaries[0]
                self._x_upper_bound = private_max if domain_boundaries[1] is None else domain_boundaries[1]
            else:
                std = differential_privacy.compute_private_std(dataset, (private_min, private_max), epsilon)
                self._x_lower_bound = private_min
                self._x_upper_bound = private_max
        else:
            self._x_lower_bound = dataset.min()
            self._x_upper_bound = dataset.max()
        # check if compute lower and upper bound are reasonable
        assert self._x_upper_bound > self._x_lower_bound
        self._x_range = self._x_upper_bound - self._x_lower_bound
        # normalize the dataset before fitting    
        dataset = dataset.copy()
        dataset = (dataset - self._x_lower_bound) / self._x_range
        # save the mechanism instance 
        if self.mechanism_name == 'SCIPY':
            self.mechanism_instance = self.mechanism_class(dataset.T, **self.mechanism_parameters)
        elif self.mechanism_name == 'LSQ_FGT':
            self.mechanism_instance = self.mechanism_class(dataset, 
                                                           dimension = 1, 
                                                           coordinate_range = 2,
                                                           **self.mechanism_parameters)
            if self.sanitized:
                self.mechanism_instance.sanitize(self.epsilon)
        elif self.mechanism_name == 'LSQ_RFF':
            self.mechanism_instance = self.mechanism_class(dataset, 
                                                           dimension = 1, 
                                                           **self.mechanism_parameters)
            if self.sanitized:
                self.mechanism_instance.sanitize(self.epsilon)
        # Now set it to the actual value
        self._pdf_multiplicator = 1.0 / self.compute_cdf_with_integral(self._x_upper_bound)
    
    def compute_pdfs_for_queries(self, queries):
        if self.mechanism_name == 'SCIPY':
            pdf = lambda x: self.mechanism_instance.pdf(x.T)
        else:
            pdf = lambda x: self.mechanism_instance.kde_for_queries(x, self.sanitized)
        return pdf((queries - self._x_lower_bound) / self._x_range) * self._pdf_multiplicator
        
    def compute_pdf(self, point):
        # if the domain is bounded return 0 for values outside the boundaries
        if point > self._x_upper_bound or point < self._x_lower_bound:
            return 0
        # otherwise compute pdf
        if self.mechanism_name == 'SCIPY':
            pdf = lambda x: self.mechanism_instance.pdf(np.array([x]))
        else:
            pdf = lambda x: self.mechanism_instance.one_number_kde(x, self.sanitized)
        return pdf((point - self._x_lower_bound) / self._x_range) * (self._pdf_multiplicator / self._x_range)
        
    def compute_cdf_with_integral(self, end, start = None):
        # check if the boundaries need to be tightened
        if start and start < self._x_lower_bound:
            start = self._x_lower_bound
        if end > self._x_upper_bound:
            end = self._x_upper_bound
        start = self._x_lower_bound if start is None else start
        compute_pdf = lambda x: self.compute_pdf(x)
        return integrate.quad(compute_pdf, start, end, limit = 1000, epsabs = 10**-6)[0]
    
    def inverse_cdf_with_integral(self, q: float):
        assert q >= 0 and q <= 1
        cdf_diff = lambda y: (self.compute_cdf_with_integral(y) - q)**2
        optimization_result = minimize_scalar(cdf_diff, bounds = (self._x_lower_bound, self._x_upper_bound), tol = 10**-6)
        return optimization_result.x
    
def compare_mechanisms(mechanisms: list, domain_sizess: list, epsilons: list, n: int):
    np.random.seed(42)
    sanitized = [mechanism.sanitized for mechanism in mechanisms]
    mech_col = [name for name_list in [[mechanism.mechanism_name] * len(epsilons) if mechanism.sanitized else [mechanism.mechanism_name] for mechanism in mechanisms] for name in name_list]
    san_col = [san for san_list in [[True] * len(epsilons) if s else [False] for s in sanitized] for san in san_list]
    eps_col = [eps for eps_list in [epsilons if s else [None] for s in sanitized] for eps in eps_list]    
    comparison = None
    for ds in np.sort(domain_sizess):
        comparison_for_ds = {'mechanism': mech_col,
                             'sanitized': san_col,
                             'epsilon': eps_col}
        choices = np.array(range(0, ds))
        # we normalize the dataset to speed up computation
        dataset = np.random.choice(choices, (n, 1)) 
        cdfs = []
        # loop through the mechanisms
        for mechanism in mechanisms:
            # if mechanism is not sanitized only apply one kde
            if not mechanism.sanitized:
                # setup the mechanism
                mechanism.setup_mechanism(dataset)
                # compute the cdf value at the maximum value
                cdfs.append(mechanism.compute_cdf_with_integral(ds  - 1, 0))
                continue 
            for epsilon in epsilons:
                # setup the mechanism
                mechanism.setup_mechanism(dataset, epsilon)
                # compute the cdf value at the maximum value
                cdfs.append(mechanism.compute_cdf_with_integral(ds  - 1, 0))
        comparison_for_ds['domain_size'] = ds
        comparison_for_ds['cdf'] = cdfs
        comparison_for_ds = pd.DataFrame.from_dict(comparison_for_ds)
        print(f"CDFs for domain size {ds} computed.")
        # add to overall comparison
        if comparison is None:
            comparison = comparison_for_ds
        else:
            comparison = pd.concat([comparison, comparison_for_ds], axis = 0, ignore_index = True)
    # also compute the cdfs for a continuous variable
    dataset = np.random.uniform(0, 1, (n, 1))
    cdfs = []
    comparison_for_continuous = {'mechanism': mech_col,
                                 'sanitized': san_col,
                                 'epsilon': eps_col}
    # loop through the mechanisms
    for mechanism in mechanisms:
        # if mechanism is not sanitized only apply one kde
        if not mechanism.sanitized:
            # setup the mechanism
            mechanism.setup_mechanism(dataset)
            # compute the cdf value at the maximum value
            cdfs.append(mechanism.compute_cdf_with_integral(1, 0))
            continue 
        # otherwise loop over epsilons
        for epsilon in epsilons:
            # setup the mechanism
            mechanism.setup_mechanism(dataset, epsilon)
            # compute the cdf value at the maximum value
            cdfs.append(mechanism.compute_cdf_with_integral(1, 0))
    comparison_for_continuous['domain_size'] = math.inf
    comparison_for_continuous['cdf'] = cdfs
    comparison_for_continuous = pd.DataFrame.from_dict(comparison_for_continuous)
    print(f"CDFs for continuous domain computed.")
    # merge into overall comparison
    if comparison is None:
        comparison = comparison_for_continuous
    else:
        comparison = pd.concat([comparison, comparison_for_continuous], axis = 0, ignore_index = True)
    return comparison

# %%
