# Code for the paper "Fast Private Kernel Density Estimation via Locality Sensitive Quantization"
# By T. Wagner, Y. Naamad, N. Mishra
# Published in ICML 2023

# The code below is for the Gaussian kernel with fixed bandwidth, k(x,y) := exp(-||x-y||_2^2).
# The bandwidth can be changed by scaling the input point coordinates.
#%%
import numpy as np
import scipy as sp
from scipy.integrate import quad

def kl_divergence(p, q, x_min, x_max):
    integrand = lambda x: p(x) * np.log(p(x) / q(x)) if q(x) > 0 else 0
    result, _ = quad(integrand, x_min, x_max)
    return result
    
def jensen_shannon_distance(p, q, x_min, x_max):
    # Compute the average distribution
    m = lambda x: 0.5 * (p(x) + q(x))

    # Calculate half of the Jensen-Shannon Divergence
    kl_pm = 0.5 * kl_divergence(p, m, x_min, x_max)
    kl_qm = 0.5 * kl_divergence(q, m, x_min, x_max)

    # Compute the Jensen-Shannon Distance
    js_distance = kl_pm + kl_qm

    return js_distance

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

    def non_private_kde(self, queries):
        return np.array([self.one_query_kde(query, False) for query in queries])

    def private_kde(self, queries):
        return np.array([self.one_query_kde(query, True) for query in queries])


if __name__ == '__main__':
    ### Usage example ###

    # Generate random dataset and queries:
    dimension = 1
    coordinate_range = 100
    n_data = 100000
    n_queries = 1000
    # paper config
    #dataset = np.random.uniform(0, coordinate_range-1, (n_data, dimension))
    #queries = np.random.uniform(0, coordinate_range-1, (n_queries, dimension))
    # test non uniform distribution
    choices = np.array(range(coordinate_range))
    weights = 2**(choices / 10)
    probs = weights / np.sum(weights)
    dataset = np.random.choice(choices, (n_data, dimension), p = probs)
    queries = np.random.choice(choices, (n_queries, dimension), p = probs)

    method = "lsq-fgt" # or "lsq-fgt"
    print("DP-KDE method:", method)

    if method == "lsq-rff":
        # Init LSQ-RFF:
        num_features = 200
        mechanism = LSQ_RFF(dataset, dimension, num_features)
    elif method == "lsq-fgt":
        # init LSQ-FGT:
        rho = 4
        mechanism = LSQ_FGT(dataset, dimension, coordinate_range, rho)

    # Exact Gaussian KDE:
    from scipy.stats import gaussian_kde
    exact_kde = GaussianKDE(dataset, queries)
    scipy = gaussian_kde(dataset.T)
    exact_kde2 = scipy.pdf(queries.T)
    print("Mean error:", np.mean(np.abs(exact_kde - exact_kde2)))
    bad_kde = lambda x: GaussianKDE(dataset, np.array(x))
    correction = (1 / quad(bad_kde, 0, coordinate_range - 1)[0])
    def better_kde(x):
        if x >= 0 and x <= coordinate_range - 1:
            return GaussianKDE(dataset, np.array(x)) * correction
        else:
            return 0
    print("KL divergence 1:", kl_divergence(better_kde, scipy.pdf, 0, coordinate_range - 1))
    print("KL divergence 2:", kl_divergence(scipy.pdf, better_kde, 0, coordinate_range - 1))
    print("Jensen shannon divergence:" , jensen_shannon_distance(better_kde, scipy.pdf, 0, coordinate_range - 1))
    

# %%
dataset = pd.read_csv("~/test_data/edss_time_interval_COMPLETE_train.csv", 
                   sep = ";", 
                   decimal = ".",
                   parse_dates = [1])['diff_to_next_pi_max_gehstrecke']
dataset = np.array(dataset).reshape((len(dataset), 1))

X = np.arange(-10, 10, 26 / 100)
X = X.reshape((len(X), 1))
from scipy.stats import gaussian_kde
exact_kde = GaussianKDE(dataset, X)
scipy = gaussian_kde(dataset.T)
exact_kde2 = scipy.pdf(X.T)
bad_kde = lambda x: GaussianKDE(dataset, np.array(x))
correction = (1 / quad(bad_kde, -13, 13)[0])
better_kde_data = GaussianKDE(dataset, X) * correction
scipy_kde_data = [scipy.pdf(x)[0] for x in X]
import pandas as pd
rel_frequencies = pd.Series(dataset[:,0]).value_counts(normalize = True)
import matplotlib.pyplot as plt
plt.plot(X, better_kde_data, label = "Paper")
plt.plot(X, scipy_kde_data, label = "scipy")
plt.scatter(rel_frequencies.keys().values, rel_frequencies.values, label = "freq")
plt.yscale('log') 
plt.legend()
plt.show()
# %%
