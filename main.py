#%% Imports
import numpy as np
import pandas as pd
from lsq import GaussianKDE, Mechanism, compare_mechanisms, LSQ_FGT

#%% Compare the mechanisms
if __name__ == '__main__':
    ### Usage example ###
    exact_kde = Mechanism(mechanism_name = 'Exact')
    lsq_rff_kde = Mechanism(mechanism_name = 'LSQ_RFF', sanitized = False, mechanism_parameters = {'reps': 5000})
    lsq_rff_kde_dp = Mechanism(mechanism_name = 'LSQ_RFF', sanitized = True, mechanism_parameters = {'reps': 5000})
    lsq_fgt_kde = Mechanism(mechanism_name = 'LSQ_FGT', sanitized = False, mechanism_parameters = {'rho': 10})
    lsq_fgt_kde_dp = Mechanism(mechanism_name = 'LSQ_FGT', sanitized = True, mechanism_parameters = {'rho': 10})
    mechanisms = [exact_kde, lsq_rff_kde, lsq_rff_kde_dp, lsq_fgt_kde, lsq_fgt_kde_dp]
    domain_sizes = [10**(i + 1) for i in range(4)]
    epsilons = [10**(1 - i) for i in range(5)]
    comparison = compare_mechanisms(mechanisms, domain_sizes, epsilons, n = 10**4)
    #comparison.to_csv("~/outputs/statistics/mechanism_comparison.csv", 
                      #sep = ";", index = False)

# %% Read the computed comparison
comparison = pd.read_csv("~/outputs/statistics/mechanism_comparison.csv", 
                         sep = ";", 
                         decimal = ".")

#%%
n = 10000
col = 'pi_max_gehstrecke'
minimum = 0
maximum = 13
epsilon = 1.0
bw = 'scott'
data = pd.read_csv("~/test_data/edss_time_interval_COMPLETE_train.csv", 
                    sep = ";", 
                    decimal = ".",
                    parse_dates = [1])

# make a dataset sample of pi_max_gehstrecke
np.random.seed(42)
sample = np.random.choice(data.index, n)
dataset = data.loc[sample, col]
queries = np.arange(data[col].min() - 1, data[col].max(), step = 0.125)
 
# Generate random dataset and queries:
# dimension = 1
# coordinate_range = 14
# n_data = 100000
# n_queries = 1000
# choices = np.array(range(coordinate_range))
# probs = choices / np.sum(choices)
# dataset = np.random.choice(choices, (n_data, dimension), p = probs)
# queries = np.random.choice(choices, (n_queries, dimension), p = probs)
# paper config
# dataset = np.random.uniform(0, coordinate_range-1, (n_data, dimension))
# queries = np.random.uniform(0, coordinate_range-1, (n_queries, dimension))
# queries = np.sort(queries, axis = 0)
# Scipy KDE:
scipy_kde_test = Mechanism(mechanism_name = 'SCIPY', mechanism_parameters = {'bw_method': bw}, domain_boundaries=(minimum, maximum))
scipy_kde_test.setup_mechanism(dataset)
scipy_kde_test_estimates = scipy_kde_test.compute_pdfs_for_queries(queries)
# Exact Gaussian KDE:
exact_kde_test = Mechanism(mechanism_name = 'Exact', mechanism_parameters = {'bandwidth': bw}, domain_boundaries=(minimum, maximum ))
exact_kde_test.setup_mechanism(dataset)
exact_kde_test_estimates = exact_kde_test.compute_pdfs_for_queries(queries)
# LSQ FGT KDE non-DP
lsq_fgt_test =  Mechanism(mechanism_name = 'LSQ_FGT', sanitized = False, mechanism_parameters = {'rho': 4, 'bandwidth': bw}, domain_boundaries = (minimum, maximum))
lsq_fgt_test.setup_mechanism(dataset)
lsq_fgt_test_estimates = lsq_fgt_test.compute_pdfs_for_queries(queries)
print("Mean error Non-DP:", np.mean(np.abs(exact_kde_test_estimates - lsq_fgt_test_estimates)))
# LSQ FGT KDE DP
lsq_fgt_test_dp =  Mechanism(mechanism_name = 'LSQ_FGT', sanitized = True, mechanism_parameters = {'rho': 4, 'bandwidth': bw}, domain_boundaries = (minimum, maximum))
lsq_fgt_test_dp.setup_mechanism(dataset, epsilon = epsilon)
lsq_fgt_test_dp_estimates = lsq_fgt_test_dp.compute_pdfs_for_queries(queries)
print("Mean error DP:", np.mean(np.abs(exact_kde_test_estimates - lsq_fgt_test_dp_estimates)))
#%%
import matplotlib.pyplot as plt
plt.plot(queries, scipy_kde_test_estimates, label = "SciPy")
plt.plot(queries, exact_kde_test_estimates, label = "Exact")
plt.plot(queries, lsq_fgt_test_estimates, label = "Non DP")
plt.plot(queries, lsq_fgt_test_dp_estimates, label = "DP")
plt.title(col)
plt.legend()
plt.show()
#%% Some quantiles
step = 0.05
U = np.arange(0, 1 + step, step)
# Non-DP
quantiles = lsq_fgt_test.percent_point(U)
# DP
dp_quantiles = lsq_fgt_test_dp.percent_point(U)
# Plot quantiles
import matplotlib.pyplot as plt
plt.plot(quantiles, U, label = "Non DP")
plt.plot(dp_quantiles, U, label = "DP")
plt.title("Quantiles")
plt.legend()
plt.show()
# %%
