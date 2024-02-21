#%% Imports
import numpy as np
import pandas as pd
from lsq import GaussianKDE, Mechanism, compare_mechanisms, LSQ_FGT

#%% Compare the mechanisms
if __name__ == '__main__':
    ### Usage example ###
    scipy_kde = Mechanism(mechanism_name = 'SCIPY')
    lsq_rff_kde = Mechanism(mechanism_name = 'LSQ_RFF', sanitized = False, mechanism_parameters = {'reps': 5000})
    lsq_rff_kde_dp = Mechanism(mechanism_name = 'LSQ_RFF', sanitized = True, mechanism_parameters = {'reps': 5000})
    lsq_fgt_kde = Mechanism(mechanism_name = 'LSQ_FGT', sanitized = False, mechanism_parameters = {'rho': 10})
    lsq_fgt_kde_dp = Mechanism(mechanism_name = 'LSQ_FGT', sanitized = True, mechanism_parameters = {'rho': 10})
    mechanisms = [scipy_kde, lsq_rff_kde, lsq_rff_kde_dp, lsq_fgt_kde, lsq_fgt_kde_dp]
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
data = pd.read_csv("~/test_data/edss_time_interval_COMPLETE_train.csv", 
                   sep = ";", 
                   decimal = ".",
                   parse_dates = [1])
# Generate random dataset and queries:
dimension = 1
coordinate_range = 2
n_data = 100000
n_queries = 1000
choices = np.array(range(coordinate_range))
probs = choices / np.sum(choices)
dataset = np.random.choice(choices, (n_data, dimension), p = probs)
queries = np.random.choice(choices, (n_queries, dimension), p = probs)
# paper config
dataset = np.random.uniform(0, coordinate_range-1, (n_data, dimension))
queries = np.random.uniform(0, coordinate_range-1, (n_queries, dimension))
queries = np.sort(queries)
lsq_fgt_test =  Mechanism(mechanism_name = 'LSQ_FGT', sanitized = False, mechanism_parameters = {'rho': 4}, domain_boundaries=(0, 13))
lsq_fgt_test.setup_mechanism(dataset)
# Exact Gaussian KDE:
exact_kde = GaussianKDE(dataset, queries)
# without my class
mechanism = LSQ_FGT(dataset, dimension, coordinate_range, 4)
non_dp_no_class = mechanism.non_private_kde(queries)
# Non-DP KDE estimates:
non_dp_kde_estimate = lsq_fgt_test.compute_pdfs_for_queries(queries)
print("Mean error:", np.mean(np.abs(exact_kde - non_dp_kde_estimate)))
# DP KDE estimates:
lsq_fgt_test.setup_mechanism(dataset, epsilon = 0.1)
dp_kde_estimate = lsq_fgt_test.compute_pdfs_for_queries(queries)
print("Mean error:", np.mean(np.abs(exact_kde - dp_kde_estimate)))

#%%
import matplotlib.pyplot as plt
plt.plot(queries[:, 0], non_dp_no_class, label = "Non DP (No class)")
plt.plot(queries[:, 0], exact_kde, label = "Exact")
plt.plot(queries[:, 0], non_dp_kde_estimate, label = "Non DP")
plt.plot(queries[:, 0], dp_kde_estimate, label = "DP")
plt.legend()
plt.show()
# %%
