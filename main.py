#%% Imports
import pandas as pd
from lsq import Mechanism, compare_mechanisms

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
    epsilons = [10**(1 - i) for i in range(6)]
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
lsq_fgt_test =  Mechanism(mechanism_name = 'LSQ_FGT', sanitized = False, mechanism_parameters = {'rho': 10})
lsq_fgt_test.setup_mechanism(data['pi_max_gehstrecke'])

# %%
lsq_fgt_kde_dp.inverse_cdf_with_integral(1)