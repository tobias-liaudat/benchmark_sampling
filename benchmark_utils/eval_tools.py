import torch
import arviz as az
from statsmodels.tsa.stattools import acf
import numpy as np
from benchmark_utils import general_utils

#helper
def get_acf_index(acf_vec):
    smallest = float('inf')
    smallest_id = len(acf_vec)-1
    for i_ in range(len(acf_vec)):
        if acf_vec[i_] < smallest:
            smallest = acf_vec[i_]
            smallest_id = i_
        if acf_vec[i_] < 0:
            return i_ + 1
    return smallest_id + 1
            

# Computing ACF and ESS from a window of samples

#chain is a list of tensors, assuming gray images
def compute_acf_and_ess(chain):

    #create a tensor from the list
    # shape (window_len, batch_size, 1, dimx, dimy)
    chain_t_all = torch.stack(chain)

    # prep lists
    e_slow_list = []
    e_fast_list = []
    e_med_list = []
    lowest_median_acf_list = []
    lowest_slow_acf_list = []
    lowest_fast_acf_list = []

    # loop over the im stack (batch)
    for i_ in range(chain_t_all.shape[1]):

        chain_t = chain_t_all[:,i_,:,:,:].squeeze()

        # Fourier transform
        if general_utils.get_best_device() == "mps":
            chain_t = chain_t.cpu()
        X_chain_ft = torch.abs(torch.fft.fft2(chain_t))

        # --- Vectorise the the Markov chain
        X_chain_vec_ft = X_chain_ft.reshape(len(chain),-1)

        # --- Variance of the Markov chain
        var_sp = torch.var(X_chain_vec_ft, axis = 0)

        # compute the quantiles from var_sp
        q = torch.tensor([0.25, 0.5, 0.75])
        result = torch.quantile(var_sp, q.to(var_sp.device), interpolation='lower')

        # # --- Medium-speed trace of the Markov chain
        # now find the indexes of the quantile result
        ind_medium = torch.argwhere(var_sp == result[1])
        trace_elem_median_variance = X_chain_vec_ft[:,ind_medium]

        # --- 0.25 quantile of the Markov chain
        ind_low = torch.argwhere(var_sp == result[0])
        trace_elem_low_variance = X_chain_vec_ft[:,ind_low]

        # --- 0.75 quantile of the Markov chain
        ind_high = torch.argwhere(var_sp == result[2])
        trace_elem_high_variance = X_chain_vec_ft[:,ind_high]


        # --- effective sample size
        e_slow_list.append(az.ess(trace_elem_high_variance.reshape(-1).cpu().numpy()))
        e_fast_list.append(az.ess(trace_elem_low_variance.reshape(-1).cpu().numpy()))
        e_med_list.append(az.ess(trace_elem_median_variance.reshape(-1).cpu().numpy()))

        # --- Here we are generating the autocorrelation function for these three traces: lower, medium and faster.
        nLags = min(chain_t_all.shape[0], 50)
        median_acf = acf(trace_elem_median_variance.reshape(-1).cpu().numpy(), nlags=nLags)
        slow_acf = acf(trace_elem_high_variance.reshape(-1).cpu().numpy(), nlags=nLags)
        fast_acf = acf(trace_elem_low_variance.reshape(-1).cpu().numpy(), nlags=nLags)

        # now extract the index of the smallest value in the acf 
        # to determine the speed
        lowest_median_acf_list.append(get_acf_index(median_acf))
        lowest_slow_acf_list.append(get_acf_index(slow_acf))
        lowest_fast_acf_list.append(get_acf_index(fast_acf))

    
    return (np.mean(e_slow_list), 
            np.mean(e_med_list), 
            np.mean(e_fast_list), 
            np.mean(lowest_median_acf_list), 
            np.mean(lowest_slow_acf_list), 
            np.mean(lowest_fast_acf_list))

# testing
# import joblib

# path_base = "C:/Users/teresa-klatzer/OneDrive - University of Edinburgh/Research/poisson-pnp/"
# path_skrock = path_base + "PnP_ProxDRUNET_deblurringppnp_True_kernel_1_alpha_poisson_20_delta_frac50_noise_lvl_20_alpha_1_rot_n_flip_True/"
# chain_skrock_dict = joblib.load(path_skrock + "Reflected_PnP_DRUNET_MC_chain.joblib")
# chain_skrock = chain_skrock_dict['MC_chain'].squeeze()
# print(chain_skrock.shape)

# e_slow, e_med, e_fast, lowest_median_acf, lowest_slow_acf, lowest_fast_acf, median_acf, slow_acf, fast_acf = compute_acf_and_ess(chain_skrock)

