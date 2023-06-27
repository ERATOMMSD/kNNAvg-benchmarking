import numpy as np
from scipy.stats import norm


def mean_J(simulation_count, all_J_values):
    mean_Jv = 0
    for i in range(simulation_count):
        mean_Jv = (mean_Jv * i + all_J_values[i][0]) / (i + 1)

    return mean_Jv


def sigma_J(simulation_count, all_J_values):
    mean_Jv = mean_J(simulation_count, all_J_values)
    sigmasq_Jv = 0
    for i in range(simulation_count):
        diff = all_J_values[i][0] - mean_Jv
        sigmasq_Jv = (sigmasq_Jv * i + diff * diff) / (i + 1)
    return np.sqrt(sigmasq_Jv)


def sigma_J_unbiased(simulation_count, all_J_values):
    sigma_Jv = sigma_J(simulation_count, all_J_values)
    sigmasq_Jv = sigma_Jv * sigma_Jv
    sigmasq_Jv = (simulation_count * sigmasq_Jv) / (
        simulation_count - 1.0
    )  # unbiased
    return np.sqrt(sigma_Jv)


def stopping_lhs_alg1(simulation_count, all_J_values, simdic):
    M = simulation_count
    sigma_Jv = sigma_J_unbiased(simulation_count, all_J_values)
    eps = 10 ** -9
    Phi = 1.0
    if sigma_Jv >= eps:  # avoid division by Sigma_Jv = zero
        Phi = norm.cdf(np.sqrt(M) * simdic["TOL"] / sigma_Jv)
    return 2 * (1 - Phi)


#  Monte Carlo Stopping Criteria from
#  C. Bayer, H. Hoel, E. Von Schwerin, R. Tempone, "On Non-Asymptotic Optimal Stopping
#     Criteria in Monte Carlo Simulations", SIAM J. Sci. Comput, vol. 36, no. 2, 2014.
def find_simulation_count(simdic, compute_all_J_values):
    M0 = simdic["nSMIN"]
    all_J_values = compute_all_J_values(M0)
    M = M0
    total_M = M
    while stopping_lhs_alg1(M, all_J_values, simdic) > simdic["DELTA"]:
        if 2 * M > simdic["nSMAX"]:
            return (simdic["nSMAX"], total_M)
        M = 2 * M
        all_J_values = compute_all_J_values(M)
        total_M += M
    return (M, total_M)
