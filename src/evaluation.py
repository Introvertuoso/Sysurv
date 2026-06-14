import numpy as np
from copy import deepcopy
from tqdm import tqdm
from scipy.stats import norm
from src.methods import SySurv
from src.plotting import plot_validation

def build_dfd(dataset, config, n_subgroups=1, runs=1):
    # swap-randomization of the original data
    scores = []
    np.random.seed(config.seed)
    data_cpy = deepcopy(dataset)
    indices = np.arange(dataset.n_samples)
    perms = np.array([np.random.permutation(indices) for _ in range(runs)])

    pbar = tqdm(range(runs), unit="perm", mininterval=1, desc=f"Building DFD...")
    for step in pbar:
        data_cpy.time = data_cpy.time[perms[step]]
        data_cpy.event = data_cpy.event[perms[step]]
        method = SySurv(data_cpy, config, n_subgroups)
        method.run(progress=False)
        scores.append(max(method.best_exceptionalities)[0])
    pbar.close()

    return scores

def validate_subgroups(scores, n_subgroups, exceptionalities, plot=True):
    mu = np.mean(scores)
    sigma = np.std(scores)
    nominal_alpha = 0.05
    corrected_alpha = nominal_alpha / n_subgroups # Bonferroni correction
    cutoff = norm.isf(corrected_alpha, mu, sigma)
    
    exc_values = [e.item() if hasattr(e, 'item') else e for e in exceptionalities]
    p_values = [float(norm.sf((e - mu) / sigma)) for e in exc_values]
    is_significant = [bool(e > cutoff) for e in exc_values]
    
    if plot:
        plot_validation(scores, mu, sigma, exc_values, p_values, corrected_alpha, cutoff)
        
    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "p_values": p_values,
        "corrected_alpha": corrected_alpha,
        "cutoff": cutoff,
        "is_significant": is_significant
    }