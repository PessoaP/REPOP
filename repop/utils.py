import torch
from scipy.special import gammaln
import numpy as np
from numpy import log, pi

# General mathematical utilities for REPOP
# Precompute constant values used in the Gaussian likelihood function.
lsqrt2pi = (1 / 2) * log(2 * pi)
l10 = log(10)

# Define lambda functions for common probability calculations.
# log_comb computes the log of the binomial coefficient.
def log_comb(n, k):
    # n, k can be integer tensors; mask uses integer logic
    invalid = (k > n) | (k < 0)

    # cast for lgamma
    n = n.to(torch.float64)
    k = k.to(torch.float64)

    val = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    return torch.where(invalid, torch.full_like(val, -torch.inf), val)

# binomial_loglike computes the log likelihood for a binomial outcome.
binomial_loglike = lambda k, n, p: log_comb(n, k) + k * torch.log(p) + (n - k) * torch.log(1 - p)
# gaussian_loglike computes the log likelihood of a Gaussian given data x, mean mu, and std dev sig.
gaussian_loglike = lambda x, mu, sig: - torch.pow(((x - mu) / sig), 2) / 2 - torch.log(sig) - lsqrt2pi



# Simple normalization function.
normalize = lambda x: x / x.sum()

def logm1exp(x):
    """
    Numerically stable computation for log(1 - exp(x)).
    """
    mask = (x > -1)
    res = torch.zeros_like(x)
    res[mask] += torch.log(-torch.expm1(x[mask]))
    res[~mask] += torch.log1p(-torch.exp(x[~mask]))
    return res

def Igaussmix_loglike(n, mus, sigs, rhos):
    """
    Computes the normalized log likelihood over values `n` for a Gaussian mixture model.

    Parameters:
      n: tensor of values to evaluate (e.g., indices or data points)
      mus: means of the Gaussian components (shape: [components])
      sigs: standard deviations of the components (shape: [components])
      rhos: mixture weights (shape: [components])

    Returns:
      lpn: log probability at each n, normalized to sum to one (shape: [n])
    """
    # Reshape mus and sigs to shape [components, 1] so that broadcasting works:
    # each row is a component, each column is a value of `n`
    terms_unorm = gaussian_loglike(n, mus.reshape(-1, 1), sigs.reshape(-1, 1))

    # Normalize each row (i.e., each component) across n values
    terms = terms_unorm - torch.logsumexp(terms_unorm, axis=1).reshape(-1, 1)

    # Add log of mixing weights (reshaped to [components, 1] to match n columns)
    lpn_unorm = torch.logsumexp(terms + torch.log(rhos.reshape(-1, 1)), axis=0)

    # Normalize final log likelihood across all n (log-probability over `n`)
    lpn = lpn_unorm - torch.logsumexp(lpn_unorm, axis=0)

    return lpn


def sample_from_logps(logps, n_samples=10, pseudo_sampling=True):
    """
    logps: (N, K) tensor of unnormalized log-probabilities
    Returns: (N, n_samples) tensor of sampled category indices
    """
    N, K = logps.shape

    # normalize log-probs (stable)
    logps_stable = logps - logps.max(dim=1, keepdim=True).values
    probs = torch.exp(logps_stable)
    probs = probs / probs.sum(dim=1, keepdim=True)

    # cumulative distribution per row
    cum_probs = probs.cumsum(dim=1)

    if pseudo_sampling:
        # deterministic pseudo-uniforms
        u = torch.linspace(0, 1, n_samples + 2, device=logps.device)[1:-1]  # (n_samples,)
        U = u.expand(N, n_samples)  # broadcast
    else:
        # random uniforms, sorted per row
        U = torch.rand((N, n_samples), device=logps.device)
        U, _ = torch.sort(U, dim=1)

    # inverse-CDF: first index where CDF >= u_j
    # cum_probs: (N, K)
    # U:         (N, n_samples)
    samples = ((cum_probs.unsqueeze(2) >= U.unsqueeze(1)).int()).argmax(dim=1)

    return samples
