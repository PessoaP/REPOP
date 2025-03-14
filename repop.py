# Import necessary libraries
import torch
from numpy import sqrt, argsort, random, unique, log10, arange, log, pi
random.seed(42) 
from sklearn.mixture import GaussianMixture  # For the naive fitting of a Gaussian mixture model
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoLocator
import warnings

# Precompute constant values used in the Gaussian likelihood function.
lsqrt2pi = (1 / 2) * log(2 * pi)
l10 = log(10)

# Define lambda functions for common probability calculations.
# log_comb computes the log of the binomial coefficient.
log_comb = lambda n, k: torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
# binomial_loglike computes the log likelihood for a binomial outcome.
binomial_loglike = lambda k, n, p: log_comb(n, k) + k * torch.log(p) + (n - k) * torch.log(1 - p)
# gaussian_loglike computes the log likelihood of a Gaussian given data x, mean mu, and std dev sig.
gaussian_loglike = lambda x, mu, sig: - torch.pow(((x - mu) / sig), 2) / 2 - torch.log(sig) - lsqrt2pi
# poisson_loglike computes the log likelihood for a Poisson outcome.
poisson_loglike = lambda k, rate: k * torch.log(rate) - rate - torch.lgamma(k + 1)

# Set a weak limit constant, used later in parameter estimation.
weak_limit = 25

# Simple normalization function.
normalize = lambda x: x / x.sum()

def counts_loglike(k, n, phi):
    """
    Computes the log likelihood for binomial counts given a dilution factor phi.
    """
    lp_bin = binomial_loglike(k, n, 1. / phi)
    return lp_bin

def Igaussmix_loglike(n, mus, sigs, rhos):
    """
    Computes the log-likelihood of a Gaussian mixture model at values n.

    Parameters:
      n: tensor of indices at which to evaluate the likelihood.
      mus: tensor of means for each mixture component.
      sigs: tensor of standard deviations for each component.
      rhos: tensor of mixing proportions for each component.

    Returns:
      lpn: tensor of log-likelihoods normalized to sum to one.
    """
    # Compute unnormalized log likelihood for each component over n.
    terms_unorm = gaussian_loglike(n, mus.reshape(-1, 1), sigs.reshape(-1, 1))
    # Normalize the likelihood of each component over n.
    terms = terms_unorm - torch.logsumexp(terms_unorm, axis=1).reshape(-1, 1)
    # Compute the overall log likelihood for each n via a weighted sum over components.
    lpn_unorm = torch.logsumexp(terms + torch.log(rhos.reshape(-1, 1)), axis=0)
    # Normalize the overall likelihood so that its exponential sums to one.
    lpn = lpn_unorm - torch.logsumexp(lpn_unorm, axis=0)


    return lpn

def theta2params(theta, components=weak_limit):
    """
    Convert the optimization vector theta into mixture parameters (means, sigmas, weights).

    The vector theta is assumed to contain:
       - First 'components' entries: log(means)
       - Next 'components' entries: log(sigmas)
       - Remaining entries: log(weights)

    Returns:
      mus, sigs, and normalized rhos.
    """
    mus = torch.exp(theta[:components])
    sigs = torch.exp(theta[components:2*components])
    rhos = torch.exp(theta[2*components:])
    return mus, sigs, normalize(rhos)

def params2theta(mus, sigs, rhos):
    """
    Convert mixture parameters into a single theta vector by taking logarithms.
    """
    return torch.hstack((torch.log(mus),
                         torch.log(sigs),
                         torch.log(rhos)))

def logm1exp(x):
    """
    Numerically stable computation for log(1 - exp(x)).
    """
    mask = (x > -1)
    res = torch.zeros_like(x)
    res[mask] += torch.log(-torch.expm1(x[mask]))
    res[~mask] += torch.log1p(-torch.exp(x[~mask]))
    return res

def dils_switch(dils, N, threshold):
    """
    Compute correction terms for the dilution process using a joint binomial framework.

    For each unique dilution value in 'dils', this function computes:
      - binomial_logZ: a log partition function (clamped to be at most 0)
      - pdils: cumulative probability terms from previous dilutions.

    Returns:
      A tuple (logZdils, pdils) with corrections for each data point.
    """
    n = torch.arange(N).to(dils.device)
    k = torch.arange(threshold + 1).reshape(-1, 1).to(dils.device)

    dils_unique, inverse = torch.unique(dils, return_inverse=True, sorted=True)
    dils_num = dils_unique.size(0)
    logZdils, pdils = [], []
    lp_antes = torch.zeros_like(n).float()

    for i in range(dils_num):
        d = dils_unique[i]
        binomial_logZ = torch.clamp(torch.logsumexp(counts_loglike(k, n, d), axis=0), max=0)
        logZdils.append(binomial_logZ)
        pdils.append(binomial_logZ + lp_antes)
        lp_antes += logm1exp(binomial_logZ)

    return torch.vstack(logZdils)[inverse.reshape(-1)], torch.vstack(pdils)[inverse.reshape(-1)]

def get_lpkdil_n(counts,dils,n,threshold,Nmax):
    # Compute the log likelihood for the counts/dilution pairs using either full n range or threshold.
    if threshold == -1:
        return counts_loglike(counts, n, dils)
    else:
        lpk_diln_unnorm = counts_loglike(counts, n, dils)
        logZ, lpdil_n = dils_switch(dils, Nmax, threshold)
        lpk_diln = lpk_diln_unnorm - logZ + lpdil_n
        return lpk_diln
        

# The dataset class encapsulates the data along with methods for estimating and reconstructing
# the underlying bacterial population distribution from plate counts.
class dataset():
    def __init__(self, counts, dils, threshold=-1):
        # Use GPU if available.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Convert counts and dilutions to column tensors.
        self.counts = torch.tensor(counts.reshape(-1, 1))
        self.dils = torch.tensor(dils.reshape(-1, 1))

        self.ndatapoints = self.counts.size(0)
        self.threshold = threshold

        # Compute the maximum likelihood (naive) estimate: counts multiplied by the dilution factors.
        self.ML = (counts * dils).clip(min=1).reshape(-1, 1)
        self.Nmin = 1
        self.Nmax = 2 * self.ML.max() + 1
        self.width = torch.tensor(self.Nmax, device=self.device)
        self.n = torch.arange(self.Nmax)

        #self.lpkdil_n = get_lpkdil_n(self.counts,self.dils,self.n,threshold,self.Nmax).to(self.device)
        self.n = self.n.to(self.device)        

        # Set the weak limit based on the number of datapoints.
        self.weaklimit = min(weak_limit, int(sqrt(self.counts.numel())))

    def lprior(self, mus, sigs, rhos, components=weak_limit):
        """
        Compute a log-prior over the mixture parameters.
        A weak prior is imposed on the means (after scaling) and sigmas.
        """
        mus_shifted = (mus - self.Nmin) / self.width
        lp = 0.1 * (torch.log(mus_shifted) + torch.log(1 - mus_shifted))
        lp += gaussian_loglike(torch.log(sigs), torch.log(mus), torch.ones_like(mus)) - torch.log(sigs)
        if components != 1:
            return self.rhosprior.log_prob(rhos) + lp.sum()
        return lp.sum()

    def loglike(self, theta, components, total=False):
        """
        Compute the log likelihood of the data given the mixture parameters encoded in theta.
        """
        mus, sigs, rhos = theta2params(theta, components)
        lpn_th = Igaussmix_loglike(self.n, mus, sigs, rhos)
        lpkn_th = lpn_th + self.lpkdil_n
        ans = torch.logsumexp(lpkn_th, axis=1)
        if total:
            return ans.sum()
        return ans

    def ML_estimate(self, components=weak_limit):
        """
        Estimate mixture parameters from the naive maximum likelihood data using a Gaussian Mixture Model.
        """
        gmm = GaussianMixture(n_components=components, covariance_type='full')
        gmm.fit(self.ML)

        prov_mus = gmm.means_.reshape(-1)
        prov_sigs = sqrt(gmm.covariances_).reshape(-1)
        prov_rhos = gmm.weights_

        # Sort the estimated parameters by their weights in descending order.
        indices = argsort(-prov_rhos)
        prov_mus, prov_sigs, prov_rhos = prov_mus[indices], prov_sigs[indices], prov_rhos[indices]

        self.ML_estimated = (torch.tensor(prov_mus), torch.tensor(prov_sigs), torch.tensor(prov_rhos))
        return self.ML_estimated

    def evaluate(self, components=weak_limit, tol=1e-5, lr=0.01, observe=False, dir_factor=0.9, component_cut=1/50):
        """
        Optimize the mixture model parameters (theta) by maximizing the data log-likelihood plus prior.
        Uses an Adam optimizer and periodically reorders the parameters.
        """
        self.lpkdil_n = get_lpkdil_n(self.counts.to(self.device),
                                     self.dils.to(self.device),
                                     self.n.to(self.device),
                                     self.threshold,self.Nmax).to(self.device)

        if components == weak_limit:
            components = self.weaklimit #If no number of components where specified it will change the smallest number of components between the default and the square root of the number of datapoind
        if not torch.cuda.is_available():
            warnings.warn( "CUDA-compatible GPU not detected. REPOP is optimized for working on GPU, and performance may be significantly slower on a CPU." )

        #print(components)
        # Define a Dirichlet prior on the mixture weights.
        self.alpha = normalize((dir_factor) ** (torch.arange(components)))
        self.alpha *= 2.1 / self.alpha[-1]
        self.rhosprior = torch.distributions.Dirichlet(self.alpha.to(self.device))

        self.ML_estimate(components)
        th = params2theta(*self.ML_estimated).to(self.device).requires_grad_(True)
        self.ev = theta2params(th.clone().detach(), components)

        # Define the loss as the negative log posterior normalized by the number of data points.
        loss = lambda x: -self.loglike(x, components).mean() - self.lprior(*theta2params(x, components), components) / self.ndatapoints
        optimizer = torch.optim.Adam([th], lr=lr)
        torch.autograd.set_detect_anomaly(True)

        keep = True
        i = 0
        while keep:
            l = loss(th)
            if ~(torch.isnan(l) | torch.isinf(l)):
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    keep = ((th.grad)**2).sum() > tol
            else:  # If an anomaly occurs, revert to the previous checkpoint.
                th.data = params2theta(*self.ev).to(self.device)

            with torch.no_grad():
                i += 1
                if observe and i % 100 == 5:
                    print(l.item(), self.loglike(th, components).sum().item(), self.lprior(*theta2params(th, components), components).item())
                if i % 100 == 42:
                    m, s, r = theta2params(th.clone().detach(), components)
                    ind = torch.argsort(-r)
                    m, s, r = m[ind], s[ind], r[ind]
                    self.ev = (m, s, r)
                    th.data = params2theta(*self.ev).to(self.device)
            optimizer.zero_grad()
            del l

        m, s, r = theta2params(th.clone().detach(), components)
        ind = r > r.max() * component_cut
        m, s, r = m[ind], s[ind], r[ind]
        ind = torch.argsort(-r)
        self.ev = (m[ind], s[ind], normalize(r[ind]))

        del self.lpkdil_n

        return self.ev
    
    def get_reconstruction(self, narray=None, cpu=True):
        if narray is None:
            x = self.n[1:]
        else:
            x = narray*1.
        m, s, r =  self.ev
        p = torch.exp(Igaussmix_loglike(x, m, s, r))
        if cpu:
            x,p = x.cpu(),p.cpu()

        return x,p
    
    def get_logreconstruction(self, narray=None, cpu=True, base = 10.):
        if narray is None:
            x = self.n[1:]
        lbase = log(base)
        x, p = self.get_reconstruction(x,cpu)

        log_narray = torch.log(x)/ lbase
        p_logspace = p * x * lbase

        return log_narray, p_logspace


    def dil_hist(self, ax):
        """
        Plot a histogram of the raw counts (ignoring dilution) on the provided axis.
        """
        h = ax.hist(self.counts.reshape(-1), alpha=0.25, bins=15, density=True)
        ax.set_xlabel('Counts', fontsize=15)
        ax.set_ylabel('Density', fontsize=15)
        return h

    def dil_imshow(self, ax, fig):
        """
        Plot an image (heatmap) of counts grouped by dilution.
        """
        dils = torch.unique(self.dils)
        g = []
        # For each unique dilution value (sorted in descending order), compute the histogram of counts.
        for dil in dils[argsort(-dils)]:
            # Try to use the threshold if available.
            g_dil = torch.zeros(self.counts.max() + 1, dtype=int)
            try:
                g_dil = torch.zeros(self.threshold + 1, dtype=int)
            except:
                pass
            k, fk = torch.unique(self.counts[self.dils == dil], return_counts=True)
            g_dil[k] += fk
            g.append(g_dil.numpy())

        # Bin the histogram if it is too finely grained.
        bin_size = (g[-1].size) // 50
        if bin_size > 1:
            bins = arange(0, g[-1].size - 1 + bin_size, bin_size)
            bins[-1] = g[-1].size - 1
            g = [[gi[low:high].sum() for (low, high) in zip(bins[:-1], bins[1:])] for gi in g]
        else:
            bins = arange(g[-1].size)

        aspect_ratio = 10
        height = len(g)
        width = len(g[0])
        extent = [0, width, 0, height * aspect_ratio]
        yticks = torch.arange(len(dils)) * aspect_ratio + aspect_ratio / 2

        ax.set_xticklabels((ax.get_xticks() * bins[-1]).astype(int))
        ax.set_yticks(yticks)
        ax.set_yticklabels([r'{}'.format(val) for val in dils.numpy().astype(int)])
        ax.tick_params(axis='y', labelrotation=90, labeltop=True)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')
        im = ax.imshow(g, extent=extent, aspect='auto')
        cbar = fig.colorbar(im, ax=ax)
        ax.set_xlabel('Counts', fontsize=15)
        ax.set_ylabel('Dilution', fontsize=15)

    def real_plots(self, ax, th_gt=None, bins=30):
        """
        Plot a histogram of the (dilution x counts) and overlay the reconstructed
        distribution (p(n)) from the Gaussian mixture model. Optionally, plot the ground truth.
        """
        x, p = self.get_reconstruction(cpu=True)
        ax.plot(x, p, label=r'REPOP')
        if th_gt is not None:
            p_gt = torch.exp(Igaussmix_loglike(x, *theta2params(th_gt, th_gt.size(0) // 3)))
            ax.plot(x, p_gt, label=r'Ground truth', color='k')
        h_high = ax.hist((self.counts * self.dils).reshape(-1), alpha=0.5,
                            bins=bins, density=True,
                            label=r'Dilution $\times$ Counts')
        
    def log_plots(self, ax, th_gt=None):
        """
        Plot a histogram of the log10(dilution x counts) and overlay the reconstructed
        distribution (p(n)) from the Gaussian mixture model. Optionally, plot the ground truth.
        """
        # x = self.n[1:].cpu()
        # m, s, r = [v.cpu() for v in self.ev]
        # p = torch.exp(Igaussmix_loglike(x, m.cpu(), s.cpu(), r.cpu()))
        # p_logspace = p * x * l10  # scale by x and a constant

        n_logspace,p_logspace = self.get_logreconstruction(cpu=True)
        ax.plot(n_logspace, p_logspace, label=r'REPOP')
        h = ax.hist(torch.log10((self.counts * self.dils)).clamp(0).reshape(-1),
                    alpha=0.5, bins=30, density=True, label=r'Dilution $\times$ Counts')
        
        # If any count is zero, color its bin red.
        if torch.any(self.counts == 0):

            bin_edges = h[1]
            bin_heights = h[0]
            for i in range(len(bin_edges) - 1):
                if bin_edges[i] <= 0 < bin_edges[i + 1]:
                    bin_zero_index = i
                    break

            ax.bar((bin_edges[bin_zero_index] + bin_edges[bin_zero_index + 1]) / 2, bin_heights[i],
                width=bin_edges[bin_zero_index + 1] - bin_edges[bin_zero_index], alpha=0.25, color='red')
        # Compute the reconstructed distribution using the Gaussian mixture likelihood.

        ax.set_ylim(0, 1.1 * (p_logspace.max()))
        if th_gt is not None:
            x = torch.pow(10,n_logspace)
            p_gt = torch.exp(Igaussmix_loglike(x, *theta2params(th_gt, th_gt.size(0) // 3)))
            y_gt = p_gt * x * l10
            ax.plot(n_logspace, y_gt, label=r'Ground truth', color='k')
            ax.set_ylim(0, 1.1 * max(p_logspace.max(), y_gt.max()))


        ax.set_xlim(h[1][0] * 0.9, h[1][-1] * 1.01)
        ax.set_xticklabels([rf"$10^{{{int(tick)}}}$" for tick in (ax.get_xticks())])
        ax.set_xlabel(r'Number of bacteria', fontsize=15)
        ax.set_ylabel('Density')


    def make_plot(self, filename=None, th_gt=None, xlabel='real', legend=True):
        """
        Create a combined plot for the dataset.
        If the dilution schedule is constant, plot a histogram of counts; otherwise, plot an imshow of dilutions
        alongside log plots. Optionally save the figure to a file.
        """
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        if torch.all(self.dils == self.dils[0]):
            h = self.dil_hist(ax[0])
            self.real_plots( ax[1], th_gt, bins = h[1] * (self.dils[0].item()))

            ax[1].set_xticks(ax[0].get_xticks() * self.dils[0].item())
            ax[1].set_xlim(ax[0].get_xlim()[0] * self.dils[0].item(), ax[0].get_xlim()[1] * self.dils[0].item())
            ax[1].set_xlabel('Number of bacteria', fontsize=15)
            ax[1].set_ylabel('Density', fontsize=15)

        else:
            self.dil_imshow(ax[0], fig)
            self.log_plots(ax[1], th_gt)
        for axi in ax:
            axi.tick_params(axis='both', which='major', labelsize=12)
            axi.yaxis.get_offset_text().set_fontsize(10)
            axi.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True)
            axi.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: '{:.0f}'.format(x)))
        if legend:
            ax[1].legend(fontsize=10)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=500)
        return fig
    
def plot_sci_not(ax):
    
    # Set scientific notation on both axes
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # Force scientific notation when necessary
    
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.xaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    

def set_gt(mus,sigs,rhos):
    return params2theta(torch.tensor(mus),torch.tensor(sigs),torch.tensor(rhos))