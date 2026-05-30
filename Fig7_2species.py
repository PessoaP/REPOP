# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from synth_data.cases import multinomial_with_completion
import torch
import repop


# %%
means = np.array([[4000, 24000],
                  [14000, 8000],
                  [8000, 16000]])

covs = np.array([[[200**2,  -0.7*200*1000],
                  [-0.7*200*1000, 1000**2]],
                 [[1500**2, 0.7*1500*1000],
                  [0.7*1500*1000, 1000**2]],
                 [[1000**2, 0],
                  [0,     1000**2]]])

weights = np.array([1/3, 1/3, 1/3])

np.random.seed(0)

N_sam = 1500
comps = np.random.choice(3, size=N_sam, p=weights)
samples = np.array([np.random.multivariate_normal(means[k], covs[k]).astype(int) for k in comps])


# %%
def plot_scatter_hist(samples, gt_params = None, dataset_x=None, dataset_y=None, more_samples = None):

    marginal_means, marginal_stds, marginal_weights = gt_params if gt_params is not None else (None, None, None)
    fig = plt.figure(figsize=(7, 7))

    gs = fig.add_gridspec(
        2, 2,
        width_ratios=(1, 3),
        height_ratios=(3, 1),
        hspace=0.05,
        wspace=0.05
    )

    
    
    ax_histx = fig.add_subplot(gs[1, 1],)
    ax_histy = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1], sharex=ax_histx, sharey=ax_histy)
    
    c = 'tab:blue'
    if dataset_x is not None and dataset_y is not None:
        ax_histx.plot(*dataset_x.get_reconstruction(), label="REPOP")
        ax_histy.plot(*(dataset_y.get_reconstruction())[::-1], label="REPOP")
        c = 'tab:orange'

    # marginal histograms
    ax_histx.hist(samples[:, 0], bins=40, density=True, alpha = .5, label=r'Dilution $\times$ Counts')
    ax_histy.hist(samples[:, 1], bins=40, orientation="horizontal", density=True, alpha = .5, label=r'Dilution $\times$ Counts')


    # scatter

    if more_samples is not None:
        samp,color,label = more_samples
        ax_scatter.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.25, color=c, label=r'Dilution $\times$ Counts')
        ax_scatter.scatter(samp[:, 0], samp[:, 1], s=8, alpha=0.25, color=color, label=label)
        ax_scatter.legend()
    else:
        ax_scatter.scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.25, color=c)
    ax_histy.set_ylabel("Number of bacteria ($n^B$)", fontsize=15)
    ax_histx.set_xlabel("Number of bacteria ($n^A$)", fontsize=15)


    # remove repeated tick labels
    ax_scatter.tick_params(labelbottom=False, labelleft=False)
    ax_histx.tick_params(labelbottom=True, labelleft=False)
    ax_histy.tick_params(labelbottom=False, labelleft=True)

    # important: do NOT force equal aspect
    ax_scatter.set_aspect("auto")

    # scientific notation, 10^4
    for axi in [ax_histx, ax_scatter, ax_histy]:
        #axi.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        repop.plot_sci_not(axi)

    if marginal_means is not None:
        print(marginal_means[0, :],marginal_stds[0,:], marginal_weights)
        n_x = np.arange(samples[:,0].max()+1)
        px  = repop.Igaussmix_loglike(torch.tensor(n_x), 
                                      torch.tensor(marginal_means.T[0, :]), 
                                      torch.tensor(marginal_stds.T[0,:]), 
                                      torch.tensor(marginal_weights)).numpy()
        ax_histx.plot(n_x, np.exp(px), color='k', label ='Ground truth')
        ax_histx.set_xlim(0, samples[:,0].max()+1000)

        n_y = np.arange(samples[:,1].max()+1)
        py  = repop.Igaussmix_loglike(torch.tensor(n_y),
                                      torch.tensor(marginal_means.T[1, :]), 
                                      torch.tensor(marginal_stds.T[1,:]), 
                                      torch.tensor(marginal_weights)).numpy()
        ax_histy.plot(np.exp(py), n_y, color='k', label ='Ground truth')
        ax_histy.set_ylim(0, samples[:,1].max()+1000)

    ax_histx.legend()
    #ax_histy.legend()
    plt.tight_layout()
    #plt.show()


# %%
dil = 200
k_samples = np.random.binomial(samples.T, 1/dil)
k = torch.tensor(k_samples, dtype=torch.float32)


# %%
data1 = repop.dataset(k[0], torch.ones_like(k[0], dtype=torch.float32)*dil)
data2 = repop.dataset(k[1], torch.ones_like(k[1], dtype=torch.float32)*dil)


data1.evaluate()
data2.evaluate()


# %%

plot_scatter_hist(k_samples.T*dil,(means, torch.stack([torch.sqrt(torch.tensor(cv.diagonal())) for cv in covs]),weights),data1,data2,(samples,'k','Original samples'))
plt.savefig("graphs/Fig7_2species.png", dpi=900)
plt.savefig("graphs/Fig7_2species.svg", transparent=True)

# %%



