# %%
import os
import torch
from repop import dataset,params2theta, theta2params, plot_sci_not, Igaussmix_loglike
import pandas as pd
from matplotlib import pyplot as plt
from synth_data import cases


#%%
def rel_error(data,case):
    mus,sigs,rhos = data.ev
    rel_error = torch.abs(mus.cpu().reshape(-1,1)/case.mus-1)
    return torch.sum(rel_error.min(axis=1)[0]*rhos.cpu())

#%%
def KL_GT(data, case):
    n = data.n

    m_found, s_found, r_found = data.ev
    log_p_found = Igaussmix_loglike(n, m_found, s_found, r_found).cpu()

    m, s, r = [torch.tensor(x) for x in (case.mus, case.sigs, case.rhos)]
    log_p_gt = Igaussmix_loglike(n.cpu(), m, s, r)

    # KL(p_gt || p_found) = sum p_gt * (log p_gt - log p_found)
    p_gt = torch.exp(log_p_gt)

    kl = torch.sum((p_gt * (log_p_gt - log_p_found))[p_gt > 0])  # Only consider terms where p_gt > 0 to avoid NaNs

    return kl
# %%
def load_dataset(filename,Nmax=10**10,remove_zeros=True,cutoff=50):
    df=pd.read_csv(filename)[:Nmax]
    cts,dils = df['Counts'].to_numpy(),df['Dilution'].to_numpy().astype(float)
    if remove_zeros:
        cts,dils = cts[cts!=0],dils[cts!=0]
    return dataset(cts,dils,cutoff)

# %%

if __name__ == "__main__":
    for case in (cases.casem1, cases.casem2):
        th_gt = params2theta(*torch.tensor((case.mus,case.sigs,case.rhos)))
        fig,axs = plt.subplots(4,2,figsize=(7,3*3),sharex='col',sharey='col') 
        first = True

        df = []
        Ns = range(25,1001,25)
        N_plot = (25, 100, 250, 1000)

        for Nmax in Ns:
            data2 = load_dataset('synth_data/synth_{}.csv'.format(case.name),Nmax=Nmax,cutoff=-1)    
            data2.evaluate(observe=False)     

            if first:
                cmax = data2.counts.max().item()+3
                bins = torch.linspace(0,cmax,30).numpy()
                dil = data2.dils[0].item()
                first = False    

            #calculate relative error and KL divergence
            df.append({'Nmax': Nmax, 
                        'Relative error': rel_error(data2,case).item(), 
                        'KL divergence': KL_GT(data2, case).item()})
            #print(df[-1])
            #del data2.lpkdil_n
            #del data2

            if Nmax in N_plot:
                axi = axs[N_plot.index(Nmax)]
                data2.real_plots(axi[1],th_gt,bins=bins*dil)
                data2.dil_hist(axi[0])
                axi[0].set_ylabel('{} datapoints'.format(Nmax),fontsize=12)
                axi[0].set_xlabel('')
                axi[0].set_xlim((0,cmax)), axi[1].set_xlim((0,cmax*dil))
            
            pd.DataFrame(df).to_csv('graphs/synth/{}_errors.csv'.format(case.name), index=False)

        ticks = torch.arange(0,cmax+49,50).numpy()

        axs[-1][0].set_xlabel('Counts',fontsize=18)
        axs[-1][1].set_xlabel('Number of bacteria',fontsize=18)
        axi[0].set_xlim((0,cmax)), axi[1].set_xlim((0,cmax*dil))
        axs[0][1].legend()


        ymax = axs[-1][0].get_ylim()[-1], 1/2.2*(case.rhos/case.sigs).max()
        [(axi[0].set_ylim((0,ymax[0])), axi[1].set_ylim((0,ymax[1])), plot_sci_not(axi[0]), plot_sci_not(axi[1])) for axi in axs]

        plt.tight_layout()

        os.makedirs("graphs/synth", exist_ok=True)
        fig.savefig('graphs/synth/{}_all.svg'.format(case.name),transparent=True)


