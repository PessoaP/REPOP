# %%
import torch
from repop import *
import pandas as pd
from matplotlib import pyplot as plt
from synth_data import cases

import numpy as np

# %%
def load_dataset(filename,Nmax=10**1000,remove_zeros=True,cutoff=-1):
    df=pd.read_csv(filename)[:Nmax]
    cts,dils = df['Counts'].to_numpy(),df['Dilution'].to_numpy().astype(float)
    if remove_zeros:
        cts,dils = cts[cts!=0],dils[cts!=0]
    return dataset(cts,dils,cutoff)

def rel_error(data,case):
    mus,sigs,rhos = data.ev
    rel_error = torch.abs(mus.cpu().reshape(-1,1)/case.mus-1)
    return torch.sum(rel_error.min(axis=1)[0]*rhos.cpu())

# %%
def load_data_gt(case,datapoints,cutoff=True):
    filename = 'synth_data/synth_'+ case.name+ '.csv'
    th_gt = params2theta(torch.tensor(case.mus),
                        torch.tensor(case.sigs),
                        torch.tensor(case.rhos))
    if cutoff:
        return load_dataset(filename,Nmax=datapoints,cutoff=case.cutoff),th_gt
    return load_dataset(filename,Nmax=datapoints),th_gt

# %%
for (case,datapoints) in zip((cases.case2,cases.case3),(1000,500)):
    print(case.name)
    
    data,th_gt =  load_data_gt(case,datapoints,True)
    data.evaluate()
    print('Relative error -- Threshold', rel_error(data,case))

    data_noTH,th_gt =  load_data_gt(case,datapoints,cutoff=False)
    data_noTH.evaluate()
    print('Relative error - No cutoff', rel_error(data_noTH,case))

    fig,ax = plt.subplots(1,3,figsize=(12,3))
    data.dil_imshow(ax[0],fig)
    data.log_plots(ax[1],th_gt)
    data_noTH.log_plots(ax[2],th_gt)


    ax[1].set_title('Accounting for cutoff (model 3)',fontsize=12)
    ax[1].legend()
    ax[2].set_title('Not accounting for cutoff (model 2)',fontsize=12)
    [axi.set_ylabel('Density', fontsize= ax[0].yaxis.label.get_size()) for axi in ax[1:]]

    #legend=False
    for ct in np.log10(data.cutoff*torch.unique(data.dils).numpy()):
        ax[2].axvline(ct,color='red',linestyle='dashed')

    plt.tight_layout()
    fig.savefig('graphs/synth/'+case.name,dpi=900)


