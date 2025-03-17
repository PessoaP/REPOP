# %%
import torch
from repop import dataset,params2theta,theta2params,plot_sci_not
import pandas as pd
from matplotlib import pyplot as plt
from synth_data import cases

# %%
def load_dataset(filename,Nmax=10**10,remove_zeros=True,cutoff=50):
    df=pd.read_csv(filename)[:Nmax]
    cts,dils = df['Counts'].to_numpy(),df['Dilution'].to_numpy().astype(float)
    if remove_zeros:
        cts,dils = cts[cts!=0],dils[cts!=0]
    return dataset(cts,dils,cutoff)

# %%
for case in (cases.casem1, cases.casem2):
  th_gt = params2theta(*torch.tensor((case.mus,case.sigs,case.rhos)))
  fig,axs = plt.subplots(4,2,figsize=(7,3*3),sharex='col',sharey='col') 
  first = True

  for Nmax, axi in zip(reversed((25, 100, 250, 1000)), reversed(axs)):

      data2 = load_dataset('synth_data/synth_{}.csv'.format(case.name),Nmax=Nmax,cutoff=-1)    
      data2.evaluate(observe=False)
      #print(data2.ev)
      if first:
        cmax = data2.counts.max().item()+3
        bins = torch.linspace(0,cmax,30).numpy()
        dil = data2.dils[0].item()
        first = False      
      
      data2.real_plots(axi[1],th_gt,bins=bins*dil)
      data2.dil_hist(axi[0])
      axi[0].set_ylabel('{} datapoints'.format(Nmax),fontsize=12)
      axi[0].set_xlabel('')
      
      del data2.lpkdil_n
      del data2

  ticks = torch.arange(0,cmax+49,50).numpy()

  axs[-1][0].set_xlabel('Counts',fontsize=18)
  axs[-1][1].set_xlabel('Number of bacteria',fontsize=18)
  axi[0].set_xlim((0,cmax)), axi[1].set_xlim((0,cmax*dil))
  axs[0][1].legend()


  ymax = axs[-1][0].get_ylim()[-1], 1/2.2*(case.rhos/case.sigs).max()
  [(axi[0].set_ylim((0,ymax[0])), axi[1].set_ylim((0,ymax[1])), plot_sci_not(axi[0]), plot_sci_not(axi[1])) for axi in axs]

  plt.tight_layout()
  fig.savefig('graphs/synth/{}_all.png'.format(case.name),dpi=900)


