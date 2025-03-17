# %%
import torch
from repop import dataset,params2theta,theta2params
from matplotlib import pyplot as plt
import pandas as pd

import numpy as np
np.random.seed(42)

# %%
cutoff = 300
df = pd.read_csv('real_data/OD_exp.csv',header=None, names=['OD_dil','20','200','2000'])
df['OD_dil'] = df['OD_dil'].fillna(method='ffill')
OD_label = [    float(i.split('"')[1])  for i in df['OD_dil'] ]

df.iloc[:,1] = pd.to_numeric(df.iloc[:,1], errors='coerce').fillna(np.inf)
df.iloc[:,2] = pd.to_numeric(df.iloc[:,2], errors='coerce').fillna(np.inf)
df.iloc[:,3] = pd.to_numeric(df.iloc[:,3], errors='coerce').fillna(0)

ks = df.to_numpy()[:,1:4]
dil_schedule = np.array((20,200,2000))#/.9



# %%
cts,dils =[],[]
for k_line in ks:
    index = np.argmax(k_line <= cutoff) if np.any(ks <= cutoff) else len(ks) - 1
    cts.append( k_line[index] )
    dils.append( dil_schedule[index] )
    #print('Day ')

cts,dils = np.array(cts).astype(int),np.array(dils).astype(float)
od = np.array([float(d.split('"')[1]) for d in df['OD_dil']])
data =  dataset(cts[od!=1],dils[od!=1],cutoff)
data.evaluate(components=int(np.sqrt(data.counts.numel())))

# %%
fig,ax = plt.subplots(1,2,figsize=(10,3))
data.dil_imshow(ax[0],fig)
data.log_plots(ax[1])
plt.tight_layout()
ax[1].legend()
xlim = ax[1].get_xlim()
ax[1].set_ylim(0,2.4)

plt.savefig('graphs/3dils.png',dpi=500)






# %%
from synth_data import cases
#make a case with the 4 higher probability peaks
m,s,r = [d[:4].cpu() for d in data.ev]
r=r/r.sum()
cs = cases.case(m.numpy(),s.numpy(),r.numpy(),'Case redone')

cts,dils = cs.sample_data(size=500)

data_synth =  dataset(cts,dils,cutoff=300)
data_synth.evaluate()

fig,ax = plt.subplots(1,2,figsize=(10,3))
data_synth.dil_imshow(ax[0],fig)
data_synth.log_plots(ax[1],params2theta(m,s,r))

ax[1].set_xlim(xlim)
plt.tight_layout()
ax[1].legend()
ax[1].set_ylim(0,2.6)

plt.savefig('graphs/3dils_synth.png',dpi=500)



