# %%
import os
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
print(len(df))

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

plt.savefig('graphs/3dils.svg',transparent=True)





# %%
from synth_data import cases
#make a case with the 4 higher probability peaks
m,s,r = [d[:4].cpu() for d in data.ev]

r=r/r.sum()
cs = cases.case(m.numpy(),s.numpy(),r.numpy(),'Case redone')

cts,dils = cs.sample_data(size=750)

data_synth =  dataset(cts,dils,cutoff=300)
data_synth.evaluate()

fig,ax = plt.subplots(1,2,figsize=(10,3))
data_synth.dil_imshow(ax[0],fig)
data_synth.log_plots(ax[1],params2theta(m,s,r))
#[ax[1].axvline(np.log10(m[i]), linestyle='--') for i in range(4)]

ax[1].set_xlim(xlim)
plt.tight_layout()
ax[1].legend()
ax[1].set_ylim(0,2.6)

plt.savefig('graphs/3dils_synth.svg',transparent=True)






import repop

fig, axn = plt.subplots(2, 2, figsize=(10, 6), sharex='col')


n = data_synth.n.cpu()

for i in range(4):
    p_i = torch.exp(repop.Igaussmix_loglike(n, m[i], s[i], torch.ones_like(s[i])))
    axn[0, 0].plot(n, p_i)
    axn[0, 1].plot(torch.log(n[1:])/2.3, p_i[1:]*n[1:]*2.3)
    
    #axn[0, 0].set_xlabel('Number of bacteria', fontsize=15)
    axn[0, 0].set_ylabel('Density per component', fontsize=15)
    #axn[0, 1].set_xlabel(r'$\log_{10}$ (Number of bacteria)', fontsize=15)
    #axn[0, 1].set_ylabel('Density per component', fontsize=15)

p_mix = torch.exp(repop.Igaussmix_loglike(n, m, s, r))

axn[1, 0].plot(n, p_mix)
axn[1, 0].set_xlabel('Number of bacteria', fontsize=15)
axn[1, 0].set_ylabel('Density of mixture', fontsize=15)
axn[1, 1].plot(torch.log(n[1:])/2.3, p_mix[1:]*n[1:]*2.3)
axn[1, 1].set_xlabel(r'$\log_{10}$ (Number of bacteria)', fontsize=15)
#axn[1, 1].set_ylabel('Density of mixture', fontsize=15)


for axi in axn[:, 0].flatten():
    repop.plot_sci_not(axi)
    axi.tick_params(axis='both', which='major', labelsize=12)
    axi.yaxis.get_offset_text().set_fontsize(10)
    axi.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True)
    axi.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: '{:.0f}'.format(x)))
    axi.set_xlim(0, 5*1e4)

for axi in axn[:, 1].flatten():
    axi.set_xlim(2,5)

plt.tight_layout()
plt.savefig("graphs/3dils_sanity.png", dpi=300)




# %%
