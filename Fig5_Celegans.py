# %%
import torch
from repop import dataset,params2theta,theta2params
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, FuncFormatter
import pandas as pd

import numpy as np
np.random.seed(42)

# %%
cutoff = 300
df = pd.read_csv('real_data/Exp_1_Ecoli_population.csv')
df.iloc[:,1] = pd.to_numeric(df.iloc[:,1], errors='coerce').fillna(np.inf)
df.iloc[:,2] = pd.to_numeric(df.iloc[:,2], errors='coerce').fillna(0)
dil_schedule = np.array((20,200,2000))/.9

# %%
def get_counts(df,day):
    ks = df.to_numpy()[df['Day']==day,1:3]

    cts,dils =[],[]
    for k_line in ks:
        index = np.argmax(k_line <= cutoff) if np.any(ks <= cutoff) else len(ks) - 1
        cts.append( k_line[index] )
        dils.append( dil_schedule[index] )

    cts,dils = np.array(cts).astype(int),np.array(dils).astype(float)
    keep = np.logical_and(cts<=300,  cts>=0)

    return cts[keep],dils[keep]

# %%
days = (3,5,7,9)

data_day =[]
for day in days:
    cts,dils = get_counts(df,day)
    data_day.append(dataset(cts,dils))
    (data_day[-1]).evaluate(observe=False)

# %%
layout = [
    ['lin0', 'log0', 'log0'],
    ['lin1', 'log1', 'log1'],
    ['lin2', 'log2', 'log2'],
    ['lin3', 'log3', 'log3']
]
fig, axes = plt.subplot_mosaic(layout, figsize=(8, 6))

for i, dt in enumerate(data_day[:4]):
    #dt.real_plots(axes[f'lin{i}'])
    x, p = dt.get_reconstruction(cpu=True)
    axes[f'lin{i}'].plot(x[x<120], p[x<120])
    axes[f'lin{i}'].set_xlim(0,100)

    cts = (dt.counts * dt.dils).reshape(-1)
    cts = cts[cts<100]

    axes[f'lin{i}'].hist(cts, alpha=0.5,
                            bins=10, density=True)


    axes[f'lin{i}'].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axes[f'lin{i}'].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    dt.log_plots(axes[f'log{i}'],bins=np.linspace(2,5.25,31))
    axes[f'log{i}'].set_xlim(2,5.25)


ym_lin = max(axes[f'lin{i}'].get_ylim()[1] for i in range(4))
ym_log = max(axes[f'log{i}'].get_ylim()[1] for i in range(4))
for i in range(4):
    axes[f'lin{i}'].set_ylim(0, ym_lin * 1.1)
    axes[f'log{i}'].set_ylim(0, ym_log * 1.1)

# %%
for i in range(4):
    axes[f'lin{i}'].set_ylabel(f'Day {days[i]}', fontsize=12)
    axes[f'log{i}'].set_ylabel('')  # avoid redundancy
    axes[f'lin{i}'].set_xlabel('')  # clear x-labels except bottom
    axes[f'log{i}'].set_xlabel('')


# Add legend to top-left
axes['log0'].legend()
axes['lin3'].set_xlabel('Number of bacteria', fontsize=15)
axes['log3'].set_xlabel(r'$\log_{10}$ (Number of bacteria)', fontsize=15)

plt.tight_layout()
plt.savefig('graphs/worms_mosaic.png', dpi=500)



