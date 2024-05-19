import numpy as np
import pandas as pd
from numba import njit

seed=42
np.random.seed(seed)

@njit(cache=True)
def make_data(n_sam):
    cts = np.zeros_like(n_sam)
    dils = np.zeros_like(n_sam)
    for i in range(n_sam.size):
        done = False
        dil = 20
        while not(done):
            counts = np.random.binomial(n_sam[i],1/dil)
            if counts <= 50:
                cts[i] = counts
                dils[i] = dil
                done=True
            else:
                dil*=10

    return cts,dils





class case():
    def __init__(self,mus,sigs,rhos,name):
        self.mus,self.sigs,self.rhos = mus, sigs, rhos
        self.n_c = len(rhos)
        self.name=name

    def sample_n(self,size):
        ind = np.random.choice(len(self.rhos), size, p=self.rhos)
        return (np.random.normal(self.mus[ind], self.sigs[ind])).round().astype(int)
    
    def sample_data(self,size):
        n_sam = self.sample_n(size)
        return make_data(n_sam)
    
    def sample_save(self,size):
        cts,dils = self.sample_data(size)
        df = pd.DataFrame({'Counts':cts, 'Dilution':dils})
        df.to_csv('synth_{}.csv'.format(self.name),index=False)


case1 = case(np.array([1000,5000,10000]),
            np.array([100,300,800]),
            np.array([2/5,1/5,2/5]),'case1') #keep

case2 = case(np.array([1000,5000,15000]),
            np.array([100,300,1000]),
            np.array([1/6,3/6,2/6]),'case2') #new

case3 = case(np.array([1000,2500,7000]),
            np.array([100,300,500]),
            np.array([2/6,3/6,1/6]),'case3') #keep


case4 = case(np.array([15000,12000,7000]),
            np.array([3000,1000,500]),
            np.array([2/6,3/6,1/6]),'case4') #keep



@njit
def set_seed(value):
    np.random.seed(value)

if __name__ == "__main__":
    set_seed(seed)

    N = 1000000
    case1.sample_save(N)
    case2.sample_save(N)
    case3.sample_save(N)
    case4.sample_save(N)

