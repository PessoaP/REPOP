import numpy as np
import pandas as pd

seed=42
np.random.seed(seed)

def multinomial_with_completion(n, probs):
    """
    Perform a multinomial draw with probabilities that do not sum to 1.
    The missing probability is added as a new category, which is removed afterward.
    
    Parameters:
        n (int): Number of trials.
        probs (float array): Probabilities for the categories (may not sum to 1).
    
    Returns:
        samples (array): Result of the multinomial draw without the artificial category.
    """
    
    sum_probs = probs.sum()
   
    # Add missing probability as a new category
    if sum_probs < 1:
        probs = np.append(probs, 1 - sum_probs)
    
    # Perform the multinomial draw
    samples = np.random.multinomial(n, probs)
    
    # Remove the artificial category (last element)
    if sum_probs < 1:
        samples = samples[:-1]
    
    return samples


#@njit(cache=True)
def make_data(n_sam,cutoff=300,dils=20.0*np.power(10,np.arange(4))):
    cts = np.zeros_like(n_sam)
    dil = np.zeros_like(n_sam)
    probs = 1/dils

    for i in range(n_sam.size):
        #print(dils)
        ks = multinomial_with_completion(n_sam[i], probs)
        index = np.argmax(ks <= cutoff) if np.any(ks <= cutoff) else len(ks) - 1

        cts[i] = ks[index]
        dil[i] = dils[index]

    return cts,dil


class case():
    def __init__(self,mus,sigs,rhos,name,cutoff=300,dils=20.0*np.power(10,np.arange(4))):
        self.mus,self.sigs,self.rhos = mus, sigs, rhos
        self.n_c = len(rhos)
        self.name=name
        self.cutoff=cutoff
        self.dil_schedule=dils

    def sample_n(self,size):
        ind = np.random.choice(len(self.rhos), size, p=self.rhos)
        Ns =  (np.random.normal(self.mus[ind], self.sigs[ind])).round().astype(int)
        while np.any(Ns<=0):
            Ns[Ns<=0] = self.sample_n((Ns<=0).sum())
        return Ns
        
    
    def sample_data(self,size=1000000):
        n_sam = self.sample_n(size)
        if isinstance(self.dil_schedule,np.ndarray):
            return make_data(n_sam,self.cutoff,self.dil_schedule)
        return np.random.binomial(n_sam,np.ones_like(n_sam)/self.dil_schedule),np.ones_like(n_sam)*self.dil_schedule
    
    def sample_save(self,size=1000000,filename=''):
        if filename == '':
            filename = 'synth_{}.csv'.format(self.name)
        cts,dils = self.sample_data(size)
        df = pd.DataFrame({'Counts':cts, 'Dilution':dils})
        df.to_csv(filename,index=False)


case1 = case(np.array([1000,5000,10000]),
            np.array([100,300,800]),
            np.array([2/5,1/5,2/5]),'case1',cutoff=50) 

case2 = case(np.array([1000,5000,15000]),
            np.array([100,300,1000]),
            np.array([1/6,3/6,2/6]),'case2',cutoff=50) #keep

case3 = case(6*np.array([1000,5000,10000]),
            12*np.array([100,300,800]),
            np.array([2/5,1/5,2/5]),'case3',cutoff=300) #keep

case4 = case(6*np.array([1000,5000,15000]),
            9*np.array([100,300,1000]),
            np.array([1/6,3/6,2/6]),'case4',cutoff=300) 


case0 = case(np.array([8000]),
             np.array([500]),
             np.array([1]),'unimodal',cutoff=None,dils=200) #keep

casem1 = case(np.array([4000,8000,14000]),
              np.array([200,1500,1000]),
              np.array([.25,.4,.35]),'multimodal_harder',cutoff=None,dils=200) #keep

casem2 = case(np.array([8000,16000,24000]),
              np.array([1000,1000,1000]),
              np.array([.3,.2,.5]),'multimodal_easier',cutoff=None,dils=200) #keep


if __name__ == "__main__":
    #set_seed(seed)

    case1.sample_save()
    case2.sample_save()
    case3.sample_save()
    case4.sample_save()

    case0.sample_save()
    casem1.sample_save()
    casem2.sample_save()

