import torch
from numpy import sqrt,argsort,random
random.seed(42)
from sklearn.mixture import GaussianMixture

log_comb = lambda n, k: torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
gaussian_loglike = lambda x, mu, sig: - torch.pow(((x - mu) / sig), 2) / 2  - torch.log(sig) - (1 / 2) * torch.log(torch.tensor(2 * torch.pi)) 
binomial_loglike = lambda k, n, phi: log_comb(n, k) + k * torch.log(phi) + (n - k) * torch.log(1 - phi)
poisson_loglike = lambda k,rate: k*torch.log(rate) - rate - torch.lgamma(k+1)
geometric_loglike = lambda k,p: torch.log(p) + (k-1)*torch.log(1-p)

normalize = lambda x: x/x.sum()

def counts_loglike(k,n,phi):
    lp_poi = poisson_loglike(k,n/phi)
    lp_bin = binomial_loglike(k,n,1./phi)
    return torch.where ((n>phi)*(phi>100),lp_poi,lp_bin)

def Igaussmix_loglike(n,mus,sigs,rhos,unit=False):
    terms_unorm = gaussian_loglike(n,mus.reshape(-1,1),sigs.reshape(-1,1))
    terms = terms_unorm- torch.logsumexp(terms_unorm,axis=1).reshape(-1,1)
    lpn_unorm = torch.logsumexp(terms+torch.log(rhos.reshape(-1,1)),axis=0)
    lpn = lpn_unorm - torch.logsumexp(lpn_unorm,axis=0)

    if unit:
        print('should be all ones', torch.exp(terms).sum(axis=1) )
        print('intermediate',torch.logsumexp(lpn_unorm,axis=0))
        print('should be one     ', torch.exp(lpn).sum() )
    return lpn

def theta2params(theta,components=2):
    mus = torch.exp(theta[:components]).clone()
    sigs = torch.exp(theta[components:2*components]+theta[:components])
    rhos = torch.exp(theta[2*components:]).clone()
    rhos /= rhos.sum()
    return mus,sigs,rhos

def params2theta(mus,sigs,rhos):
    return torch.hstack((torch.log(mus),
                         torch.log(sigs)-torch.log(mus),
                         torch.log(rhos)))

def dils_switch(dils,N,cutoff):
    n = torch.arange(N)
    k = torch.arange(cutoff+1).reshape(-1,1)

    dils_unique, inverse = torch.unique(dils,return_inverse=True,sorted=True)
    #maybe needs change above
    dils_num = dils_unique.size(0)
    logZdils,pdils = [], []
    lp_antes = torch.zeros_like(n)

    for i in range(dils_num):
        d = dils_unique[i]
        binomial_logZ = torch.logsumexp(counts_loglike(k,n,d),axis=0)
        binomial_logZ = (1000*binomial_logZ).round()/1000
        logZdils.append(binomial_logZ)
        pdils.append(binomial_logZ+lp_antes)
        lp_antes += torch.log(1-torch.exp(binomial_logZ))

    return torch.vstack(logZdils)[inverse.reshape(-1)],torch.vstack(pdils)[inverse.reshape(-1)]

def lprior(mus,sigs,rhos,p=.5):
    if isinstance(rhos, (torch.Tensor)):
        return geometric_loglike(torch.tensor(rhos.size(0)),torch.tensor(p)) 
    return geometric_loglike(torch.tensor(rhos.size),torch.tensor(p))  #+rho_prior


class dataset():
    def __init__(self,counts,dils,cutoff=-1,max_comp=10):
        self.counts = torch.tensor(counts.reshape(-1,1))
        self.dils = torch.tensor(dils.reshape(-1,1))

        self.max_comp = max_comp

        self.ML = (counts*dils).clip(min=1).reshape(-1,1)
        self.N = 2*(self.ML).max()+1

        self.n = torch.arange(self.N)
        
        if cutoff == -1:
            print(cutoff)
            self.lpkdil_n = counts_loglike(self.counts,self.n,self.dils)
        else:
            lpk_diln_unnorm = counts_loglike(self.counts,self.n,self.dils)
            logZ, lpdil_n = dils_switch(self.dils,self.N,cutoff)
            lpk_diln = lpk_diln_unnorm - logZ + lpdil_n
            self.lpkdil_n = lpk_diln

        #below implements the gpu acceleration if a GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n = self.n.to(self.device)
        self.lpkdil_n = self.lpkdil_n.to(self.device)

    
    def loglike(self,theta,components=2,total=False):
        mus,sigs,rhos = theta2params(theta,components)
        lpn_th = Igaussmix_loglike(self.n,mus,sigs,rhos)
        lpkn_th = (lpn_th+self.lpkdil_n)
        ans = torch.logsumexp(lpkn_th,axis=1)
        if total:
            return ans.sum()
        return ans
    
    def ML_estimate(self,components=-1):
        ml_comp = []
        best_logpost = -torch.inf
        components_iterable = range(1,1+self.max_comp) if (components == -1) else [components]
        for c in components_iterable:
            gmm = GaussianMixture(n_components=c, covariance_type='full')
            gmm.fit(self.ML)

            prov_mus = gmm.means_.reshape(-1)
            prov_sigs = sqrt(gmm.covariances_).reshape(-1)
            prov_rhos = gmm.weights_

            map_est = gmm.score_samples(self.ML).sum() + lprior(prov_mus,prov_sigs,prov_rhos)

            indices = argsort(prov_mus)
            prov_mus,prov_sigs,prov_rhos = prov_mus[indices],prov_sigs[indices],prov_rhos[indices]
             
            ml_comp.append((torch.tensor(prov_mus),torch.tensor(prov_sigs),torch.tensor(prov_rhos)))
            if map_est>best_logpost:
                best_logpost = map_est
                self.ML_estimated = (torch.tensor(prov_mus),torch.tensor(prov_sigs),torch.tensor(prov_rhos))
        self.ML_all = ml_comp

    

    
    def evaluate(self, components=-1,tol=1e-3,lr=.01,observe=True):
        components_iterable = range(1,1+self.max_comp) if (components == -1) else [components]
        self.ML_estimate(components)
        best_logpost = -torch.inf
        ev_comp=[]
        for (c,ml_ext_c) in zip(components_iterable,self.ML_all):
            loss = lambda x: -self.loglike(x,components=c).mean()

            
            th = (1.0*params2theta(ml_ext_c[0],ml_ext_c[1]*.7,ml_ext_c[2])).to(self.device).requires_grad_(True)

            optimizer = torch.optim.Adam([th],lr=lr)

            torch.autograd.set_detect_anomaly(True)
            keep = True
            i=0
            while keep: # Number of iterations
                # Compute the function value at the current point
                l = loss(th)
                l.backward()
                optimizer.step()
                
                with torch.no_grad():
                    keep = ((th.grad)**2).sum()>tol
                    i+=1
                    if observe:
                        print(f"Iteration {i}, x = {th.detach().cpu().numpy()}, f(x) = {l.item()}")

                optimizer.zero_grad()
                del l

            with torch.no_grad():
                map_est = self.loglike(th,components=c).sum() +lprior(*theta2params(th))
                if map_est > best_logpost:
                    best_logpost = map_est
                    self.ev = theta2params(th.clone().detach(),components=c)
                ev_comp.append(theta2params(th.clone().detach(),components=c))

        self.ev_comp = ev_comp
        return self.ev