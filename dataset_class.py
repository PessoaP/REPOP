import torch
from numpy import sqrt,argsort,random,unique,log10
random.seed(42)
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt

log_comb = lambda n, k: torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
gaussian_loglike = lambda x, mu, sig: - torch.pow(((x - mu) / sig), 2) / 2  - torch.log(sig) - (1 / 2) * torch.log(torch.tensor(2 * torch.pi)) 
binomial_loglike = lambda k, n, p: log_comb(n, k) + k * torch.log(p) + (n - k) * torch.log(1 - p)
poisson_loglike = lambda k,rate: k*torch.log(rate) - rate - torch.lgamma(k+1)

weak_limit = 25
normalize = lambda x: x/x.sum()

def counts_loglike(k,n,phi):
    lp_bin = binomial_loglike(k,n,1./phi)
    lp_poi = poisson_loglike(k,n/phi)
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

def theta2params(theta,components=weak_limit):
    mus = torch.exp(theta[:components]).clone()
    #sigs = torch.exp(theta[components:2*components]+theta[:components])
    sigs = torch.exp(theta[components:2*components]).clone()
    rhos = torch.exp(theta[2*components:]).clone()
    return mus,sigs,normalize(rhos)

def params2theta(mus,sigs,rhos):
    return torch.hstack((torch.log(mus),
                         #torch.log(sigs)-torch.log(mus),
                         torch.log(sigs),
                         torch.log(rhos)))

def fixorder(th,components):
        m,s,r = theta2params(th.clone().detach(),components)
        ind = torch.argsort(-r)
        return params2theta(m[ind],s[ind],r[ind])

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



class dataset():
    def __init__(self,counts,dils,cutoff=-1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.counts = torch.tensor(counts.reshape(-1,1))
        self.dils = torch.tensor(dils.reshape(-1,1))
        self.ndatapoints=self.counts.size(0)

        self.ML = (counts*dils).clip(min=1).reshape(-1,1)
        self.Nmin = 1
        self.Nmax = 2*self.ML.max()+1
        self.width = torch.tensor(self.Nmax,device=self.device)
        self.n = torch.arange(self.Nmax)
        
        if cutoff == -1:
            self.lpkdil_n = counts_loglike(self.counts,self.n,self.dils)
        else:
            lpk_diln_unnorm = counts_loglike(self.counts,self.n,self.dils)
            logZ, lpdil_n = dils_switch(self.dils,self.Nmax,cutoff)
            lpk_diln = lpk_diln_unnorm - logZ + lpdil_n
            self.lpkdil_n = lpk_diln

        self.n = self.n.to(self.device)
        self.lpkdil_n = self.lpkdil_n.to(self.device)

   

    def lprior(self,mus,sigs,rhos,components=weak_limit):
        mus_shifted = (mus-self.Nmin)/self.width
        lp = .01*(torch.log(mus_shifted)+torch.log(1-mus_shifted)) -torch.log(self.width)
        lp += gaussian_loglike(torch.log(sigs),torch.log(mus),torch.ones_like(mus)) - torch.log(sigs)
        #lp += .2*( torch.log(sigs) - torch.log(mus) ) - sigs/mus*1.2 - torch.log(mus)
        if components!=1:
            return self.rhosprior.log_prob(rhos) + lp.sum()
        return lp.sum()
        
    
    def loglike(self,theta,components,total=False):
        mus,sigs,rhos = theta2params(theta,components)
        lpn_th = Igaussmix_loglike(self.n,mus,sigs,rhos)
        lpkn_th = (lpn_th+self.lpkdil_n)
        ans = torch.logsumexp(lpkn_th,axis=1)
        if total:
            return ans.sum()
        return ans
    
    def ML_estimate(self,components=weak_limit):
        
        gmm = GaussianMixture(n_components=components, covariance_type='full')
        gmm.fit(self.ML)

        prov_mus = gmm.means_.reshape(-1)
        prov_sigs = sqrt(gmm.covariances_).reshape(-1)
        prov_rhos = gmm.weights_

        indices = argsort(-prov_rhos)
        prov_mus,prov_sigs,prov_rhos = prov_mus[indices],prov_sigs[indices],prov_rhos[indices]
             
        self.ML_estimated = (torch.tensor(prov_mus),torch.tensor(prov_sigs),torch.tensor(prov_rhos))
        return self.ML_estimated

    

    
    def evaluate(self, components=weak_limit,tol=1e-5,lr=.01,observe=True,dir_factor=.9):
        #self.alpha = normalize((.9)**(torch.arange(components)))*120
        self.alpha = normalize((dir_factor)**(torch.arange(components)))
        self.alpha *=1.1/self.alpha[-1]
        #self.alpha = 150*((dir_factor)**(torch.arange(components)))

        self.rhosprior = torch.distributions.Dirichlet(self.alpha.to(self.device))

        self.ML_estimate(components)
        th = (1.0*params2theta(self.ML_estimated[0],
                               torch.sqrt(self.ML_estimated[0]),
                               #self.ML_estimated[1],
                               self.ML_estimated[2])).to(self.device).requires_grad_(True)

        loss = lambda x: -self.loglike(x,components).mean()  - self.lprior(*theta2params(x,components),components)/self.ndatapoints
        optimizer = torch.optim.Adam([th],lr=lr)
        torch.autograd.set_detect_anomaly(True)
        
        keep = True
        i=0
        while keep: 
            l = loss(th)
            l.backward()
            optimizer.step()
               
            with torch.no_grad():
                keep = ((th.grad)**2).sum()>tol
                i+=1
                if observe and i%10==5:
                    print(l.item(), self.loglike(th,components).sum().item(),self.lprior(*theta2params(th,components),components).item())

                if i%500 == 98:
                    m,s,r = theta2params(th.clone().detach(),components)
                    ind = torch.argsort(-r)
                    m,s,r = m[ind],s[ind],r[ind]
                    self.ev = m,s,r

                    ind = r>r.max()/500
                    self.ev = m[ind],s[ind],r[ind]

            optimizer.zero_grad()
            
            del l

        m,s,r = theta2params(th.clone().detach(),components)
        ind = r>r.max()/100
        m,s,r = m[ind],s[ind],r[ind]
        ind = torch.argsort(-r)
        self.ev = m[ind],s[ind],normalize(r[ind])

        return self.ev
    

    
    def dill_hist(self,ax):
        h = ax.hist(self.counts.reshape(-1),alpha=.25,bins=15,density=True)
        ax.set_xlabel('Counts',fontsize=12)
        ax.set_ylabel('Density',fontsize=12)
        return h
    
    def dill_imshow(self,ax,fig):
            dils = torch.unique(self.dils)
            g = []
            for dil in dils[argsort(-dils)]:
                g_dil = torch.zeros(self.counts.max()+1,dtype=int)
                k,fk = torch.unique(self.counts[self.dils==dil],return_counts=True)
                g_dil[k] += fk
                g.append(g_dil.numpy())

            aspect_ratio = 10  
            height = len(g)    
            width = len(g[0])

            extent = [0, width, 0, height * aspect_ratio]
            yticks = torch.arange(len(dils)) * aspect_ratio + aspect_ratio / 2
            ax.set_yticks(yticks)
            ax.set_yticklabels([r' ${:.1f} \times 10^{}$ '.format(val / (10**int(log10(val))),int(log10(val))) for val in dils.numpy().astype(int)])
            ax.tick_params(axis='y', labelrotation=90, labeltop=True)

            for label in ax.get_yticklabels():
                label.set_verticalalignment('center')
            im = ax.imshow(g, extent=extent, aspect='auto')
            cbar = fig.colorbar(im, ax=ax)

            ax.set_xlabel('Counts',fontsize=12)
            ax.set_ylabel('Dilution',fontsize=12)

    def log_plots(self,ax,th_gt=None):
            l10 = 2.30258509

            x = self.n[1:].cpu()
            m,s,r = [v.cpu() for v in self.ev]

            h = ax.hist(torch.log10(self.counts*self.dils).reshape(-1),alpha=.25,bins=30,density=True,label=r'Dilution $\times$ Counts')
            
            p = torch.exp(Igaussmix_loglike(x,m.cpu(),s.cpu(),r.cpu())) 
            y_ev = p*x*l10
            ax.plot(torch.log10(x),y_ev,label=r'Reconstructed $p(n)$')
            ax.set_ylim(0,1.1*(y_ev.max()))

            if not(th_gt is None):        
                p_gt = torch.exp(Igaussmix_loglike(x,*theta2params(th_gt,th_gt.size(0)//3)))
                y_gt = p_gt*x*l10

                ax.plot(torch.log10(x),y_gt,label=r'Ground truth',color='k')
                ax.set_ylim(0,1.1*max(y_ev.max(),y_gt.max()))
            
            ax.set_xlim(h[1][0]*.9,h[1][-1]*1.01)
            
            ax.set_xlabel(r'$\log_{10}$ (Number of bacteria)',fontsize=12)
            ax.set_ylabel('Density')
    
    def make_plot(self,filename=None,th_gt=None):
        
        fig,ax = plt.subplots(1,2,figsize=(10,4))    

        if torch.all(self.dils == self.dils[0]):
            h = self.dill_hist(ax[0])


            x = self.n[1:].cpu()
            m,s,r = [v.cpu() for v in self.ev]
            p = torch.exp(Igaussmix_loglike(x,m.cpu(),s.cpu(),r.cpu()))

            ax[1].plot(x,p,label=r'Reconstructed $p(n)$')
            h_high = ax[1].hist((self.counts*self.dils).reshape(-1),alpha=.25,bins=h[1]*(self.dils[0].item()),density=True,label=r'Dilution $\times$ Counts')

            if not(th_gt is None):        
                p_gt = torch.exp(Igaussmix_loglike(x,*theta2params(th_gt,th_gt.size(0)//3)))
                ax[1].plot(x,p_gt,label=r'Ground truth',color='k')
            ax[1].set_xlim(h_high[1][0]*.95,h_high[1][-1]*1.05)
            ax[1].set_xlabel('Number of bacteria',fontsize=12)
            ax[1].set_ylabel('Density',fontsize=12)

            for axi in ax:
                axi.ticklabel_format(style='sci', axis='both', scilimits=(0,0), useMathText=True)
                axi.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: '{:.0f}'.format(x)))
            ax[1].set_xticks(ax[0].get_xticks()*self.dils[0].item())

        else:
            self.dill_imshow(ax[0],fig)
            self.log_plots(ax[1],th_gt)

        

        [axi.tick_params(axis='both', which='major', labelsize=12) for axi in ax]
        [axi.yaxis.get_offset_text().set_fontsize(10) for axi in ax ]
        ax[1].legend(fontsize=10)
        plt.tight_layout()

        if not(filename is None):
            plt.savefig(filename,dpi=500)

        #plt.show()
        return fig