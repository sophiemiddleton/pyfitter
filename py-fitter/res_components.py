import zfit
import pickle as pkl

class res_components:
    """
    Class contains parameterization for the resolutions function, specifically for the CELL distribution
    
    """
    def __init__(self, p_bins = [95., 97., 99., 101., 103., 105.], res_sample = '/exp/mu2e/data/users/sdittmer/SignalShape/skimmed_flat_mom.pkl', res_type = 'gen'):
        """Initialise the resolution function handler
        Args:
            p_bins = momentum bins where valid
            res_sample = path to location of the .pkl fits #FIXME we should put these within the repo
        """
        self.p_bins = p_bins
        self.res_sample = res_sample
        self.res_type = res_type
        self.fitpars_res = {}
        for ip in range(len(p_bins)-1):
            self.fitpars_res[(p_bins[ip],p_bins[ip+1])] = {
                f"mu{ip}"     : zfit.Parameter(f"mu{ip}",     0.0, -3.0, 0.5),
                f"sigmaL{ip}" : zfit.Parameter(f"sigmaL{ip}", 0.5,  0.0, 2.0),
                f"sigmaR{ip}" : zfit.Parameter(f"sigmaR{ip}", 0.5,  0.0, 2.0),
                f"alphaL{ip}" : zfit.Parameter(f"alphaL{ip}", 0.5,  0.0, 3.0),
                f"alphaR{ip}" : zfit.Parameter(f"alphaR{ip}", 0.5,  0.0, 3.0),
                f"nL{ip}"     : zfit.Parameter(f"nL{ip}",     2.0,  0.0, 12.0),
                f"nR{ip}"     : zfit.Parameter(f"nR{ip}",     2.0,  0.0, 12.0)
            }

    def params(self):
        return self.fitpars_res
            
    def get_nll(self,params_tot):
        """
        Get the nll associated with resolution
        """
        nlls = []
        plane = 'entrance' # TODO this should be dynamic, match sid in cut_module.py. Flag when calling main.py?
        dict_flat = pkl.load(open(self.res_sample,'rb'))
        reco_mom_flat = dict_flat[self.res_type][plane]['reco'] 
        true_mom_flat = dict_flat[self.res_type][plane]['true']

        obs_res  = zfit.Space('x',-10,10) if self.res_type == "gen" else zfit.Space('x',-1,1)
        
        # Get slice of flat p data
        for ip in range(len(self.p_bins)-1):
            flat_true_slice = true_mom_flat[(true_mom_flat>=self.p_bins[ip]) & (true_mom_flat<self.p_bins[ip+1])]
            flat_reco_slice = reco_mom_flat[(true_mom_flat>=self.p_bins[ip]) & (true_mom_flat<self.p_bins[ip+1])]
            res_slice = flat_reco_slice - flat_true_slice
            data_res = zfit.Data(data=res_slice, obs=obs_res)

            N_slice = zfit.Parameter(f'N{ip}', len(res_slice), 0.8*len(res_slice), 1.2*len(res_slice), step_size=0.1)
            params_tot.append(N_slice)
            zpars = self.fitpars_res[(self.p_bins[ip],self.p_bins[ip+1])]
            res = zfit.pdf.GeneralizedCB(obs=obs_res, mu=zpars[f'mu{ip}'], sigmal=zpars[f'sigmaL{ip}'], sigmar=zpars[f'sigmaR{ip}'], alphal=zpars[f'alphaL{ip}'], alphar=zpars[f'alphaR{ip}'], nl=zpars[f'nL{ip}'], nr=zpars[f'nR{ip}'], extended=N_slice)
            params_tot.extend(list(zpars.values()))

            nlls.append(zfit.loss.ExtendedUnbinnedNLL(model=res, data=data_res))

        return nlls
