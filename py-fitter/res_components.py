import zfit
import pickle as pkl
from landau_pdf import trunc_landau

class res_components:
    """
    Class contains parameterization for the resolutions function, specifically for the CELL distribution
    
    """
    def __init__(self, p_bins = [95., 97., 99., 101., 103., 105.], params = None, simul_source = None, res_type = 'res', pdf = 'gcb'):
        """Initialise the resolution function handler
        Args:
            p_bins       : momentum bins. Overridden by keys in params dict, if provided
            params       : params dict (for fix / constrain / float fits)
            simul_source : input data for simultaneous fit  
        """
        self.res_type = res_type
        self.pdf = pdf
        self.simul_source = simul_source
        self.fitpars = {'info' : {'pdf' : pdf}}
        if simul_source is not None:
            self.fitpars['info']['p_bins'] = []
            for ip in range(len(p_bins)-1):
                self.fitpars['info']['p_bins'].append((p_bins[ip],p_bins[ip+1]))
                if pdf == 'gcb' : 
                    self.fitpars[f"mu{ip}_{res_type}"]     = zfit.Parameter(f"mu{ip}_{res_type}",     0.0, -3.0, 0.5)
                    self.fitpars[f"sigmaL{ip}_{res_type}"] = zfit.Parameter(f"sigmaL{ip}_{res_type}", 0.5,  0.0, 2.0)
                    self.fitpars[f"sigmaR{ip}_{res_type}"] = zfit.Parameter(f"sigmaR{ip}_{res_type}", 0.5,  0.0, 2.0)
                    self.fitpars[f"alphaL{ip}_{res_type}"] = zfit.Parameter(f"alphaL{ip}_{res_type}", 0.5,  0.0, 3.0)
                    self.fitpars[f"alphaR{ip}_{res_type}"] = zfit.Parameter(f"alphaR{ip}_{res_type}", 0.5,  0.0, 3.0)
                    self.fitpars[f"nL{ip}_{res_type}"]     = zfit.Parameter(f"nL{ip}_{res_type}",     2.0,  0.0, 12.0)
                    self.fitpars[f"nR{ip}_{res_type}"]     = zfit.Parameter(f"nR{ip}_{res_type}",     2.0,  0.0, 12.0)
                elif pdf == 'landau' :
                    self.fitpars[f"loc{ip}_{res_type}"]   = zfit.Parameter(f"loc{ip}_{res_type}",   0.0, -5.0, 5.0)
                    self.fitpars[f"scale{ip}_{res_type}"] = zfit.Parameter(f"scale{ip}_{res_type}", 1.0,  0.0, 5.0)
                else:
                    print("ERROR: res_components only supports gcb or landau pdfs. Exiting...")
                    exit()
        else:
            if isinstance(params,str):
                pardict = dict(pkl.load(open(params,'rb')))
            else:
                pardict = params

            self.fitpars['info']['p_bins'] = list(pardict.keys())
            for p_bin in pardict.keys():
                for p in pardict[p_bin].keys():
                    self.fitpars[p] = zfit.Parameter(p, pardict[p_bin][p][0], pardict[p_bin][p][0]+5*pardict[p_bin][p][1], pardict[p_bin][p][0]+5*pardict[p_bin][p][2],step_size=0.0001)

    def get_params(self):
        return self.fitpars
            
    def get_nll(self,params_tot):
        """
        Get the nll associated with resolution
        """
        nlls = []
        plane = 'entrance' # TODO this should be dynamic, match sid in cut_module.py. Flag when calling main.py?
        true_mom, reco_mom = self.simul_source
        obs_res  = zfit.Space('x',-1,1) if self.res_type == 'res' else zfit.Space('x',-10,10)
        
        # Get slice of flat p data
        for ip,p_bin in enumerate(self.fitpars['info']['p_bins']):
            true_slice = true_mom[(true_mom>=p_bin[0]) & (true_mom<p_bin[1])]
            reco_slice = reco_mom[(true_mom>=p_bin[0]) & (true_mom<p_bin[1])]
            res_slice = reco_slice - true_slice
            data_res = zfit.Data(data=res_slice, obs=obs_res)
            
            N_slice = zfit.Parameter(f'N{ip}_{self.res_type}', len(res_slice), 0.8*len(res_slice), 1.2*len(res_slice), step_size=0.1)
            params_tot.append(N_slice)
            if self.pdf == 'gcb':
                res = zfit.pdf.GeneralizedCB(obs=obs_res, mu=self.fitpars[f'mu{ip}_{self.res_type}'], sigmal=self.fitpars[f'sigmaL{ip}_{self.res_type}'], sigmar=self.fitpars[f'sigmaR{ip}_{self.res_type}'], alphal=self.fitpars[f'alphaL{ip}_{self.res_type}'], alphar=self.fitpars[f'alphaR{ip}_{self.res_type}'], nl=self.fitpars[f'nL{ip}_{self.res_type}'], nr=self.fitpars[f'nR{ip}_{self.res_type}'], extended=N_slice)
            elif self.pdf == 'landau':
                res = trunc_landau(obs=obs_res, loc=self.fitpars[f'loc{ip}_{self.res_type}'], scale=self.fitpars[f'scale{ip}_{self.res_type}'], extended=N_slice)
            params_tot.extend([v for k,v in self.fitpars.items() if k != 'info'])

            nlls.append(zfit.loss.ExtendedUnbinnedNLL(model=res, data=data_res))

        return nlls
