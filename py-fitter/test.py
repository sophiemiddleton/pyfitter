import numpy as np
import awkward as ak
from fit_module import Unbinned_fit_mom
from sensitivity_runners import fit_runner_1d_ul

# load nominal data (same file you used)
data = np.load('../test_fit_init.npz', allow_pickle=True)
mom = np.asarray(data['mom_mag']) if 'mom_mag' in data.files else np.asarray(data[data.files[0]])
mom_ak = ak.Array(mom)

# initial fit (no plotting)
fitresult, par, loss, nlls, combine_pdf, constraints = Unbinned_fit_mom(mom_ak, [], [], 95.0, 115.0, False, 2, minos=False, plot_NLL=False, plot_results=False)
print('Got model, par:', par)

# try to produce one toy robustly
toy = None
sampler = None
try:
    sampler = combine_pdf.create_sampler()
    print('sampler type:', type(sampler))
except Exception as e:
    print('create_sampler failed:', e)
if sampler is not None:
    for try_method in ('sample','draw'):
        if hasattr(sampler, try_method):
            try:
                toy = getattr(sampler, try_method)(100)
                print(f'got toy from sampler.{try_method} -> type {type(toy)}')
                break
            except Exception as e:
                print(f'sampler.{try_method} failed:', e)
    # try resample then sample
    try:
        sampler.resample({par: 0.0})
        if hasattr(sampler, 'sample'):
            toy = sampler.sample(100)
            print('got toy after resample; type', type(toy))
    except Exception as e:
        print('resample path failed:', e)

# fallback to combine_pdf.sample
if toy is None and hasattr(combine_pdf, 'sample'):
    try:
        toy = combine_pdf.sample(100)
        print('got toy from combine_pdf.sample; type', type(toy))
    except Exception as e:
        print('combine_pdf.sample failed:', e)

print('toy repr:', type(toy))
# call fit runner and print full returned dict
res = fit_runner_1d_ul(toy, fit_range=(95.0,115.0), constraints_dir=None, verbose=2)
print('fit_runner_1d_ul returned:')
for k,v in res.items():
    print(' ',k, type(v))
    if k == 'error':
        print('---- error traceback ----')
        print(v)
PY
