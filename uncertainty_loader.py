import os
import json
import numpy as np
import zfit

def load_constraints_json(dirpath):
    """Load constraints.json from dirpath and return a list of constraint specs.

    Each spec is a dict with keys:
      - pname: target zfit.Parameter.name (preferred)
      - prior: {dist: 'gauss', mean:..., sigma:...}
      - default: optional central value
      - notes: optional
    """
    fp = os.path.join(dirpath, 'constraints.json')
    if not os.path.exists(fp):
        return []
    try:
        obj = json.load(open(fp, 'r'))
    except Exception:
        return []
    specs = []
    for ent in obj:
        spec = {}
        # preferred: explicit pname
        if 'pname' in ent:
            spec['pname'] = ent['pname']
        elif 'name' in ent:
            # convert dotted name to parameter-name convention used in MomModel (dots -> underscore)
            spec['pname'] = ent['name'].replace('.', '_')
        else:
            # try process/param
            proc = ent.get('process')
            param = ent.get('param')
            if proc and param:
                spec['pname'] = f"{param}_{proc}"
            else:
                # skip ambiguous entries
                continue
        spec['prior'] = ent.get('prior', {})
        spec['default'] = ent.get('default', None)
        spec['notes'] = ent.get('notes', '')
        specs.append(spec)
    return specs


def build_zfit_constraints_from_specs(pars, specs, logger=None):
    """Match specs to parameters in `pars` (iterable of zfit.Parameters) and build zfit constraints.

    Supports Gaussian priors (`prior['dist']=='gauss'`) with keys `mean` and `sigma`.
    Returns list of zfit.constraint.* objects.
    """
    name_map = {p.name: p for p in pars}
    if logger:
        logger.log(f"Available parameters for constraints: {list(name_map.keys())}", 'info')
    constraints = []
    for s in specs:
        pname = s.get('pname')
        prior = s.get('prior', {})
        p = None
        if pname in name_map:
            p = name_map[pname]
        else:
            # tolerant matching: allow parameter names that contain or end with the requested pname
            candidates = [n for n in name_map.keys() if n == pname or n.endswith(pname) or pname in n or n.endswith('_' + pname)]
            if len(candidates) == 1:
                mapped = candidates[0]
                p = name_map[mapped]
                if logger:
                    logger.log(f"Constraint target {pname} matched to existing parameter {mapped}", 'info')
            elif len(candidates) > 1:
                # pick the longest match (most specific) and log ambiguity
                candidates_sorted = sorted(candidates, key=lambda x: len(x), reverse=True)
                mapped = candidates_sorted[0]
                p = name_map[mapped]
                if logger:
                    logger.log(f"Constraint target {pname} ambiguous, picked parameter {mapped}", 'info')
            else:
                if logger:
                    logger.log(f"Constraint target {pname} not found among parameters; skipping", 'info')
                continue
        dist = prior.get('dist', 'gauss')
        if dist == 'gauss':
            mean = prior.get('mean', None)
            sigma = prior.get('sigma', None)
            if mean is None or sigma is None:
                if logger:
                    logger.log(f"Gaussian prior for {pname} missing mean/sigma; skipping", 'info')
                continue
            try:
                constraints.append(zfit.constraint.GaussianConstraint(p, observation=mean, uncertainty=sigma))
            except Exception:
                if logger:
                    logger.log(f"Failed to create GaussianConstraint for {pname}", 'error')
        else:
            if logger:
                logger.log(f"Unsupported prior dist '{dist}' for {pname}; skipping", 'info')
    return constraints


def load_templates_npz(dirpath):
    """Load templates.npz if present and return its dict-like content.

    Returns dict of arrays (or empty dict if not present).
    """
    fp = os.path.join(dirpath, 'templates.npz')
    if not os.path.exists(fp):
        return {}
    try:
        arr = np.load(fp)
        return {k: arr[k] for k in arr.files}
    except Exception:
        return {}
