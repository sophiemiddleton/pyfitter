import argparse
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
import awkward as ak

import fit_module
from import_module import ImportClass
from cut_module import CutClass

def stream_mc(args):
    """Load ROOT file(s) and return a flat NumPy array of MC first-surface momenta."""
    if args.file.endswith(".txt"):
        with open(args.file) as f:
            files = [ln.strip() for ln in f if ln.strip()]
    else:
        files = [args.file]

    branches_trk = ["trk", "trksegs", "trksegpars_lh", "trkqual"]
    branches_crv = ["crvsummary", "crvcoincs"]
    branches_mc  = ["trkmcsim", "trksegsmc"]

    cuts   = CutClass(args.cuts, True)
    all_mc = []

    for fp in files:
        mds    = ImportClass(fp, args.dirname, args.treename)
        arr_trk = mds.Import(branches_trk)
        arr_trk = mds.AddVectorMag(arr_trk, "trksegs", "mom")
        arr_crv = mds.Import(branches_crv)

        arr_mc  = mds.Import(branches_mc)
        arr_mc  = mds.AddVectorMag(arr_mc, "trksegsmc", "mom")

        arr_cut = cuts.ApplyCut(arr_trk, arr_crv)
        _, mc_np = extract_with_loops(arr_cut, arr_mc)
        all_mc.append(mc_np)

        del arr_trk, arr_crv, arr_mc, arr_cut

    return np.concatenate(all_mc) if all_mc else np.empty(0)

def extract_with_loops(array_cut, array_mc):
    """Flatten out reconstructed vs MC momenta and return (reco, mc) arrays."""
    reco_nested = array_cut["trksegs", "mom.mag"].to_list()
    mc_nested   = array_mc["trksegsmc", "mom.mag"].to_list()

    reco_vals, mc_vals = [], []
    for reco_evt, mc_evt in zip(reco_nested, mc_nested):
        for i, reco_trk in enumerate(reco_evt):
            mc_trk   = mc_evt[i] if i < len(mc_evt) else []
            mc_first = mc_trk[0] if isinstance(mc_trk, list) and mc_trk else None
            for seg in reco_trk:
                if seg is None: continue
                reco_vals.append(seg)
                mc_vals.append(mc_first)

    reco_np = np.array(reco_vals, dtype=float)
    mc_np   = np.array([m if m is not None else np.nan for m in mc_vals],
                       dtype=float)
    return reco_np, mc_np

def main(args):
    # 1) stream only MC momenta
    data_mc = stream_mc(args)

    # 2) fit efficiency PDF
    result, pdf, N = fit_module.Unbinned_fit_efficiency(
        data_mc, (95, 104.97), degree=4)

    # 3) freeze parameters & pickle
    for p in pdf.get_params(floating=True):
        p.float = False
    with open("efficiency_PDF.pkl", "wb") as f:
        pickle.dump(pdf, f)

    # 4) print summary
    print("Fit status:", result.valid)
    print("Message:  ", result.message)
    print("Parameters:\n", result)

    with open("efficiency_fit_result.txt", "w") as f:
        f.write(str(result))

    # 5) plot & save
    fit_module.plot_fit_result(data_mc, (95, 104.97), pdf, N)
    plt.savefig("efficiency_fit_result.png")
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Efficiency-only fit")
    p.add_argument("--file",     required=True,
                   help="ROOT file or .txt listing ROOT files")
    p.add_argument("--dirname",  default="EventNtuple",
                   help="TDirectory name")
    p.add_argument("--treename", default="ntuple",
                   help="TTree name")
    p.add_argument("--cuts",     default="SU2020",
                   help="Cuts tag for CutClass")
    args = p.parse_args()
    main(args)