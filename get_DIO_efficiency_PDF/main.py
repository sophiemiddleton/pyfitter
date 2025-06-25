import argparse
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
import awkward as ak

import fit_module
from cut_module import CutClass
from import_module import ImportClass

def stream_files(args):
    """Load one or more ROOT files (or a .txt list) and return a flat array
    of MC first‐surface momenta for all selected segments."""
    if args.file.endswith(".txt"):
        with open(args.file) as f:
            files = [ln.strip() for ln in f if ln.strip()]
    else:
        files = [args.file]

    branches_trk = ["trk", "trksegs", "trksegpars_lh", "trkqual"]
    branches_crv = ["crvsummary", "crvcoincs"]
    branches_mc  = ["trkmcsim", "trksegsmc"]

    cuts   = CutClass(args.cuts, True)
    mc_all = []

    for fp in files:
        print(f"Processing file: {fp}")
        mds    = ImportClass(fp, args.dirname, args.treename)
        arr_trk = mds.Import(branches_trk)
        arr_trk = mds.AddVectorMag(arr_trk, "trksegs", "mom")
        arr_crv = mds.Import(branches_crv)

        arr_mc  = mds.Import(branches_mc)
        arr_mc  = mds.AddVectorMag(arr_mc, "trksegsmc", "mom")

        arr_cut = cuts.ApplyCut(arr_trk, arr_crv)
        _, mc_np = extract_with_loops(arr_cut, arr_mc)
        mc_all.append(mc_np)

        # free memory
        del arr_trk, arr_crv, arr_mc, arr_cut

    return np.concatenate(mc_all) if mc_all else np.empty(0)

def extract_with_loops(array_cut, array_mc):
    """Flatten out all reconstructed‐segment vs MC‐segment momenta."""
    reco_nested = array_cut["trksegs", "mom.mag"].to_list()
    mc_nested   = array_mc["trksegsmc", "mom.mag"].to_list()

    reco_vals, mc_vals = [], []
    for reco_evt, mc_evt in zip(reco_nested, mc_nested):
        for itrk, reco_trk in enumerate(reco_evt):
            mc_trk   = mc_evt[itrk] if itrk < len(mc_evt) else []
            mc_first = mc_trk[0] if isinstance(mc_trk, list) and mc_trk else None
            for seg_mom in reco_trk:
                if seg_mom is None:
                    continue
                reco_vals.append(seg_mom)
                mc_vals.append(mc_first)

    reco_np = np.asarray(reco_vals, dtype=float)
    mc_np   = np.asarray([m if m is not None else np.nan for m in mc_vals], dtype=float)
    return reco_np, mc_np

def main(args):
    # 1) load only the MC momenta we need
    data_mc = stream_files(args)

    # 2) fit the efficiency PDF
    result_eff, PDF_eff, N_eff = fit_module.Unbinned_fit_efficiency(
        data_mc, (95, 104.97), degree=4)

    # 3) freeze & save the PDF object
    for p in PDF_eff.get_params(floating=True):
        p.float = False
    with open("efficiency_PDF.pkl", "wb") as f:
        pickle.dump(PDF_eff, f)

    # 4) record and print the fit summary
    print("Efficiency fit result:", result_eff)
    print("Message:", result_eff.message)
    print("Status:", result_eff.valid)
    with open("efficiency_fit_result.txt", "w") as f:
        f.write(str(result_eff))

    # 5) plot & save
    fit_module.plot_fit_result(data_mc, (95, 104.97), PDF_eff, N_eff)
    plt.savefig("plot_efficiency_fit_result.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit efficiency PDF")
    parser.add_argument("--file",     type=str, required=True,
                        help="ROOT file or .txt list of ROOT files")
    parser.add_argument("--dirname",  type=str, default="EventNtuple",
                        help="TDirectory name in the ROOT file")
    parser.add_argument("--treename", type=str, default="ntuple",
                        help="TTree name in the ROOT file")
    parser.add_argument("--cuts",     type=str, default="SU2020",
                        help="Cut configuration identifier")
    args = parser.parse_args()
    main(args)
