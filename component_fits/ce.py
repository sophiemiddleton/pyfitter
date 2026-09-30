import argparse
from pathlib import Path

import awkward as ak
import matplotlib as mpl
import matplotlib.font_manager as mfm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import zfit

# Import publication style helpers (ensure mu2e_plot_style.py is in your PYTHONPATH)
from mu2e_plot_style import (
    OKABE_ITO,
    apply_publication_style,
    draw_watermark,
    style_axis,
    style_pull_axis,
)

# Set up publication-style matplotlib typography & aesthetics
preferred_serifs = ["DejaVu Serif", "Times New Roman", "Times", "Palatino"]
available_fonts = {f.name for f in mfm.fontManager.ttflist}
chosen_serif = next(
    (f for f in preferred_serifs if f in available_fonts), "DejaVu Serif"
)

apply_publication_style()

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [chosen_serif],
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
    }
)


# ==============================================================================
# 1. Conversion Electron (CE) DSCB Fitter Class
# ==============================================================================
class ConversionElectronFitter:
    """Class to perform parametric extended unbinned fits using a Double-Sided Crystal Ball (DSCB)."""

    def __init__(self):
        self.print_prefix = "[CE Fitter] "
        print(f"{self.print_prefix}Initialised")

    def fit_CELL_momentum_dscb(
        self, data_list, start, end, opt, label, nbins, out_file="CE_dscb.pdf"
    ):
        """Fits a Double-Sided Crystal Ball shape to the reconstructed momentum data
        using an extended unbinned maximum likelihood fit.
        """
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(8, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            constrained_layout=True,
        )

        norm = 0.0
        for i, data in enumerate(data_list):
            mom_mag_skim = ak.drop_none(ak.nan_to_none(data))
            mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))

            if len(mom_np) == 0:
                continue

            # Define observable space and zfit data
            obs_mom = zfit.Space("x", limits=(start, end))
            mom_zfit = zfit.Data.from_numpy(array=mom_np, obs=obs_mom)

            # Fit parameters for DSCB and extended yield
            N_Flat = zfit.Parameter(f"N_Flat_{i}", float(len(mom_np)), 10.0, 10000000.0)
            mu = zfit.Parameter(f"mu_{i}", 104.0, start, end)
            sigma = zfit.Parameter(f"sigma_{i}", 0.35, 0.05, 3.0)
            alphal = zfit.Parameter(f"alphal_{i}", 1.5, 0.1, 10.0)
            nl = zfit.Parameter(f"nl_{i}", 3.0, 0.1, 30.0)
            alphar = zfit.Parameter(f"alphar_{i}", 1.5, 0.1, 10.0)
            nr = zfit.Parameter(f"nr_{i}", 3.0, 0.1, 30.0)

            #ax1.set_yscale("log")
            dscb_model = zfit.pdf.DoubleCB(
                mu=mu,
                sigma=sigma,
                alphal=alphal,
                nl=nl,
                alphar=alphar,
                nr=nr,
                obs=obs_mom,
                extended=N_Flat,
            )

            # Minimization & Hesse Errors
            nll = zfit.loss.ExtendedUnbinnedNLL(model=dscb_model, data=mom_zfit)
            minimizer = zfit.minimize.Minuit()
            result = minimizer.minimize(loss=nll)
            hesse_errors = result.hesse()

            # --- Binning and Geometry Setup ---
            fit_range = (start, end)
            bin_width = (fit_range[1] - fit_range[0]) / nbins

            # Extract parameter values and uncertainties safely
            yield_val = result.params[N_Flat]["value"]
            mu_val, mu_err = result.params[mu]["value"], hesse_errors[mu]["error"]
            sig_val, sig_err = result.params[sigma]["value"], hesse_errors[sigma]["error"]
            al_val, al_err = result.params[alphal]["value"], hesse_errors[alphal]["error"]
            nl_val, nl_err = result.params[nl]["value"], hesse_errors[nl]["error"]
            ar_val, ar_err = result.params[alphar]["value"], hesse_errors[alphar]["error"]
            nr_val, nr_err = result.params[nr]["value"], hesse_errors[nr]["error"]

            # --- 1. Fit Curve Projection ---
            mom_plot = np.linspace(fit_range[0], fit_range[1], 1000)
            pdf_values = dscb_model.pdf(mom_plot).numpy()
            dscb_model_curve = pdf_values * yield_val * bin_width

            ax1.plot(
                mom_plot,
                dscb_model_curve,
                color=OKABE_ITO["vermillion"],
                linestyle="-",
                linewidth=2.0,
                label="DSCB Fit",
                zorder=3,
            )

            # --- 2. Data Histograms & Errorbars ---
            counts, bins = np.histogram(mom_np, bins=nbins, range=fit_range)
            data_bin_center = (bins[:-1] + bins[1:]) / 2
            errors = np.sqrt(counts)
            nonzero_mask = counts > 0

            # Plot binned simulation data
            ax1.errorbar(
                data_bin_center[nonzero_mask],
                counts[nonzero_mask],
                yerr=errors[nonzero_mask],
                fmt="o",
                color=OKABE_ITO["black"],
                markerfacecolor="white",
                markeredgecolor=OKABE_ITO["black"],
                markersize=4,
                capsize=0,
                elinewidth=1,
                label="CELL Mix simulated sample",
                zorder=2,
            )

            # --- 3. Dynamic Axis Range & Labels ---
            ax1.set_ylabel(f"Events / {bin_width:.2f} MeV/$c$")

            min_y = max(0.1, np.min(counts[nonzero_mask]) * 0.2)
            max_y = np.max(counts[nonzero_mask]) *1.3
            ax1.set_ylim(min_y, max_y)

            # Watermark & Legends
            draw_watermark(ax1, loc="left")
            ax1.legend(loc="upper right", frameon=False, fontsize=12)

            # --- 4. Fit Parameter Text Box ---
            param_text = (
                f"$\\mu = {mu_val:.3f} \\pm {mu_err:.3f}$ MeV/$c$\n"
                f"$\\sigma = {sig_val:.3f} \\pm {sig_err:.3f}$ MeV/$c$\n"
                f"$\\alpha_{{l}} = {al_val:.2f} \\pm {al_err:.2f}$\n"
                f"$n_{{l}} = {nl_val:.2f} \\pm {nl_err:.2f}$\n"
                f"$\\alpha_{{r}} = {ar_val:.2f} \\pm {ar_err:.2f}$\n"
                f"$n_{{r}} = {nr_val:.2f} \\pm {nr_err:.2f}$"
            )
            props = dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.85,
                edgecolor="gray",
                linewidth=0.8,
            )
            """
            ax1.text(
                0.05,
                0.05,
                param_text,
                transform=ax1.transAxes,
                fontsize=11,
                horizontalalignment="left",
                verticalalignment="bottom",
                bbox=props,
            )
            """
            # --- 5. Pull Plot Calculations ---
            fit_at_bin_center = (
                dscb_model.pdf(data_bin_center).numpy() * yield_val * bin_width
            )

            residual = counts - fit_at_bin_center
            pull = np.divide(
                residual,
                fit_at_bin_center,
                out=np.zeros_like(residual, dtype=np.float64),
                where=fit_at_bin_center > 0,
            )

            ax2.bar(
                data_bin_center[nonzero_mask],
                pull[nonzero_mask],
                width=bin_width * 0.8,
                color=OKABE_ITO["blue"],
                align="center",
                alpha=0.6,
            )

            norm = yield_val

        # Pull Axis Styling
        ax2.axhline(0, color="gray", linestyle="--", linewidth=1.0)
        ax2.set_ylabel(r"$(N_{\mathrm{data}} - N_{\mathrm{fit}}) / N_{\mathrm{fit}}$")
        ax2.set_xlabel(label)
        ax2.set_ylim(-0.5,0.5)

        # Apply standard axis formatting helpers
        style_axis(ax1)
        style_pull_axis(ax2)

        # Log-scale minor tick formatting adjustments
        ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1.0))

        plt.savefig(out_file)
        plt.show()

        return norm

    def fit_time_exponential(
        self,
        data_list,
        labels,
        start,
        end,
        nbins=50,
        out_file="CE_time.pdf",
        normalize=False,
        target_yield=None,
    ):
        """Fits reconstructed CE times with an extended exponential model.

        The exponential slope is initialized to -1/864 ns^-1.
        """
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(8, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            constrained_layout=True,
        )

        fit_range = (start, end)
        bin_width = (end - start) / nbins
        normalize = normalize or target_yield is not None
        data_colors = [OKABE_ITO["black"],OKABE_ITO["blue"]]
        fit_colors = [OKABE_ITO["vermillion"], OKABE_ITO["purple"]]
        last_norm = 0.0

        for i, data in enumerate(data_list):
            time_skim = ak.drop_none(ak.nan_to_none(data))
            time_np = ak.to_numpy(ak.flatten(time_skim, axis=None))
            time_np = time_np[(time_np >= start) & (time_np <= end)]
            n_events_raw = len(time_np)

            if n_events_raw == 0:
                print(f"{self.print_prefix}Warning: Dataset '{labels[i]}' has 0 events.")
                continue

            obs_time = zfit.Space("x", limits=fit_range)
            time_zfit = zfit.Data.from_numpy(array=time_np, obs=obs_time)
            n_ce = zfit.Parameter(f"N_CE_time_{i}", n_events_raw, 0, n_events_raw * 10)
            lam = zfit.Parameter(f"lambda_CE_time_{i}", -0.001131, -1.0, -1e-6)
            fitcurve = zfit.pdf.Exponential(obs=obs_time, lam=lam, extended=n_ce)

            nll = zfit.loss.ExtendedUnbinnedNLL(model=fitcurve, data=time_zfit)
            result = zfit.minimize.Minuit().minimize(loss=nll)
            hesse_errors = result.hesse()

            if normalize:
                scaled_yield = target_yield if target_yield is not None else 100000.0
                norm_factor = scaled_yield / n_events_raw
            else:
                norm_factor = 1.0

            counts_raw, bins = np.histogram(time_np, bins=nbins, range=fit_range)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            counts_plot = counts_raw * norm_factor
            errors_plot = np.sqrt(counts_raw) * norm_factor
            nonzero_mask = counts_raw > 0

            ax1.errorbar(
                bin_centers[nonzero_mask],
                counts_plot[nonzero_mask],
                yerr=errors_plot[nonzero_mask],
                fmt="o",
                color=data_colors[i % len(data_colors)],
                markerfacecolor="white",
                markeredgecolor=data_colors[i % len(data_colors)],
                markersize=4,
                capsize=0,
                elinewidth=1,
                label=f"{labels[i]} sample",
            )

            time_plot = np.linspace(start, end, 500).reshape(-1, 1)
            fit_curve = zfit.run(fitcurve.ext_pdf(time_plot)) * bin_width * norm_factor
            fit_at_bins_raw = (
                zfit.run(fitcurve.ext_pdf(bin_centers.reshape(-1, 1))) * bin_width
            )
            chi2_mask = counts_raw > 0
            chi2_val = np.sum(
                (counts_raw[chi2_mask] - fit_at_bins_raw[chi2_mask]) ** 2
                / counts_raw[chi2_mask]
            )
            ndf = np.count_nonzero(chi2_mask) - len(result.params)
            lam_value = result.params[lam]["value"]
            lam_error = hesse_errors.get(lam, {}).get("error", 0.0)

            ax1.plot(
                time_plot.flatten(),
                fit_curve.flatten(),
                color=fit_colors[i % len(fit_colors)],
                linewidth=2.0,
                label=(
                    f"{labels[i]} Fit\n"
                    f"$\\lambda = {lam_value:.6f} \\pm {lam_error:.6f}$ ns$^{{-1}}$\n"
                    #f"$\\chi^2 / \\text{{ndf}} = {chi2_val:.1f} / {ndf}$"
                ),
            )

            pull_errors = np.where(counts_raw > 0, errors_plot, norm_factor)
            pull = (counts_plot - fit_at_bins_raw * norm_factor) / pull_errors
            ax2.bar(
                bin_centers,
                pull,
                width=bin_width * 0.8,
                color=data_colors[i % len(data_colors)],
                align="center",
                alpha=0.6,
            )
            last_norm = result.params[n_ce]["value"]

        unit_suffix = " [A.U.]" if normalize else ""
        ax1.set_ylabel(f"Events / {bin_width:.1f} ns{unit_suffix}")
        ax1.legend(frameon=False, loc="upper right", fontsize=11)
        draw_watermark(ax1, loc="left")
        ax2.axhline(0, color="gray", linestyle="--", linewidth=1.0)
        ax2.set_ylabel(r"Pull [$\sigma$]")
        ax2.set_xlabel("Track Time [ns]")
        ax2.set_ylim(-3.5, 3.5)
        style_axis(ax1)
        style_pull_axis(ax2)
        plt.savefig(out_file)
        plt.show()

        return last_norm


# ==============================================================================
# 2. Parquet Execution Runner
# ==============================================================================
def run_ce_dscb_fit_from_parquet(
    parquet_file_path,
    momentum_column="momentum",
    start_mom=95.0,
    end_mom=110.0,
    nbins=60,
    label=r"Reconstructed Momentum [MeV/$c$]",
    tag="",
    outdir=".",
):
    """Loads momentum data from a parquet file and executes the DSCB fit.

    Args:
        parquet_file_path (str): Path to post-cut parquet file.
        momentum_column (str): Name of momentum column in dataframe.
        start_mom (float): Lower fit boundary (MeV/c).
        end_mom (float): Upper fit boundary (MeV/c).
        nbins (int): Number of histogram bins.
        label (str): X-axis label for plot.
        tag (str): Suffix appended to output file name.
        outdir (str): Directory where PDF file will be saved.
    """
    print(f"Loading parquet dataset from: {parquet_file_path}")
    df = pd.read_parquet(parquet_file_path)

    if momentum_column not in df.columns:
        raise KeyError(
            f"Column '{momentum_column}' not found in {parquet_file_path}. "
            f"Available columns: {list(df.columns)}"
        )

    suffix = f"_{tag}" if tag else ""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = str(out_path / f"CE_dscb{suffix}.pdf")

    mom_array = ak.Array(df[momentum_column].to_numpy())
    time_array = ak.Array(df["time"].to_numpy())

    fitter = ConversionElectronFitter()

    print(
        f"Executing DSCB fit in range ({start_mom}, {end_mom}) MeV/c with {nbins} bins..."
    )
    fitted_yield = fitter.fit_CELL_momentum_dscb(
        data_list=[mom_array],
        start=start_mom,
        end=end_mom,
        opt=None,
        label=label,
        nbins=nbins,
        out_file=out_file,
    )
    fitter.fit_time_exponential(
        data_list=[time_array],
        labels=["Trakc Time [ns]"],
        start=475,
        end=1650,
        nbins=50,
        out_file="fits/CE_time_v80_1d.pdf"
    )

    print(f"Fit complete. Extracted raw fitted yield: {fitted_yield:.1f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run conversion electron (CE) Double-Sided Crystal Ball fit on a parquet dataset."
    )
    parser.add_argument(
        "-p", "--parquet", required=True, help="Path to the input parquet file."
    )
    parser.add_argument(
        "-t",
        "--tag",
        default="",
        help="Tag appended to output fit file names, e.g. CE_dscb_<tag>.pdf.",
    )
    parser.add_argument(
        "-o", "--outdir", default=".", help="Directory for the output fit files."
    )
    parser.add_argument(
        "-m",
        "--momentum-column",
        default="momentum",
        help="Column name for momentum in the dataset.",
    )
    parser.add_argument(
        "--range",
        type=float,
        nargs=2,
        default=[100, 110],
        metavar=("START", "END"),
        help="Fit boundaries in MeV/c (default: 100.0 106.0).",
    )
    parser.add_argument(
        "-b",
        "--nbins",
        type=int,
        default=60,
        help="Number of histogram bins (default: 60).",
    )
    parser.add_argument(
        "-l",
        "--label",
        default=r"Reconstructed Momentum [MeV/$c$]",
        help="X-axis label for the plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_ce_dscb_fit_from_parquet(
        parquet_file_path=args.parquet,
        momentum_column=args.momentum_column,
        start_mom=args.range[0],
        end_mom=args.range[1],
        nbins=args.nbins,
        label=args.label,
        tag=args.tag,
        outdir=args.outdir,
    )