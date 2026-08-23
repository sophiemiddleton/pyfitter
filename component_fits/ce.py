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
        self, data_list, start, end, opt, label, nbins
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
            N_Flat = zfit.Parameter(f"N_Flat_{i}", len(mom_np), 100, 10000000)
            mu = zfit.Parameter(f"mu_{i}", 104.0, 100.0, 106.0)
            sigma = zfit.Parameter(f"sigma_{i}", 0.35, 0.05, 3.0)
            alphal = zfit.Parameter(f"alphal_{i}", 1.5, 0.1, 10.0)
            nl = zfit.Parameter(f"nl_{i}", 3.0, 0.1, 30.0)
            alphar = zfit.Parameter(f"alphar_{i}", 1.5, 0.1, 10.0)
            nr = zfit.Parameter(f"nr_{i}", 3.0, 0.1, 30.0)

            ax1.set_yscale("log")
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
            fit_range = (obs_mom.lower[0, 0], obs_mom.upper[0, 0])
            bin_width = (fit_range[1] - fit_range[0]) / nbins

            # --- 1. Fit Curve Projection ---
            mom_plot = np.linspace(fit_range[0], fit_range[1], 500).reshape(-1, 1)
            dscb_model_curve = (
                zfit.run(dscb_model.ext_pdf(mom_plot)) * bin_width
            )

            ax1.plot(
                mom_plot.flatten(),
                dscb_model_curve.flatten(),
                color=OKABE_ITO["vermillion"],
                linestyle="-",
                linewidth=2.0,
                label="DSCB Fit",
            )

            # --- 2. Data Histograms & Errorbars ---
            counts, bins = np.histogram(mom_np, bins=nbins, range=fit_range)
            data_bin_center = (bins[:-1] + bins[1:]) / 2
            errors = np.sqrt(counts)
            nonzero_mask = counts > 0

            # Statistical Uncertainty Markers
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
                label="Stat. unc.",
            )

            # Data Histogram Step Line
            ax1.hist(
                mom_np,
                bins=nbins,
                range=fit_range,
                color=OKABE_ITO["black"],
                histtype="step",
                linewidth=1.2,
                alpha=0.7,
                label="Simulation",
            )

            # --- 3. Dynamic Axis Range & Labels ---
            ax1.set_ylabel(f"Events / {bin_width:.2f} MeV/$c$")

            min_y = max(0.1, np.min(counts[nonzero_mask]) * 0.2)
            max_y = np.max(counts[nonzero_mask]) * 10.0
            ax1.set_ylim(min_y, max_y)

            # Watermark & Legends
            draw_watermark(ax1, loc="left")
            ax1.legend(loc="upper right", frameon=False, fontsize=12)

            # --- 4. Fit Parameter Text Box ---
            param_text = (
                f"$\\mu = {result.params[mu]['value']:.3f} \\pm {hesse_errors[mu]['error']:.3f}$\\n"
                f"$\\sigma = {result.params[sigma]['value']:.3f} \\pm {hesse_errors[sigma]['error']:.3f}$\\n"
                f"$\\alpha_{{l}} = {result.params[alphal]['value']:.2f} \\pm {hesse_errors[alphal]['error']:.2f}$\\n"
                f"$n_{{l}} = {result.params[nl]['value']:.2f} \\pm {hesse_errors[nl]['error']:.2f}$\\n"
                f"$\\alpha_{{r}} = {result.params[alphar]['value']:.2f} \\pm {hesse_errors[alphar]['error']:.2f}$\\n"
                f"$n_{{r}} = {result.params[nr]['value']:.2f} \\pm {hesse_errors[nr]['error']:.2f}$"
            )
            props = dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.85,
                edgecolor="gray",
                linewidth=0.8,
            )
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

            # --- 5. Pull Plot Calculations ---
            data_bin_center_2d = data_bin_center.reshape(-1, 1)
            fit_at_bin_center = (
                zfit.run(dscb_model.ext_pdf(data_bin_center_2d)) * bin_width
            )

            valid_mask = counts > 0
            residual = counts - fit_at_bin_center
            pull = np.divide(
                residual,
                errors,
                where=valid_mask,
                out=np.zeros_like(counts, dtype=np.float64),
            )

            ax2.bar(
                data_bin_center[valid_mask],
                pull[valid_mask],
                width=bin_width * 0.8,
                color=OKABE_ITO["black"],
                align="center",
                alpha=0.6,
            )

            norm = result.params[N_Flat]["value"]

        # Pull Axis Styling
        ax2.axhline(0, color="gray", linestyle="--", linewidth=1.0)
        ax2.set_ylabel(r"Pull [$\sigma$]")
        ax2.set_xlabel(label)
        ax2.set_ylim(-3.5, 3.5)

        # Apply standard axis formatting helpers
        style_axis(ax1)
        style_pull_axis(ax2)

        # Log-scale minor tick formatting adjustments
        ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1.0))

        plt.savefig("CE_dscb.pdf")
        plt.show()

        return norm


# ==============================================================================
# 2. Parquet Execution Runner
# ==============================================================================
def run_ce_dscb_fit_from_parquet(
    parquet_file_path,
    momentum_column="momentum",
    start_mom=100.0,
    end_mom=106.0,
    nbins=60,
    label=r"Reconstructed Momentum [MeV/$c$]",
):
    """Loads momentum data from a parquet file and executes the DSCB fit.

    Args:
        parquet_file_path (str): Path to post-cut parquet file.
        momentum_column (str): Name of momentum column in the dataframe.
        start_mom (float): Lower fit boundary (MeV/c).
        end_mom (float): Upper fit boundary (MeV/c).
        nbins (int): Number of histogram bins.
        label (str): X-axis label for plot.
    """
    print(f"Loading parquet dataset from: {parquet_file_path}")
    df = pd.read_parquet(parquet_file_path)

    if momentum_column not in df.columns:
        raise KeyError(
            f"Column '{momentum_column}' not found in {parquet_file_path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Convert pandas series to Awkward Array
    mom_array = ak.Array(df[momentum_column].to_numpy())

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
    )

    print(f"Fit complete. Extracted raw fitted yield: {fitted_yield:.1f}")


if __name__ == "__main__":
    # Update file path to your dataset location
    parquet_filename = "ce_postcut.parquet"

    run_ce_dscb_fit_from_parquet(
        parquet_file_path=parquet_filename,
        momentum_column="momentum",
        start_mom=100.0,
        end_mom=106.0,
        nbins=60,
        label=r"Reconstructed Momentum [MeV/$c$]",
    )