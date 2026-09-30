import argparse
from pathlib import Path

import awkward as ak
import matplotlib as mpl
import matplotlib.font_manager as mfm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import zfit

# Import publication style helpers (ensure mu2e_plot_style.py is in your PYTHONPATH)
from mu2e_plot_style import (
    OKABE_ITO,
    apply_publication_style,
    draw_watermark,
    style_axis,
    style_pull_axis,
)

# Set global publication styling
apply_publication_style()

# Set up font configuration
preferred_serifs = ['DejaVu Serif', 'Times New Roman', 'Times', 'Palatino']
available_fonts = {f.name for f in mfm.fontManager.ttflist}
chosen_serif = next((f for f in preferred_serifs if f in available_fonts), 'DejaVu Serif')

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': [chosen_serif],
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 9,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
})


# ==============================================================================
# 1. Custom PDF Definition (poly58 Background)
# ==============================================================================

# Physics Constants (Mu2e Standards 2025)
E_MAX = 104.97            # MeV
ALPHA = 1.0 / 137.035999  # Fine structure constant
M_E   = 0.510998          # Electron mass [MeV]
M_MU  = 105.194           # Muon mass [MeV]


class poly58(zfit.pdf.ZPDF):
    """5th to 8th order polynomial for DIO background (momentum space)."""
    _N_OBS = 1
    _PARAMS = ['a5', 'a6', 'a7', 'a8']

    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x)
        a5 = self.params['a5']
        a6 = self.params['a6']
        a7 = self.params['a7']
        a8 = self.params['a8']
        
        m_Al = 25133.0  # Mass of Aluminum atom [MeV]
        delta = tf.nn.relu(M_MU - x - (x**2) / (2.0 * m_Al))
        return a5 * delta**5 + a6 * delta**6 + a7 * delta**7 + a8 * delta**8


def make_poly58_pdf(obs, name="poly58_pdf"):
    """Factory function to instantiate poly58 with specified parameter bounds."""
    poly_params = {
        'a5': (8.97879e-17, 1e-17, 1e-16),
        'a6': (1.17169e-17, 1e-18, 1e-16),
        'a7': (-1.06599e-19, -1e-18, -1e-19),
        'a8': (8.14251e-20, 1e-20, 1e-19),
    }

    zpars = {}
    for p_name, (val, lower, upper) in poly_params.items():
        zpars[p_name] = zfit.Parameter(
            f"{p_name}_{name}",
            val,
            lower,
            upper,
        )

    return poly58(obs=obs, name=name, **zpars)


# ==============================================================================
# 2. DIO Parametric Unbinned Fitter Class (Single Component Background)
# ==============================================================================
class DIO:

    def __init__(self):
        self.print_prefix = "[DIO] "
        print(f"{self.print_prefix}Initialised")

    def fit_momentum(
        self,
        data_list,
        labels=None,
        pdf_func=None,
        out_file="DIOfit.pdf",
        normalize=False,
        target_yield=None,
    ):
        """Performs a single-component unbinned extended maximum likelihood fit.

        Args:
            data_list: List of momentum data arrays to fit.
            labels: List of dataset label strings.
            pdf_func: Optional callable returning the unextended PDF for a given observable.
            out_file: Path of the output plot file.
            normalize: If True, normalizes yields to 100k target events.
            target_yield: Explicit yield to normalize the histogram and fit to.
        """
        if labels is None:
            labels = ["Ext. DIO"]

        normalize = normalize or target_yield is not None

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(8, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            constrained_layout=True,
        )

        # Colors for publication clarity
        data_colors = [OKABE_ITO["black"], OKABE_ITO["blue"], OKABE_ITO["green"]]
        fit_colors = [OKABE_ITO["vermillion"], OKABE_ITO["orange"], OKABE_ITO["purple"]]

        last_norm = 0.0

        for i, data in enumerate(data_list):
            # --- 1. Clean Data ---
            mom_mag_skim = ak.nan_to_none(data)
            mom_mag_skim = ak.drop_none(mom_mag_skim)
            mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))

            obs_range = (100,110)
            obs_mom = zfit.Space("x", limits=obs_range)

            mom_np = mom_np[(mom_np >= obs_range[0]) & (mom_np <= obs_range[1])]
            n_events_raw = len(mom_np)

            if n_events_raw == 0:
                print(
                    f"{self.print_prefix}Warning: Dataset '{labels[i]}' has 0 events."
                )
                continue

            mom_zfit = zfit.Data.from_numpy(array=mom_np, obs=obs_mom)

            # --- 2. Construct Single-Component Extended PDF ---
            if pdf_func is None:
                unext_pdf = make_poly58_pdf(obs_mom, name=f"dio_bkg_{i}")
            else:
                unext_pdf = pdf_func(obs_mom)

            N_DIO = zfit.Parameter(
                f"N_DIO_{i}",
                n_events_raw,
                0,
                n_events_raw * 3,
                label=r"N_{\text{DIO}}",
            )
            fitcurve = unext_pdf.create_extended(N_DIO)

            # --- 3. Fit Execution ---
            nll = zfit.loss.ExtendedUnbinnedNLL(model=fitcurve, data=mom_zfit)
            minimizer = zfit.minimize.Minuit()

            result = minimizer.minimize(loss=nll)
            hesse_dict = result.hesse()
            print(f"\n{self.print_prefix}Fit result for {labels[i]}:")
            print(result)

            # --- 4. Visualisation Setup ---
            n_bins = 50
            bin_width = (obs_range[1] - obs_range[0]) / n_bins

            counts_raw, bins = np.histogram(
                mom_np, bins=n_bins, range=obs_range
            )
            data_bin_center = (bins[:-1] + bins[1:]) / 2

            # Determine normalization factor
            if normalize:
                target_events = target_yield if target_yield is not None else 100000.0
                norm_factor = target_events / n_events_raw
            else:
                norm_factor = 1.0

            counts_plot = counts_raw * norm_factor
            errors_raw = np.sqrt(counts_raw)
            errors_plot = errors_raw * norm_factor
            nonzero_mask = counts_raw > 0

            ax1.errorbar(
                data_bin_center[nonzero_mask],
                counts_plot[nonzero_mask],
                yerr=errors_plot[nonzero_mask],
                fmt="o",
                color=data_colors[i % len(data_colors)],
                markerfacecolor="white",
                markeredgecolor=data_colors[i % len(data_colors)],
                markersize=4,
                capsize=0,
                elinewidth=1,
                label=f"{labels[i]} simulated sample",
            )

            # --- 5. Plot Model Curve ---
            mom_plot = np.linspace(obs_range[0], obs_range[1], 500)
            data_plot = zfit.Data.from_numpy(array=mom_plot, obs=obs_mom)

            total_fitted_yield = result.params[N_DIO]["value"]

            pdf_vals = fitcurve.pdf(data_plot).numpy()
            total_curve_y = pdf_vals * total_fitted_yield * bin_width * norm_factor

            # --- 6. Residuals & Pull Calculation ---
            data_bin_center_zfit = zfit.Data.from_numpy(
                array=data_bin_center, obs=obs_mom
            )
            fit_at_bin_center = (
                fitcurve.pdf(data_bin_center_zfit).numpy().flatten()
                * total_fitted_yield
                * bin_width
            )

            chi2_mask = counts_raw > 0
            chi2_val = np.sum(
                ((counts_raw[chi2_mask] - fit_at_bin_center[chi2_mask]) ** 2)
                / counts_raw[chi2_mask]
            )
            ndf = np.count_nonzero(chi2_mask) - len(result.params)

            param_val = total_fitted_yield
            param_err = hesse_dict.get(N_DIO, {}).get("error", 0.0)

            # Construct legend label with stats included
            fit_label = (
                f"{labels[i]} Fit\n"
                f"$N_{{DIO}} = {param_val:.1f} \\pm {param_err:.1f}$\n"
                f"$\\chi^2 / \\text{{ndf}} = {chi2_val:.1f} / {ndf}$"
            )

            ax1.plot(
                mom_plot,
                total_curve_y,
                color=fit_colors[i % len(fit_colors)],
                linestyle="-",
                linewidth=2.0,
                label=fit_label,
            )

            residual = counts_plot - fit_at_bin_center * norm_factor
            pull_errors = np.where(counts_raw > 0, errors_plot, 1.0 * norm_factor)
            pull = residual / pull_errors

            # Blue vertical bars for the pull panel
            ax2.bar(
                data_bin_center,
                pull,
                width=bin_width * 0.8,
                color=OKABE_ITO["blue"],
                align="center",
            )

            last_norm = total_fitted_yield

        # --- 7. Apply Publication Styling & Formatting ---
        unit_suffix = " [A.U.]" if normalize else ""
        ax1.set_ylabel(f"Events / {bin_width:.2f} MeV/$c${unit_suffix}")

        ax1.legend(
            frameon=False,
            loc="upper right",
            fontsize=13,
            handlelength=2.0,
        )

        draw_watermark(ax1, loc="left")

        ax2.axhline(0, color="gray", linestyle="--", linewidth=1.0)
        ax2.set_ylabel(r"Pull [$\sigma$]")
        ax2.set_xlabel(r"Reconstructed Momentum [MeV/$c$]")
        ax2.set_ylim(-3.5, 3.5)

        style_axis(ax1)
        style_pull_axis(ax2)

        plt.savefig(out_file)
        plt.show()

        return last_norm


# ==============================================================================
# 3. Parquet Execution Runner
# ==============================================================================
def run_dio_fit_from_parquet(
    parquet_file_path,
    sample_label="Internal DIO",
    tag="",
    outdir=".",
    normalize=False,
    target_yield=None,
):
    print(f"Loading dataset from: {parquet_file_path}")
    df = pd.read_parquet(parquet_file_path)

    mom_array = ak.Array(df["momentum"].to_numpy())

    suffix = f"_{tag}" if tag else ""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    dio_fitter = DIO()
    fitted_yield = dio_fitter.fit_momentum(
        data_list=[mom_array],
        labels=[sample_label],
        pdf_func=make_poly58_pdf,
        out_file=str(out_path / f"DIOfit{suffix}.pdf"),
        normalize=normalize,
        target_yield=target_yield,
    )
    print(f"Fit complete. Total fitted DIO yield: {fitted_yield:.1f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the DIO momentum fit on a parquet dataset."
    )
    parser.add_argument(
        "-p", "--parquet", required=True, help="Path to the input parquet file."
    )
    parser.add_argument(
        "-t",
        "--tag",
        default="",
        help="Tag appended to the output fit file name, e.g. DIOfit_<tag>.pdf.",
    )
    parser.add_argument(
        "-o", "--outdir", default=".", help="Directory for the output fit files."
    )
    parser.add_argument(
        "-l", "--label", default="Ext. DIO", help="Sample label shown on the plot."
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize yields to 100k events."
    )
    parser.add_argument(
        "-n",
        "--target-yield",
        type=float,
        default=None,
        help="Rescale the histogram and fit curve to this total yield (overrides --normalize).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_dio_fit_from_parquet(
        parquet_file_path=args.parquet,
        sample_label=args.label,
        tag=args.tag,
        outdir=args.outdir,
        normalize=args.normalize,
        target_yield=args.target_yield,
    )