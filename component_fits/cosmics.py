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

# Set up font configuration
preferred_serifs = ['DejaVu Serif', 'Times New Roman', 'Times', 'Palatino']
available_fonts = {f.name for f in mfm.fontManager.ttflist}
chosen_serif = next((f for f in preferred_serifs if f in available_fonts), 'DejaVu Serif')

apply_publication_style()

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


class Cosmics():
    """Class to conduct comparisons between cut or data sets"""

    def __init__(self):
        # Custom prefix for log messages from this processor
        self.print_prefix = "[Compare] "
        print(f"{self.print_prefix}Initialised")

    def fit_time(
        self,
        data_list,
        labels,
        deg=1,
        normalize=False,
        out_file="CosmicTime.pdf",
        target_yield=None,
    ):
        """Plots the reconstructed time data and its statistical uncertainties

        using an extended unbinned maximum likelihood fit with a Chebyshev polynomial shape.

        Args:
            data_list: List of time data arrays to fit.
            labels: List of dataset label strings.
            deg: Degree of the Chebyshev polynomial (default is 1).
            normalize: If True, normalizes yields to 100k target events. If False (default), plots raw counts.
            out_file: Path of the output plot file.
            target_yield: Explicit yield to normalize the histogram and fit to (overrides the 100k default).
        """        
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(8, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            constrained_layout=True,
        )

        data_colors = [OKABE_ITO["black"], OKABE_ITO["blue"]]
        fit_colors = [OKABE_ITO["vermillion"], OKABE_ITO["purple"]]

        fit_range = (480, 700)
        n_bins = 50
        bin_width = (fit_range[1] - fit_range[0]) / n_bins
        normalize = normalize or target_yield is not None

        last_norm = 0.0
        for i, data in enumerate(data_list):
            time_skim = ak.nan_to_none(data)
            time_skim = ak.drop_none(time_skim)

            obs_time = zfit.Space("x", limits=fit_range)
            time_np = ak.to_numpy(ak.flatten(time_skim, axis=None))
            time_zfit = zfit.Data.from_numpy(array=time_np, obs=obs_time)
            n_events_raw = len(time_np)

            if n_events_raw == 0:
                continue

            N_Cosmic = zfit.Parameter(
                f"N_Cosmic_time_{i}", n_events_raw, 100, n_events_raw * 10
            )

            # Set up Chebyshev polynomial parameters
            cheby_coeffs = [
                zfit.Parameter(f"c{c_idx}_{i}", 0.0, -1.0, 1.0)
                for c_idx in range(deg)
            ]
            fitcurve = zfit.pdf.Chebyshev(
                obs=obs_time, coeffs=cheby_coeffs, extended=N_Cosmic
            )

            nll = zfit.loss.ExtendedUnbinnedNLL(model=fitcurve, data=time_zfit)
            minimizer = zfit.minimize.Minuit()
            result = minimizer.minimize(loss=nll)
            hesse_errors = result.hesse()

            # Determine normalization factor
            if normalize:
                target_events = target_yield if target_yield is not None else 100000.0
                norm_factor = target_events / n_events_raw
            else:
                norm_factor = 1.0

            counts_raw, bins = np.histogram(
                time_np, bins=n_bins, range=fit_range
            )
            data_bin_center = (bins[:-1] + bins[1:]) / 2

            counts_plot = counts_raw * norm_factor
            errors_plot = np.sqrt(counts_raw) * norm_factor
            nonzero_mask = counts_raw > 0

            # Data Points
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
                label=f"{labels[i]} sample",
            )

            # Histogram Step Shape
            weights = np.full_like(time_np, norm_factor)
            ax1.hist(
                time_np,
                bins=n_bins,
                range=fit_range,
                weights=weights,
                color=data_colors[i % len(data_colors)],
                histtype="step",
                linewidth=1.2,
                alpha=0.7,
            )

            # Fit Model Curve
            time_plot = np.linspace(fit_range[0], fit_range[1], 500).reshape(-1, 1)
            fitcurve_curve = (
                zfit.run(fitcurve.ext_pdf(time_plot)) * bin_width * norm_factor
            )

            # Chi2 / ndf calculation
            fit_at_bin_center_raw = (
                zfit.run(fitcurve.ext_pdf(data_bin_center.reshape(-1, 1)))
                * bin_width
            )
            chi2_mask = counts_raw > 0
            chi2_val = np.sum(
                ((counts_raw[chi2_mask] - fit_at_bin_center_raw[chi2_mask]) ** 2)
                / counts_raw[chi2_mask]
            )
            ndf = np.count_nonzero(chi2_mask) - len(result.params)

            # Format Chebyshev parameter values for the legend
            coeff_strings = []
            for c_param in cheby_coeffs:
                val = result.params[c_param]["value"]
                err = hesse_errors.get(c_param, {}).get("error", 0.0)
                c_name = c_param.name.split("_")[0]
                coeff_strings.append(f"${c_name} = {val:.4f} \\pm {err:.4f}$")

            coeff_legend_text = "\n".join(coeff_strings)
            fit_label = (
                f"{labels[i]} Fit (Chebyshev deg {deg})\n"
                f"{coeff_legend_text}\n"
                f"$\\chi^2 / \\text{{ndf}} = {chi2_val:.1f} / {ndf}$"
            )

            ax1.plot(
                time_plot.flatten(),
                fitcurve_curve.flatten(),
                color=fit_colors[i % len(fit_colors)],
                linestyle="-",
                linewidth=2.0,
                label=fit_label,
            )

            # Residual Pulls
            residual_plot = counts_plot - (fit_at_bin_center_raw * norm_factor)
            pull_errors = np.where(counts_raw > 0, errors_plot, 1.0 * norm_factor)
            pull = residual_plot / pull_errors

            ax2.bar(
                data_bin_center,
                pull,
                width=bin_width * 0.8,
                color=data_colors[i % len(data_colors)],
                align="center",
                alpha=0.6,
            )

            last_norm = result.params[N_Cosmic]["value"]

        # Y-axis unit adjustment based on normalization flag
        if normalize:
            ax1.set_ylabel(f"Events / {bin_width:.1f} ns [A.U.]")
        else:
            ax1.set_ylabel(f"Events / {bin_width:.1f} ns")

        # Headroom expansion for top-right legend
        y_max = ax1.get_ylim()[1]
        ax1.set_ylim(0, y_max * 1.25)

        # Clean top-right 2-column legend
        ax1.legend(
            frameon=False,
            loc="upper right",
            ncols=2,
            fontsize=11,
            handlelength=1.5,
            columnspacing=1.2,
            labelspacing=0.4,
        )

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

    def fit_momentum(
        self,
        data_list,
        labels,
        normalize=False,
        out_file="CosmicMomentum.pdf",
        target_yield=None,
        poly_order=1,
    ):
        """Plots the reconstructed momentum data and its statistical uncertainties

        using an extended unbinned maximum likelihood fit with a Chebyshev polynomial shape,
        including goodness of fit and pull distribution.

        Args:
            data_list: List of momentum data arrays to fit
            labels: List of dataset label strings
            normalize: If True, normalizes yields to 100k target events. If False (default), plots raw counts.
            out_file: Path of the output plot file.
            target_yield: Explicit yield to normalize the histogram and fit to (overrides the 100k default).
            poly_order: Chebyshev polynomial order for the momentum fit (1 or 2).
        """
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(8, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            constrained_layout=True,
        )

        data_colors = [OKABE_ITO["black"], OKABE_ITO["blue"]]
        fit_colors = [OKABE_ITO["vermillion"], OKABE_ITO["purple"]]

        fit_range = (100, 110)
        n_bins = 50
        bin_width = (fit_range[1] - fit_range[0]) / n_bins
        normalize = normalize or target_yield is not None

        last_norm = 0.0
        for i, data in enumerate(data_list):
            mom_mag_skim = ak.nan_to_none(data)
            mom_mag_skim = ak.drop_none(mom_mag_skim)

            obs_mom = zfit.Space("x", limits=fit_range)
            mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))
            mom_zfit = zfit.Data.from_numpy(array=mom_np, obs=obs_mom)
            n_events_raw = len(mom_np)

            if n_events_raw == 0:
                continue

            if poly_order not in (1, 2):
                raise ValueError(f"poly_order must be 1 or 2, got {poly_order}")

            N_Cosmic = zfit.Parameter(
                f"N_Cosmic_mom_{i}", n_events_raw, 100, n_events_raw * 10
            )
            coeffs = [
                zfit.Parameter(f"c{idx + 1}_{i}", 0.1, -1, 1)
                for idx in range(poly_order)
            ]

            poly_model = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=coeffs, extended=N_Cosmic)

            nll = zfit.loss.ExtendedUnbinnedNLL(model=poly_model, data=mom_zfit)
            minimizer = zfit.minimize.Minuit()
            result = minimizer.minimize(loss=nll)
            hesse_errors = result.hesse()

            # Determine normalization factor
            if normalize:
                target_events = target_yield if target_yield is not None else 100000.0
                norm_factor = target_events / n_events_raw
            else:
                norm_factor = 1.0

            counts_raw, bins = np.histogram(
                mom_np, bins=n_bins, range=fit_range
            )
            data_bin_center = (bins[:-1] + bins[1:]) / 2

            counts_plot = counts_raw * norm_factor
            errors_plot = np.sqrt(counts_raw) * norm_factor
            nonzero_mask = counts_raw > 0

            # Data Points
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
                label=f"{labels[i]} sample",
            )

            # Histogram Step Shape
            weights = np.full_like(mom_np, norm_factor)
            ax1.hist(
                mom_np,
                bins=n_bins,
                range=fit_range,
                weights=weights,
                color=data_colors[i % len(data_colors)],
                histtype="step",
                linewidth=1.2,
                alpha=0.7,
            )

            # Fit Model Curve
            mom_plot = np.linspace(fit_range[0], fit_range[1], 500).reshape(-1, 1)
            fitcurve_curve = (
                zfit.run(poly_model.ext_pdf(mom_plot)) * bin_width * norm_factor
            )

            # Chi2 / ndf calculation
            fit_at_bin_center_raw = (
                zfit.run(poly_model.ext_pdf(data_bin_center.reshape(-1, 1)))
                * bin_width
            )
            chi2_mask = counts_raw > 0
            chi2_val = np.sum(
                ((counts_raw[chi2_mask] - fit_at_bin_center_raw[chi2_mask]) ** 2)
                / counts_raw[chi2_mask]
            )
            ndf = np.count_nonzero(chi2_mask) - len(result.params)

            coeff_strings = []
            for c_param in coeffs:
                c_val = result.params[c_param]["value"]
                c_err = hesse_errors.get(c_param, {}).get("error", 0.0)
                c_name = c_param.name.split("_")[0]
                coeff_strings.append(f"${c_name} = {c_val:.4f} \\pm {c_err:.4f}$")

            # Extended statistics label formatted for the 2-column legend
            fit_label = (
                f"{labels[i]} Fit (Chebyshev order {poly_order})\n"
                f"{chr(10).join(coeff_strings)}\n"
                f"$\\chi^2 / \\text{{ndf}} = {chi2_val:.1f} / {ndf}$"
            )

            ax1.plot(
                mom_plot.flatten(),
                fitcurve_curve.flatten(),
                color=fit_colors[i % len(fit_colors)],
                linestyle="-",
                linewidth=2.0,
                label=fit_label,
            )

            # Residual Pulls
            residual_plot = counts_plot - (fit_at_bin_center_raw * norm_factor)
            pull_errors = np.where(counts_raw > 0, errors_plot, 1.0 * norm_factor)
            pull = residual_plot / pull_errors

            ax2.bar(
                data_bin_center,
                pull,
                width=bin_width * 0.8,
                color=data_colors[i % len(data_colors)],
                align="center",
                alpha=0.6,
            )

            last_norm = result.params[N_Cosmic]["value"]

        # Y-axis unit adjustment based on normalization flag
        if normalize:
            ax1.set_ylabel(f"Events / {bin_width:.2f} MeV/c [A.U.]")
        else:
            ax1.set_ylabel(f"Events / {bin_width:.2f} MeV/c")

        # Headroom expansion for top-right legend
        y_max = ax1.get_ylim()[1]
        ax1.set_ylim(0, y_max * 1.25)

        # Clean top-right 2-column legend
        ax1.legend(
            frameon=False,
            loc="upper right",
            ncols=2,
            fontsize=11,
            handlelength=1.5,
            columnspacing=1.2,
            labelspacing=0.4,
        )

        draw_watermark(ax1, loc="left")

        ax2.axhline(0, color="gray", linestyle="--", linewidth=1.0)
        ax2.set_ylabel(r"Pull [$\sigma$]")
        ax2.set_xlabel("Reconstructed Momentum [MeV/c]")
        ax2.set_ylim(-3.5, 3.5)

        style_axis(ax1)
        style_pull_axis(ax2)

        plt.savefig(out_file)
        plt.show()

        return last_norm


def run_fits_from_parquet(
    parquet_file_path,
    sample_label="Cosmics",
    deg_time=1,
    normalize=False,
    tag="",
    outdir=".",
    target_yield=None,
    poly_order=1,
):
    """Reads a parquet file and runs the time and momentum fits using Cosmics.

    Args:
        parquet_file_path (str): Path to parquet file generated by write_parquet_output.
        sample_label (str): Label string to display on plots.
        deg_time (int): Chebyshev polynomial degree for the time fit.
        normalize (bool): Normalize data yield to 100k events if True.
        tag (str): Suffix appended to the output fit file names.
        outdir (str): Directory in which to write the output fit files.
        target_yield (float | None): Explicit yield to normalize the histogram and fit to.
        poly_order (int): Chebyshev polynomial order for the momentum fit (1 or 2).
    """
    print(f"Loading parquet dataset from: {parquet_file_path}")
    df = pd.read_parquet(parquet_file_path)

    # Wrap Series to Awkward arrays for compatibility with drop_none / flatten logic inside Cosmics
    mom_array = ak.Array(df["momentum"].to_numpy())
    time_array = ak.Array(df["time"].to_numpy())

    cosmic_analyzer = Cosmics()

    suffix = f"_{tag}" if tag else ""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Running time fit...")
    cosmic_analyzer.fit_time(
        data_list=[time_array],
        labels=[sample_label],
        deg=deg_time,
        normalize=normalize,
        out_file=str(out_path / f"CosmicTime{suffix}.pdf"),
        target_yield=target_yield,
    )

    print("Running momentum fit...")
    cosmic_analyzer.fit_momentum(
        data_list=[mom_array],
        labels=[sample_label],
        normalize=normalize,
        out_file=str(out_path / f"CosmicMomentum{suffix}.pdf"),
        target_yield=target_yield,
        poly_order=poly_order,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cosmic time and momentum fits on a parquet dataset."
    )
    parser.add_argument(
        "-p", "--parquet", required=True, help="Path to the input parquet file."
    )
    parser.add_argument(
        "-t",
        "--tag",
        default="",
        help="Tag appended to the output fit file names, e.g. CosmicTime_<tag>.pdf.",
    )
    parser.add_argument(
        "-o", "--outdir", default=".", help="Directory for the output fit files."
    )
    parser.add_argument(
        "-l", "--label", default="Cosmic MC", help="Sample label shown on the plots."
    )
    parser.add_argument(
        "--deg-time", type=int, default=1, help="Chebyshev degree for the time fit."
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize yields to 100k events."
    )
    parser.add_argument(
        "--poly-order",
        type=int,
        choices=[1, 2],
        default=1,
        help="Chebyshev polynomial order for the cosmic momentum fit (1 or 2).",
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

    run_fits_from_parquet(
        parquet_file_path=args.parquet,
        sample_label=args.label,
        deg_time=args.deg_time,
        normalize=args.normalize,
        tag=args.tag,
        outdir=args.outdir,
        target_yield=args.target_yield,
        poly_order=args.poly_order,
    )