#!/usr/bin/env python3
"""
Waterfall plot generator for systematic uncertainties.

Shows how individual systematics stack to build total uncertainty.
Publication-ready plots for both Phase 2A and 2B results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class WaterfallPlotter:
    """Generate waterfall plots for systematic uncertainties."""
    
    def __init__(self, output_dir: str = 'uncertainties/outputs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_results(self, results_file: str) -> Dict:
        """Load results from JSON file (from Phase 2A or 2B)."""
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def build_waterfall_data(self, results: Dict, stat_uncertainty: float = 0.0) -> Tuple[List[str], List[float], List[float]]:
        """
        Convert results dict to waterfall plot data.
        
        Returns:
            labels: List of systematic names
            values: Individual systematic impacts
            cumulative: Cumulative uncertainty at each step
        """
        # Group results by systematic (average ±/- directions)
        impacts = defaultdict(list)
        for key, res in results.items():
            if res is None:
                continue
            sys_name = res.get('systematic', key.split('_')[0])
            shift = abs(res.get('poi_shift', 0))
            impacts[sys_name].append(shift)
        
        # Average each systematic (if both +/- available)
        avg_impacts = {sys: np.mean(shifts) for sys, shifts in impacts.items()}
        
        # Sort by impact (largest first)
        sorted_impacts = sorted(avg_impacts.items(), key=lambda x: x[1], reverse=True)
        
        # Build cumulative
        labels = []
        values = []
        cumulative = [stat_uncertainty]  # Start with stat
        
        total_syst_sq = stat_uncertainty ** 2
        
        for sys_name, impact in sorted_impacts:
            labels.append(sys_name)
            values.append(impact)
            total_syst_sq += impact ** 2
            cumulative.append(np.sqrt(total_syst_sq))
        
        return labels, values, cumulative
    
    def plot_waterfall(self, results: Dict, stat_uncertainty: float = 0.0, 
                      title: str = "Systematic Uncertainty Waterfall", 
                      output_file: Optional[str] = None) -> Path:
        """
        Create waterfall plot showing systematic stacking.
        
        Args:
            results: Results dict from Phase 2A or 2B
            stat_uncertainty: Statistical uncertainty baseline
            title: Plot title
            output_file: Where to save (defaults to waterfall.png)
        
        Returns:
            Path to saved figure
        """
        if output_file is None:
            output_file = self.output_dir / 'waterfall.png'
        else:
            output_file = self.output_dir / output_file
        
        labels, values, cumulative = self.build_waterfall_data(results, stat_uncertainty)
        
        if not labels:
            print("No systematic results to plot")
            return output_file
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # X positions
        x = np.arange(len(labels))
        width = 0.6
        
        # Colors: stat = blue, systematics = orange, total = green
        colors = ['orange'] * len(labels)
        
        # Plot bars
        bars = ax.bar(x, values, width, label='Individual systematics', 
                      color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
        
        # Plot cumulative line
        cumulative_x = np.arange(len(cumulative)) - 0.5
        ax.plot(cumulative_x, cumulative, 'g-o', linewidth=2.5, markersize=8, 
               label='Cumulative uncertainty', zorder=5)
        
        # Add horizontal line at stat uncertainty
        ax.axhline(stat_uncertainty, color='b', linestyle='--', linewidth=2, 
                  label=f'Stat. unc. (±{stat_uncertainty:.2f})', alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Systematic Uncertainty Source', fontsize=12, fontweight='bold')
        ax.set_ylabel('Uncertainty (events)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        ax.legend(fontsize=11, loc='upper left')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Add cumulative values at line points
        for i, cum in enumerate(cumulative):
            ax.text(i - 0.5, cum + 5, f'{cum:.2f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color='green')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Waterfall plot saved to {output_file}")
        return output_file
    
    def plot_comparison(self, results_2a: Dict, results_2b: Dict, 
                       stat_uncertainty: float = 0.0,
                       output_file: Optional[str] = None) -> Path:
        """
        Create side-by-side comparison of Phase 2A vs 2B impacts.
        
        Args:
            results_2a: Results from constraint-based (Phase 2A)
            results_2b: Results from profile likelihood (Phase 2B)
            stat_uncertainty: Statistical uncertainty
            output_file: Where to save
        
        Returns:
            Path to saved figure
        """
        if output_file is None:
            output_file = self.output_dir / 'waterfall_comparison.png'
        else:
            output_file = self.output_dir / output_file
        
        labels_2a, values_2a, cum_2a = self.build_waterfall_data(results_2a, stat_uncertainty)
        labels_2b, values_2b, cum_2b = self.build_waterfall_data(results_2b, stat_uncertainty)
        
        # Combine labels (union of both)
        all_labels = sorted(set(list(labels_2a) + list(labels_2b)))
        
        # Align values
        vals_2a_aligned = [next((v for l, v in zip(labels_2a, values_2a) if l == lab), 0.0) 
                          for lab in all_labels]
        vals_2b_aligned = [next((v for l, v in zip(labels_2b, values_2b) if l == lab), 0.0) 
                          for lab in all_labels]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
        x = np.arange(len(all_labels))
        width = 0.35
        
        # Phase 2A
        ax1.bar(x - width/2, vals_2a_aligned, width, label='Phase 2A impacts',
               color='steelblue', edgecolor='black', linewidth=1, alpha=0.8)
        ax1.axhline(stat_uncertainty, color='b', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Stat. unc. (±{stat_uncertainty:.2f})')
        ax1.set_ylabel('Uncertainty (events)', fontsize=12, fontweight='bold')
        ax1.set_title('Phase 2A: Constraint-Based Systematics', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_labels, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3, linestyle=':')
        ax1.legend(fontsize=10)
        
        # Phase 2B
        ax2.bar(x - width/2, vals_2b_aligned, width, label='Phase 2B impacts',
               color='darkorange', edgecolor='black', linewidth=1, alpha=0.8)
        ax2.axhline(stat_uncertainty, color='b', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Stat. unc. (±{stat_uncertainty:.2f})')
        ax2.set_title('Phase 2B: Profile Likelihood', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_labels, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3, linestyle=':')
        ax2.legend(fontsize=10)
        
        plt.suptitle('Systematic Impact Comparison: Phase 2A vs 2B', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison plot saved to {output_file}")
        return output_file


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate waterfall plots for systematic uncertainties')
    parser.add_argument('--results-file', type=str, default='uncertainties/outputs/profile_systematic_results.json',
                       help='Results file to plot')
    parser.add_argument('--stat-unc', type=float, default=0.0,
                       help='Statistical uncertainty (events)')
    parser.add_argument('--title', type=str, default='Systematic Uncertainty Waterfall',
                       help='Plot title')
    parser.add_argument('--output', type=str, default='waterfall.png',
                       help='Output filename')
    parser.add_argument('--compare-with', type=str, default=None,
                       help='Second results file for comparison plot')
    
    args = parser.parse_args()
    
    plotter = WaterfallPlotter()
    
    # Load primary results
    if not Path(args.results_file).exists():
        print(f"Error: {args.results_file} not found")
        exit(1)
    
    results = plotter.load_results(args.results_file)
    
    # Generate main plot
    plotter.plot_waterfall(results, args.stat_unc, args.title, args.output)
    
    # Generate comparison if requested
    if args.compare_with and Path(args.compare_with).exists():
        results_2 = plotter.load_results(args.compare_with)
        plotter.plot_comparison(results, results_2, args.stat_unc)
