#!/usr/bin/env python3
"""Generate a simple templates.npz for the Cosmic_test uncertainty package.

Creates arrays: 'nominal', 'up', 'down', and 'bins'.
"""
import numpy as np
import os

outdir = os.path.dirname(__file__)
bins = np.linspace(95.0, 115.0, 51)  # 50 bins
centers = 0.5 * (bins[:-1] + bins[1:])

# simple smooth nominal shape (decaying linear + small Gaussian bump)
nominal = 0.5 + 0.02 * (centers - 95.0) + 2.0 * np.exp(-0.5 * ((centers - 102.5) / 1.2)**2)
nominal = nominal / np.sum(nominal)  # normalize to unity (relative shape)

# up/down variations (shape shifts)
up = nominal * (1.0 + 0.15 * np.sin((centers - 95.0) / 20.0 * np.pi))
down = nominal * (1.0 - 0.12 * np.cos((centers - 95.0) / 20.0 * np.pi))

# renormalize each to same integral as nominal
up = up / np.sum(up) * np.sum(nominal)
down = down / np.sum(down) * np.sum(nominal)

np.savez(os.path.join(outdir, 'templates.npz'), nominal=nominal, up=up, down=down, bins=bins)
print('Wrote templates.npz in', outdir)
