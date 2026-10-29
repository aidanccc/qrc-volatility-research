"""Fit one noise scale per IonQ noise model so the local model matches the cloud simulator.

    .venv/bin/python -m analysis.ionq_cloud.calibrate --output results/ionq-noise-YYYY-MM-DD

The match statistic is the feature shrinkage: the through-origin slope of noisy
<Z> against noiseless <Z>, pooled over every cloud-check circuit for that noise
model (both models, both depths, all available history lengths). The local scale
multiplies all three calibration error rates (RXX, RY, readout). Bisection on
the scale; writes tables/cloud_matched_scales.json.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from hardware.paper_circuits import PAPER_MODELS
from analysis.noise.noise_model import z_expectations, apply_readout
from analysis.ionq_cloud.local import DEPTHS, readout_flip, simulate
from analysis.ionq_cloud.submit import CHECK_MONTHS


def shrink(noisy, clean):
    return float((clean * noisy).sum() / (clean ** 2).sum())


def local_shrink(noise, scale, check):
    noisy, clean = [], []
    for (model, depth, months), g in check.groupby(['model', 'depth', 'history_months']):
        prob = simulate(noise, model, depth, CHECK_MONTHS, history=months, scale=scale)
        flip = readout_flip(noise, scale)
        z = {(l, e + 1): z_expectations(apply_readout(prob[i, e], flip))
             for i, l in enumerate(CHECK_MONTHS) for e in range(PAPER_MODELS[model]['virtual_nodes'])}
        noisy += [z[(r.month, r.node)][r.qubit] for r in g.itertuples()]
        clean += list(g.noiseless)
    return shrink(np.array(noisy), np.array(clean))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    tables = parser.parse_args().output / 'tables'
    check = pd.read_csv(tables / 'cloud_check.csv')
    result = {'statistic': 'through-origin slope of noisy vs noiseless <Z>, pooled over cloud-check circuits',
              'scale': {}, 'target_shrink_cloud': {}, 'shrink_local_at_1x': {}, 'shrink_local_at_fit': {}}
    for noise in ['aria-1', 'forte-1']:
        c = check[check.noise == noise]
        target = shrink(c.cloud.to_numpy(), c.noiseless.to_numpy())
        lo, hi = 0.2, 6.0
        for _ in range(9):
            mid = (lo * hi) ** .5
            lo, hi = (mid, hi) if local_shrink(noise, mid, c) > target else (lo, mid)
        scale = round((lo * hi) ** .5, 2)
        result['scale'][noise] = scale
        result['target_shrink_cloud'][noise] = target
        result['shrink_local_at_1x'][noise] = local_shrink(noise, 1.0, c)
        result['shrink_local_at_fit'][noise] = local_shrink(noise, scale, c)
        print(noise, {k: v[noise] for k, v in result.items() if isinstance(v, dict)}, flush=True)
    (tables / 'cloud_matched_scales.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
