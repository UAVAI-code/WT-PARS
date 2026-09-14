"""Generate a small procedural scene; no measured spectral library is used."""
from pathlib import Path
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'data')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / f).exists() for f in ('demo_input.npz', 'demo_reference.npz')):
        parser.error('Output already exists; choose a new --output directory.')
    rng = np.random.default_rng(20260914)
    h, w, bands, k = 24, 24, 32, 3
    x = np.linspace(0, 1, bands)
    e = np.stack([0.20 + 0.30*x + 0.10*np.exp(-((x-0.30)/0.16)**2),
                  0.25 + 0.25*x - 0.10*np.exp(-((x-0.65)/0.13)**2),
                  0.15 + 0.42*x + 0.08*np.exp(-((x-0.75)/0.18)**2)], axis=1)
    fields = np.stack([gaussian_filter(rng.normal(size=(h,w)), 3) for _ in range(k)])
    fields = fields / fields.std(axis=(1,2), keepdims=True)
    a = np.exp(fields - fields.max(axis=0, keepdims=True))
    a /= a.sum(axis=0, keepdims=True)
    for j, (r,c) in enumerate([(3,3), (3,18), (18,10)]):
        a[:,r:r+3,c:c+3] = 0
        a[j,r:r+3,c:c+3] = 1
    cube = np.einsum('bk,khw->hwb', e, a)
    cube = np.clip(cube + rng.normal(0, .001, cube.shape), 0, 1)
    np.savez_compressed(args.output / 'demo_input.npz', cube=cube.astype('float32'))
    np.savez_compressed(args.output / 'demo_reference.npz', endmembers=e.astype('float32'),
                        abundances=a.astype('float32'), names=np.array(['demo_material_1','demo_material_2','demo_material_3']))
    print(args.output)

if __name__ == '__main__':
    main()
