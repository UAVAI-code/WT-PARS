"""Run the actual WT-PARS core with a small CPU-only demonstration profile."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import platform
import sys
import time
import numpy as np
import scipy
import sklearn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.cross_unmixer import build_query_only_proxy_pool, match_to_anchors
from src.victim import VCAFCLSVictim
from src.weak_tangent_search import WeakTangentConfig, WeakTangentParetoSearch

def json_value(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['targeted','untargeted','both'], default='both')
    p.add_argument('--profile', choices=['demo','reference'], default='demo')
    p.add_argument('--input', type=Path, default=ROOT / 'data/demo_input.npz')
    p.add_argument('--reference', type=Path, help='Optional posthoc E/A truth; never passed to attack.')
    p.add_argument('--endmembers', type=int, default=3)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/demo')
    args = p.parse_args()
    with np.load(args.input, allow_pickle=False) as z: cube = z['cube'].astype(np.float32)
    if cube.ndim != 3 or not np.isfinite(cube).all() or cube.min() < 0 or cube.max() > 1:
        p.error('Input must contain a finite H x W x B cube in [0,1]. No automatic rescaling.')
    if not 2 <= args.endmembers <= min(cube.shape[-1], cube.shape[0]*cube.shape[1]):
        p.error('Invalid --endmembers.')
    modes = ['untargeted','targeted'] if args.mode == 'both' else [args.mode]
    if any((args.output / mode).exists() for mode in modes):
        p.error('Output mode directory exists; choose a fresh --output to preserve previous results.')
    for mode in modes:
        config_dict = json.loads((ROOT / f'configs/{mode}_{args.profile}.json').read_text())
        config_dict['seed'] = args.seed
        config = WeakTangentConfig(**config_dict)
        services = build_query_only_proxy_pool(ROOT / 'baselines', args.endmembers, fast=True, vca_seed_pairs=2)
        start = time.perf_counter()
        result = WeakTangentParetoSearch(services, config).run(cube)
        seconds = time.perf_counter() - start
        # Evaluation is performed only after search; this service is never queried by search.
        victim = VCAFCLSVictim(ROOT / 'baselines', args.endmembers, seed=17, n_vca_runs=4)
        clean, attacked = victim.query(cube), victim.query(result.attacked_cube)
        clean_a, clean_e, _ = match_to_anchors(result.canonical_endmembers, clean)
        attacked_a, attacked_e, _ = match_to_anchors(result.canonical_endmembers, attacked)
        delta = result.attacked_cube.astype(np.float64) - cube
        summary = dict(mode=mode, profile=args.profile, config=asdict(config),
            input_shape=list(cube.shape), attack_seconds=seconds, device='cpu',
            environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
            selected_objectives=result.selected.objectives,
            selected_measures=asdict(result.selected.measures), query_accounting=result.query_accounting,
            candidate_queries=result.candidate_queries, service_queries=result.service_queries,
            cache_hits=result.cache_hits, optimizer_evaluations=result.optimizer_evaluations,
            optimizer_generations=result.optimizer_generations, optimizer_implementation=result.optimizer_implementation,
            posthoc_service_calls=2, ground_truth_used_during_search=False,
            evaluation=dict(clean_reconstruction_sre_db=clean.reconstruction_sre_db,
                attacked_reconstruction_sre_db=attacked.reconstruction_sre_db,
                mean_abundance_l1_shift=float(np.abs(attacked_a-clean_a).sum(axis=0).mean())),
            distortion=dict(linf=float(np.abs(delta).max()), rms=float(np.sqrt(np.mean(delta**2)))))
        if args.reference:
            # This file is deliberately opened after generation, never by the optimizer.
            with np.load(args.reference, allow_pickle=False) as z:
                ref_e, ref_a = z['endmembers'], z['abundances']
            if ref_e.shape != clean.endmembers.shape or ref_a.shape != clean.abundances.shape:
                raise ValueError('Reference shapes do not match input and material count.')
            for name, output in [('clean',clean), ('attacked',attacked)]:
                a, _, _ = match_to_anchors(ref_e, output)
                summary['evaluation'][name+'_abundance_sre_db'] = float(20*np.log10((np.linalg.norm(ref_a)+1e-8)/(np.linalg.norm(a-ref_a)+1e-8)))
            summary['evaluation']['delta_sre_db'] = summary['evaluation']['clean_abundance_sre_db'] - summary['evaluation']['attacked_abundance_sre_db']
        assert np.isfinite(result.attacked_cube).all()
        assert result.attacked_cube.min() >= 0 and result.attacked_cube.max() <= 1
        assert summary['distortion']['linf'] <= config.linf_cap + 1e-6
        assert result.selected.measures.mean_sam <= config.mean_sam_cap + 1e-6
        assert result.optimizer_evaluations > 0 and result.optimizer_generations > 0
        assert 'torch' not in sys.modules, 'Demo unexpectedly imported torch.'
        dest = args.output / mode
        dest.mkdir(parents=True)
        np.savez_compressed(dest / 'arrays.npz', cube=result.attacked_cube,
            clean_abundances=clean_a, attacked_abundances=attacked_a,
            clean_endmembers=clean_e, attacked_endmembers=attacked_e,
            canonical_endmembers=result.canonical_endmembers, target_map=result.target_map,
            directions=result.directions, pareto_objectives=np.stack([r.objectives for r in result.pareto]))
        (dest / 'summary.json').write_text(json.dumps(summary, indent=2, default=json_value, allow_nan=False)+'\n')
        print(mode, json.dumps(summary['evaluation']), dest)

if __name__ == '__main__':
    main()
