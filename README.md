# WT-PARS

Minimal research code for **Weak-Tangent Pareto Adaptive Region Search (WT-PARS)**:
targeted and untargeted black-box perturbations for hyperspectral unmixing.

This repository provides the attack core, the project MOEA/D optimizer,
lightweight query services, and one small procedural example. It is a runnable
method demonstration, **not a reproduction package for every paper table**.
No paper acceptance, publication DOI, or benchmark superiority is implied.

## Quick start

Use Python 3.10 or 3.11 in a separate environment. Validation used Python 3.10.20.
From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py --mode both --reference data/demo_reference.npz
python -m unittest discover -s tests -v
```

On Windows, activate with `.venv\Scripts\activate` instead. The demo uses CPU
and one numerical-library thread; CUDA, PyTorch, pretrained networks, and cloud
services are not required. Results are saved under `outputs/demo/`. Existing
results are not overwritten; rerun with a new `--output outputs/run2`.
Linux was tested; macOS and Windows were not tested on this release.

## What the method does

1. Query several unmixing services on the clean cube. Align their returned
   endmember spectra to a common ordering, without using ground truth.
2. Find abundance redistribution directions with small spectral footprints
   across the responses. Combine these directions with data-derived spectral
   atoms to represent spatially varying perturbations.
3. Search for perturbations with an effectiveness objective and a distortion
   objective. Regional proposals and structure-aware MOEA/D refinement explore
   this trade-off while enforcing the configured input constraints.
4. Select a Pareto candidate and evaluate its output on a separate service.

Both attack modes use the same search implementation. The untargeted mode seeks
abundance changes; the targeted mode seeks redistribution from canonical
component 0 to component 1 within the top 20% source region. These are anonymous
components inferred from clean service outputs, **not ground-truth material
labels**. The target region is not computed from reference abundances.

The default query pool contains VCA-style extraction followed by FCLS or NNLS,
with two fixed initializations per solver (four endpoints, not four independent
model families). The demo evaluation service uses another extraction seed, 17,
with four extraction runs and FCLS. Its clean and attacked settings are identical.
This small example does not establish transfer across held-out model families.

## Configuration and custom inputs

`configs/*_demo.json` uses two regional steps and one MOEA/D generation with
population 6 and neighborhood 3. This intentionally small budget exercises the
actual optimization code. `maximum_candidate_queries=0` means no extra candidate
cap; the configured step and generation counts still bound the run.

`configs/*_reference.json` preserves the saved targeted/untargeted configuration
templates used to initialize the project's later transfer study: 48 regional
steps, MOEA/D population 20, neighborhood 5 and nine generations. These templates
alone do not reproduce the study's dataset, variant, seed and budget overrides.
They are not universal optimal settings or a claim of equal budgets across modes.
Use `--profile reference` to run these templates; expect more computation.
The demo changes only search size, initial block size and the extra candidate cap.
Every effective dataclass setting is written to `summary.json`.

For an input saved as `np.savez_compressed('input.npz', cube=cube)`:

```bash
python run_demo.py --mode untargeted --input input.npz --endmembers 4 --output outputs/custom
```

Input is finite float32-compatible **H x W x B** data normalized to [0,1]; no
automatic rescaling is performed. Supply the assumed material count explicitly.
Reference spectra and abundances are optional and used only after search for
evaluation. They must have shapes B x K and K x H x W, respectively.

To connect another model, implement `QueryVictim.query(cube)` returning
`UnmixingResult`, then pass a dictionary of services to
`WeakTangentParetoSearch(services, config).run(cube)`. The attack observes returned
endmembers and abundances; it does not inspect model weights or gradients.
Use `configs` as explicit starting points instead of relying on legacy defaults.
Only `proposed_moead` is bundled as a runnable optimizer; comparison optimizers
listed in the preserved registry require an external runtime not in this package.

## Outputs and interpretation

- `arrays.npz`: attacked cube, clean/attacked evaluation abundances and spectra,
  canonical spectra, target mask, directions and Pareto objective pairs. Evaluation
  outputs are aligned to canonical clean responses, not reference material names.
- `summary.json`: effective configuration, environment, selected objective values,
  distortion, query ledger and evaluation metrics.
- Optional abundance SRE is calculated after independently matching each evaluation
  output to reference endmembers. `delta_sre_db = clean SRE - attacked SRE`; positive
  values indicate degradation. SRE loss alone is not proof of targeted success;
  inspect directional measures and the target mask as well.
- `candidate_queries` counts distinct evaluated perturbations. Legacy
  `service_queries` counts search calls only. Use
  `query_accounting.total_service_calls` for clean initialization plus search and
  selection calls. A service call is one cube sent to one endpoint; cache hits and
  local fixed-dictionary calculations are reported separately. Two posthoc victim
  calls are excluded from the search ledger and recorded separately.
- `attack_seconds` measures local attack elapsed time, excluding victim evaluation.
  The unmodified ledger has no internal timer; its runtime fields remain empty.
  This demo timing is not a controlled performance benchmark.

Input numerical/spectral similarity does not establish physical realizability.
The included tests and data demonstrate a digital algorithmic workflow only.

## Data, provenance and license

The bundled 24 x 24 x 32, three-component scene is generated analytically by
`generate_demo.py`. It contains no third-party measured spectra and is **not**
Syn-K3 or a real scene used in the paper. See [data/README.md](data/README.md).
Regenerate with `python generate_demo.py --output data_regenerated`.

The attack, MOEA/D, query ledger and victim core files are copied unchanged from
a frozen project snapshot. Only required response-alignment and query adapter
definitions are extracted, with their function bodies preserved. Provenance is
recorded in `PROVENANCE.json`; file checksums are in `MANIFEST.sha256`.

Project code and the procedural example are provided under the [MIT License](LICENSE).
The MOEA/D implementation adapts parts of `mopt`; its original copyright and MIT
license are retained in [licenses/mopt-MIT.txt](licenses/mopt-MIT.txt).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Installed dependencies retain
their own licenses. No third-party benchmark dataset is relicensed here.

If this work is useful, a GitHub star or follow is appreciated. Please cite the
associated paper when using the method in research; bibliographic details will
be added when a public record is available. This is a scholarly request, not an
additional restriction on the MIT License.
