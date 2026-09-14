"""Required query adapters extracted without changing their bodies."""
from __future__ import annotations
from pathlib import Path
from typing import Protocol
import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment, nnls
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from .victim import EPS, SharedVCAProjection, UnmixingResult, VCAFCLSVictim, _load_baseline_module


class QueryVictim(Protocol):
    """A black-box unmixing service used only through ``query``."""

    name: str

    def query(self, cube: np.ndarray) -> UnmixingResult: ...


def spectral_angle_matrix(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Pairwise spectral angles for B x K reference and candidate matrices."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    numerator = reference.T @ candidate
    denominator = np.linalg.norm(reference, axis=0)[:, None] * np.linalg.norm(candidate, axis=0)[None, :] + EPS
    return np.arccos(np.clip(numerator / denominator, -1.0, 1.0))


def match_to_anchors(anchors: np.ndarray, result: UnmixingResult) -> tuple[np.ndarray, np.ndarray, float]:
    """Align a service output to fixed material anchors by minimum SAD."""
    anchors = np.asarray(anchors, dtype=np.float32)
    if anchors.ndim != 2:
        raise ValueError("anchors must have shape B x K")
    if result.endmembers.shape[1] != anchors.shape[1]:
        raise ValueError("The service and canonical anchors must use the same material count.")
    cost = spectral_angle_matrix(anchors, result.endmembers)
    rows, cols = linear_sum_assignment(cost)
    order = cols[np.argsort(rows)]
    return result.abundances[order], result.endmembers[:, order], float(np.mean(cost[rows, cols]))


class NMFFCLSVictim:
    """A deterministic NMF-endmember plus FCLS abundance service.

    It deliberately differs from VCA in endmember extraction while retaining a
        lightweight constrained abundance stage, making it practical for repeated
        black-box queries on a workstation.
    """

    name = "nmf_fcls"

    def __init__(
        self,
        baseline_root: str | Path,
        n_endmembers: int,
        seed: int = 31,
        max_iter: int = 220,
        init: str = "nndsvda",
    ) -> None:
        self.n_endmembers = int(n_endmembers)
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.init = str(init)
        _, self._solve_abundances = _load_baseline_module(Path(baseline_root))

    def query(self, cube: np.ndarray) -> UnmixingResult:
        cube = np.asarray(cube, dtype=np.float32)
        if cube.ndim != 3:
            raise ValueError(f"Expected H x W x B cube, got {cube.shape}")
        h, w, bands = cube.shape
        spectra = cube.reshape(-1, bands)
        # NMF requires nonnegative input. Reflectance products are nonnegative;
        # the shift keeps the adapter robust to numerically centred test cubes.
        offset = float(min(np.min(spectra), 0.0))
        shifted = np.maximum(spectra - offset, 0.0)
        model = NMF(
            n_components=self.n_endmembers,
            init=self.init,
            solver="cd",
            beta_loss="frobenius",
            max_iter=self.max_iter,
            tol=1e-4,
            random_state=self.seed,
        )
        with warnings.catch_warnings():
            # In a query loop, a deterministic bounded iteration count is more
            # important than fully converging an auxiliary black-box service.
            warnings.simplefilter("ignore", ConvergenceWarning)
            coefficients = model.fit_transform(shifted)

        # NMF is invariant to reciprocal component scaling: W H =
        # (W diag(q)) (diag(q)^-1 H).  Passing raw H to an ASC-constrained
        # solver therefore gives arbitrary results.  Estimate q from the data
        # alone so that the rescaled NMF coefficients obey the abundance
        # sum-to-one constraint as closely as possible, then refine abundances
        # with exact FCLS.  No reference endmembers or abundance truth enter
        # this calibration.
        asc_scale, _ = nnls(coefficients.astype(np.float64), np.ones(coefficients.shape[0]))
        positive = asc_scale[asc_scale > EPS]
        fallback = float(np.median(positive)) if positive.size else 1.0
        asc_scale = np.where(asc_scale > EPS, asc_scale, fallback)
        endmembers = (model.components_.T / asc_scale[None, :]).astype(np.float32)
        abundance = self._solve_abundances(spectra.T, endmembers, solver="fcls_exact")
        reconstruction = (endmembers.astype(np.float64) @ abundance.astype(np.float64)).T.reshape(h, w, bands)
        error = spectra - reconstruction.reshape(-1, bands)
        relative_residual = float(np.mean(np.linalg.norm(error, axis=1) / (np.linalg.norm(spectra, axis=1) + EPS)))
        reconstruction_sre_db = float(
            20.0 * np.log10((np.linalg.norm(spectra) + EPS) / (np.linalg.norm(error) + EPS))
        )
        return UnmixingResult(
            endmembers=endmembers,
            abundances=abundance.reshape(self.n_endmembers, h, w).astype(np.float32),
            reconstruction=reconstruction.astype(np.float32),
            relative_residual=relative_residual,
            reconstruction_sre_db=reconstruction_sre_db,
        )


def build_query_only_proxy_pool(
    baseline_root: str | Path,
    n_endmembers: int,
    *,
    include_nmf: bool = False,
    include_fcls: bool = True,
    include_nnls: bool = True,
    fast: bool = False,
    vca_seed_pairs: int = 1,
) -> dict[str, QueryVictim]:
    """Build blind query services without exposing a reference material matrix.

    The returned identifiers are opaque bookkeeping keys.  The response-guided
    optimiser never branches on them and receives no endmember library.  Each
    service estimates its own endmembers from the submitted cube.
    """
    if int(vca_seed_pairs) < 1:
        raise ValueError("vca_seed_pairs must be positive.")
    shared_projection = SharedVCAProjection(baseline_root, n_endmembers)
    services: dict[str, QueryVictim] = {}
    service_index = 0
    fcls_seeds = (11, 53, 89, 131)
    nnls_seeds = (29, 71, 107, 149)
    for pair in range(int(vca_seed_pairs)):
        # Deterministically extend the lists if a larger stress-test ensemble
        # is requested.  These seeds remain opaque to the optimiser.
        fcls_seed = fcls_seeds[pair] if pair < len(fcls_seeds) else 11 + 42 * pair
        nnls_seed = nnls_seeds[pair] if pair < len(nnls_seeds) else 29 + 42 * pair
        if include_fcls:
            services[f"service_{service_index}"] = VCAFCLSVictim(
                baseline_root,
                n_endmembers,
                seed=fcls_seed,
                n_vca_runs=2 if fast else 4,
                solver="fcls_exact",
                shared_projection=shared_projection,
            )
            service_index += 1
        if include_nnls:
            services[f"service_{service_index}"] = VCAFCLSVictim(
                baseline_root,
                n_endmembers,
                seed=nnls_seed,
                n_vca_runs=2 if fast else 4,
                solver="nnls",
                shared_projection=shared_projection,
            )
            service_index += 1
    if include_nmf:
        services[f"service_{service_index}"] = NMFFCLSVictim(
            baseline_root,
            n_endmembers,
            seed=31,
            max_iter=80 if fast else 300,
        )
    if not services:
        raise ValueError("At least one admitted query service is required.")
    return services
