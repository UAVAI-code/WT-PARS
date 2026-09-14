"""Deterministic query adapter for lightweight hyperspectral unmixing victims."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EPS = 1e-8


@dataclass
class UnmixingResult:
    endmembers: np.ndarray  # B x K
    abundances: np.ndarray  # K x H x W
    reconstruction: np.ndarray | None  # H x W x B; omitted for compact queries
    relative_residual: float
    reconstruction_sre_db: float


def _load_baseline_module(baseline_root: Path):
    package_root = str(baseline_root)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from hsi_trust_evidence.unmixing import estimate_endmembers_vca, solve_abundances

    return estimate_endmembers_vca, solve_abundances


class SharedVCAProjection:
    """One-input cache for the seed-independent VCA projection."""

    def __init__(self, baseline_root: str | Path, n_endmembers: int) -> None:
        package_root = str(Path(baseline_root))
        if package_root not in sys.path:
            sys.path.insert(0, package_root)
        from hsi_trust_evidence.unmixing import prepare_vca_projection

        self.n_endmembers = int(n_endmembers)
        self._prepare = prepare_vca_projection
        self._cube: np.ndarray | None = None
        self._projection: tuple[np.ndarray, np.ndarray] | None = None

    def get(self, cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if cube is not self._cube:
            bands = cube.shape[-1]
            spectra = np.moveaxis(cube, -1, 0).reshape(bands, -1)
            self._projection = self._prepare(spectra, self.n_endmembers)
            self._cube = cube
        assert self._projection is not None
        return self._projection


class VCAFCLSVictim:
    """A query-only VCA + FCLS interface with a fixed internal random seed.

    The attack code only observes this object's returned decomposition.  Fixing
    the seed makes repeated queries of the same cube deterministic, which is a
    practical requirement for query-efficient evolutionary optimisation.
    """

    def __init__(
        self,
        baseline_root: str | Path,
        n_endmembers: int,
        seed: int = 17,
        max_vca_pixels: int = 0,
        n_vca_runs: int = 4,
        solver: str = "fcls_exact",
        shared_projection: SharedVCAProjection | None = None,
    ) -> None:
        self.n_endmembers = int(n_endmembers)
        self.seed = int(seed)
        self.max_vca_pixels = int(max_vca_pixels)
        self.n_vca_runs = int(n_vca_runs)
        self.solver = solver
        self.shared_projection = shared_projection
        self._estimate_endmembers_vca, self._solve_abundances = _load_baseline_module(Path(baseline_root))

    def _query(self, cube: np.ndarray, diagnostics: bool) -> UnmixingResult:
        cube = np.asarray(cube, dtype=np.float32)
        if cube.ndim != 3:
            raise ValueError(f"Expected H x W x B cube, got {cube.shape}")
        h, w, bands = cube.shape
        x = np.moveaxis(cube, -1, 0).reshape(bands, -1)
        projection = None
        if self.shared_projection is not None and self.max_vca_pixels <= 0:
            projection = self.shared_projection.get(cube)
        endmembers, _ = self._estimate_endmembers_vca(
            x,
            n_endmembers=self.n_endmembers,
            seed=self.seed,
            max_pixels=self.max_vca_pixels,
            n_runs=self.n_vca_runs,
            projection=projection,
        )
        abundance_solver = (
            "nnls_batch" if not diagnostics and self.solver == "nnls" else self.solver
        )
        abundance = self._solve_abundances(x, endmembers, solver=abundance_solver)
        reconstruction = None
        residual = float("nan")
        sre_db = float("nan")
        if diagnostics:
            reconstruction = (endmembers.astype(np.float64) @ abundance.astype(np.float64)).T.reshape(h, w, bands)
            residual = float(
                np.mean(np.linalg.norm(x.T - reconstruction.reshape(-1, bands), axis=1) /
                        (np.linalg.norm(x.T, axis=1) + EPS))
            )
            reconstruction_error = float(np.linalg.norm(x.T - reconstruction.reshape(-1, bands)))
            signal_energy = float(np.linalg.norm(x.T))
            sre_db = float(20.0 * np.log10((signal_energy + EPS) / (reconstruction_error + EPS)))
        return UnmixingResult(
            endmembers=endmembers.astype(np.float32),
            abundances=abundance.reshape(self.n_endmembers, h, w).astype(np.float32),
            reconstruction=(
                reconstruction.astype(np.float32) if reconstruction is not None else None
            ),
            relative_residual=residual,
            reconstruction_sre_db=sre_db,
        )

    def query(self, cube: np.ndarray) -> UnmixingResult:
        return self._query(cube, diagnostics=True)

    def query_compact(self, cube: np.ndarray) -> UnmixingResult:
        """Return only fields consumed inside the optimisation loop."""
        return self._query(cube, diagnostics=False)
