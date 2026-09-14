"""VCA + NNLS/FCLS material unmixing consistency evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import nnls

from .data_io import cube_to_matrix, load_cube_mat, load_reference_cube, save_map_npz

EPS = 1e-8


@dataclass
class ScoreResult:
    metrics: dict
    maps: dict[str, np.ndarray]


def _subsample_columns(X: np.ndarray, max_pixels: int, rng: np.random.Generator) -> np.ndarray:
    if max_pixels <= 0 or X.shape[1] <= max_pixels:
        return X
    idx = rng.choice(X.shape[1], size=max_pixels, replace=False)
    return X[:, idx]


def estimate_endmembers_vca(
    X: np.ndarray,
    n_endmembers: int,
    seed: int = 0,
    max_pixels: int = 50000,
    n_runs: int = 8,
    projection: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate endmembers using a VCA-style pure-pixel search.

    The implementation projects spectra into a low-dimensional affine subspace,
    repeatedly searches extreme points with random directions, and keeps the run
    with the largest simplex volume. It is intentionally lightweight and stable
    for small unmixing benchmarks such as Samson, Jasper and Urban.
    """
    X = np.asarray(X, dtype=np.float64)
    X = np.clip(X, 0.0, 1.0)
    bands, pixels = X.shape
    if n_endmembers < 2 or n_endmembers > bands:
        raise ValueError(f"Invalid n_endmembers={n_endmembers} for bands={bands}")

    rng = np.random.default_rng(seed)
    if projection is None:
        Xs = _subsample_columns(X, max_pixels=max_pixels, rng=rng)
        Xs, Y = prepare_vca_projection(Xs, n_endmembers)
    else:
        if max_pixels > 0 and X.shape[1] > max_pixels:
            raise ValueError("A shared VCA projection is only valid without random subsampling.")
        Xs, Y = projection
        Xs = np.asarray(Xs, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

    best_idx: np.ndarray | None = None
    best_volume = -np.inf
    for _ in range(max(1, n_runs)):
        A = np.zeros((n_endmembers, n_endmembers), dtype=np.float64)
        selected = []
        for i in range(n_endmembers):
            w = rng.normal(size=(n_endmembers, 1))
            if i > 0:
                P = A[:, :i] @ np.linalg.pinv(A[:, :i])
                w = w - P @ w
            norm = np.linalg.norm(w)
            if norm < EPS:
                w = rng.normal(size=(n_endmembers, 1))
                norm = np.linalg.norm(w)
            f = w / (norm + EPS)
            proj = np.abs((f.T @ Y).ravel())
            idx = int(np.argmax(proj))
            A[:, i] = Y[:, idx]
            selected.append(idx)
        try:
            volume = abs(np.linalg.det(A))
        except np.linalg.LinAlgError:
            volume = 0.0
        if volume > best_volume:
            best_volume = volume
            best_idx = np.asarray(selected, dtype=np.int64)

    assert best_idx is not None
    E = Xs[:, best_idx].astype(np.float32)
    # Normalize endmembers to avoid degenerate scale effects.
    E = np.clip(E, 0.0, 1.0)
    return E, best_idx


def prepare_vca_projection(
    X: np.ndarray, n_endmembers: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build the deterministic VCA affine projection once for several seeds.

    Random extreme-point directions remain private to each VCA query.  Sharing
    this seed-independent projection therefore removes duplicate SVD work but
    does not change which search each service performs.
    """
    Xs = np.clip(np.asarray(X, dtype=np.float64), 0.0, 1.0)
    if Xs.ndim != 2:
        raise ValueError("X must have shape bands x pixels")
    if n_endmembers < 2 or n_endmembers > Xs.shape[0]:
        raise ValueError(
            f"Invalid n_endmembers={n_endmembers} for bands={Xs.shape[0]}"
        )
    mean = Xs.mean(axis=1, keepdims=True)
    Xc = Xs - mean
    U, _, _ = np.linalg.svd(Xc, full_matrices=False)
    Ud = U[:, : n_endmembers - 1]
    Z = Ud.T @ Xc
    Y = np.vstack([Z, np.ones((1, Z.shape[1]))])
    return Xs, Y



def project_simplex_columns(Y: np.ndarray) -> np.ndarray:
    """Project each column of Y onto the probability simplex."""
    Y = np.asarray(Y, dtype=np.float64)
    U = np.sort(Y, axis=0)[::-1]
    cssv = np.cumsum(U, axis=0) - 1.0
    ind = np.arange(1, Y.shape[0] + 1, dtype=np.float64)[:, None]
    cond = U - cssv / ind > 0
    rho = np.maximum(np.sum(cond, axis=0) - 1, 0)
    theta = cssv[rho, np.arange(Y.shape[1])] / (rho + 1.0)
    return np.maximum(Y - theta[None, :], 0.0)

def solve_abundances(
    X: np.ndarray,
    E: np.ndarray,
    solver: str = "fcls",
    sum_weight: float = 10.0,
    show_progress: bool = False,
) -> np.ndarray:
    """Estimate abundances with NNLS or a sum-to-one constrained solver.

    ``fcls_exact`` solves the convex FCLS problem exactly for the small
    dictionaries used by this project.  ``fcls_fast`` is retained only for
    backward-compatible regression of earlier exploratory experiments; its
    pseudoinverse-plus-simplex projection is not the FCLS optimum in general.
    """
    X = np.asarray(X, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    k = E.shape[1]
    if solver not in {"nnls", "nnls_batch", "fcls", "fcls_exact", "fcls_fast"}:
        raise ValueError(
            "solver must be 'nnls', 'nnls_batch', 'fcls', 'fcls_exact' or "
            "'fcls_fast'"
        )

    if solver == "fcls_fast":
        # Fast FCLS approximation: unconstrained least squares followed by
        # Euclidean projection onto the abundance simplex. This is much faster
        # than per-pixel NNLS and is adequate for evidence scoring.
        Y = np.linalg.pinv(E) @ X
        return project_simplex_columns(Y).astype(np.float32)

    if solver == "fcls_exact":
        return _solve_fcls_exact(X, E)

    if solver == "nnls_batch":
        return _solve_nnls_batch(X, E)

    if solver == "fcls":
        A = np.vstack([E, sum_weight * np.ones((1, k), dtype=np.float64)])
    else:
        A = E
    abund = np.zeros((k, X.shape[1]), dtype=np.float32)

    iterator: Iterable[int] = range(X.shape[1])
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="solve abundances")
        except Exception:
            pass

    for j in iterator:
        y = X[:, j]
        if solver == "fcls":
            b = np.concatenate([y, np.array([sum_weight], dtype=np.float64)])
        else:
            b = y
        a, _ = nnls(A, b)
        s = float(a.sum())
        if s > EPS:
            a = a / s
        abund[:, j] = a.astype(np.float32)
    return abund


def _solve_fcls_exact(X: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Solve exact FCLS by enumerating the active abundance set.

    A hyperspectral scene in this project has three to six materials.  For
    such tiny dictionaries, enumerating all non-empty supports is both faster
    than thousands of Python-level constrained optimisations and gives the
    exact solution of

        min ||E a - x||_2^2  subject to a >= 0 and 1^T a = 1.

    Each support is solved in a batch through its equality-constrained KKT
    system.  The feasible support with the smallest reconstruction error is
    retained independently for every pixel.
    """

    k, pixels = E.shape[1], X.shape[1]
    if k > 12:
        raise ValueError(
            "fcls_exact active-set enumeration is limited to 12 endmembers; "
            f"received {k}."
        )
    gram = E.T @ E
    rhs = E.T @ X
    signal_norm = np.sum(X * X, axis=0)
    best_error = np.full(pixels, np.inf, dtype=np.float64)
    best = np.zeros((k, pixels), dtype=np.float64)
    tolerance = 1e-9

    for mask in range(1, 1 << k):
        active = np.flatnonzero([(mask >> index) & 1 for index in range(k)])
        size = active.size
        kkt = np.empty((size + 1, size + 1), dtype=np.float64)
        kkt[:size, :size] = gram[np.ix_(active, active)]
        kkt[:size, size] = 1.0
        kkt[size, :size] = 1.0
        kkt[size, size] = 0.0
        target = np.vstack([rhs[active], np.ones((1, pixels), dtype=np.float64)])
        coefficients = np.linalg.lstsq(kkt, target, rcond=None)[0][:size]
        feasible = np.all(coefficients >= -tolerance, axis=0)
        if not np.any(feasible):
            continue
        coefficients = np.maximum(coefficients, 0.0)
        coefficients /= np.maximum(coefficients.sum(axis=0, keepdims=True), EPS)
        active_gram = gram[np.ix_(active, active)]
        quadratic = np.sum(coefficients * (active_gram @ coefficients), axis=0)
        error = signal_norm - 2.0 * np.sum(coefficients * rhs[active], axis=0) + quadratic
        update = feasible & (error < best_error)
        if np.any(update):
            columns = np.flatnonzero(update)
            best[:, columns] = 0.0
            best[np.ix_(active, columns)] = coefficients[:, update]
            best_error[columns] = error[columns]

    if not np.all(np.isfinite(best_error)):
        raise RuntimeError("Exact FCLS failed to find a feasible active set.")
    return best.astype(np.float32)


def _solve_nnls_batch(X: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Solve small-dictionary NNLS for every pixel by active-set enumeration.

    Hyperspectral scenes in this project use only a few endmembers. Enumerating
    their non-empty active sets turns thousands of Python-level ``nnls`` calls
    into a few dense linear-algebra operations. Degenerate unresolved columns
    fall back to SciPy's reference solver.
    """
    k, pixels = E.shape[1], X.shape[1]
    if k > 12:
        # Enumeration is intended for the small material dictionaries used by
        # the experiments; avoid exponential work for an unexpected input.
        return solve_abundances(X, E, solver="nnls")
    gram = E.T @ E
    rhs = E.T @ X
    signal_norm = np.sum(X * X, axis=0)
    best_error = np.full(pixels, np.inf, dtype=np.float64)
    best = np.zeros((k, pixels), dtype=np.float64)
    tolerance = 1e-10
    for mask in range(1, 1 << k):
        active = np.flatnonzero([(mask >> index) & 1 for index in range(k)])
        coefficients = np.linalg.lstsq(
            gram[np.ix_(active, active)], rhs[active], rcond=None
        )[0]
        feasible = np.all(coefficients >= -tolerance, axis=0)
        if not np.any(feasible):
            continue
        coefficients = np.maximum(coefficients, 0.0)
        quadratic = np.sum(coefficients * (gram[np.ix_(active, active)] @ coefficients), axis=0)
        error = signal_norm - 2.0 * np.sum(coefficients * rhs[active], axis=0) + quadratic
        update = feasible & (error < best_error)
        if np.any(update):
            best[:, update] = 0.0
            best[np.ix_(active, np.flatnonzero(update))] = coefficients[:, update]
            best_error[update] = error[update]

    unresolved = ~np.isfinite(best_error)
    for column in np.flatnonzero(unresolved):
        best[:, column], _ = nnls(E, X[:, column])
    sums = best.sum(axis=0, keepdims=True)
    best = np.divide(best, sums, out=np.zeros_like(best), where=sums > EPS)
    return best.astype(np.float32)


def _sam_map(X: np.ndarray, Xhat: np.ndarray) -> np.ndarray:
    dot = np.sum(X * Xhat, axis=0)
    denom = np.linalg.norm(X, axis=0) * np.linalg.norm(Xhat, axis=0) + EPS
    cos = np.clip(dot / denom, -1.0, 1.0)
    return np.arccos(cos).astype(np.float32)


def _abundance_tv(A: np.ndarray, h: int, w: int) -> tuple[float, np.ndarray]:
    k = A.shape[0]
    maps = A.reshape(k, h, w)
    tv = np.zeros((h, w), dtype=np.float32)
    for m in maps:
        gx = np.zeros_like(m, dtype=np.float32)
        gy = np.zeros_like(m, dtype=np.float32)
        gx[:, 1:] = np.abs(m[:, 1:] - m[:, :-1])
        gy[1:, :] = np.abs(m[1:, :] - m[:-1, :])
        tv += gx + gy
    tv /= max(k, 1)
    return float(np.mean(tv)), tv


def _entropy(A: np.ndarray) -> np.ndarray:
    k = A.shape[0]
    P = np.clip(A, EPS, 1.0)
    return (-np.sum(P * np.log(P), axis=0) / np.log(k + EPS)).astype(np.float32)


def _profile(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).ravel()
    q25, q50, q75, q90, q95 = np.percentile(v, [25, 50, 75, 90, 95])
    return {
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "q90": float(q90),
        "q95": float(q95),
        "iqr": float(max(q75 - q25, EPS)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v) + EPS),
    }


def _robust_excess(value: float, profile: dict[str, float], anchor: str = "q75") -> float:
    base = profile.get(anchor, profile["q50"])
    return max(0.0, (float(value) - base) / (profile["iqr"] + EPS))


def fit_material_reference(
    cube: np.ndarray,
    n_endmembers: int,
    seed: int = 0,
    max_vca_pixels: int = 50000,
    solver: str = "fcls",
    sum_weight: float = 10.0,
) -> dict:
    X = cube_to_matrix(cube)
    E, selected = estimate_endmembers_vca(
        X,
        n_endmembers=n_endmembers,
        seed=seed,
        max_pixels=max_vca_pixels,
        n_runs=12,
    )
    result = score_cube_material(cube, E, solver=solver, sum_weight=sum_weight)
    profiles = {
        "rec": _profile(result.maps["rec_map"]),
        "sam": _profile(result.maps["sam_map"]),
        "entropy": _profile(result.maps["entropy_map"]),
        "abundance_tv": _profile(result.maps["abundance_tv_map"]),
        "cube": dict(result.metrics),
    }
    return {
        "endmembers": E,
        "selected_indices": selected,
        "profiles": profiles,
        "n_endmembers": int(n_endmembers),
        "solver": solver,
        "sum_weight": float(sum_weight),
    }


def score_cube_material(cube: np.ndarray, E: np.ndarray, solver: str = "fcls", sum_weight: float = 10.0) -> ScoreResult:
    h, w, _ = cube.shape
    X = cube_to_matrix(cube).astype(np.float64)
    A = solve_abundances(X, E, solver=solver, sum_weight=sum_weight)
    Xhat = np.asarray(E, dtype=np.float64) @ A.astype(np.float64)

    rec = (np.linalg.norm(X - Xhat, axis=0) / (np.linalg.norm(X, axis=0) + EPS)).astype(np.float32)
    sam = _sam_map(X, Xhat)
    ent = _entropy(A)
    tv_mean, tv_map = _abundance_tv(A, h, w)

    rec_map = rec.reshape(h, w)
    sam_map = sam.reshape(h, w)
    entropy_map = ent.reshape(h, w)
    metrics = {
        "rec_mean": float(np.mean(rec)),
        "rec_q90": float(np.percentile(rec, 90)),
        "rec_q95": float(np.percentile(rec, 95)),
        "sam_mean": float(np.mean(sam)),
        "sam_q90": float(np.percentile(sam, 90)),
        "sam_q95": float(np.percentile(sam, 95)),
        "abundance_entropy_mean": float(np.mean(ent)),
        "abundance_entropy_q95": float(np.percentile(ent, 95)),
        "abundance_tv_mean": tv_mean,
    }
    maps = {
        "rec_map": rec_map,
        "sam_map": sam_map,
        "entropy_map": entropy_map,
        "abundance_tv_map": tv_map,
        "abundance": A.reshape(A.shape[0], h, w),
    }
    return ScoreResult(metrics=metrics, maps=maps)


def _mean_excess(value: float, profile: dict[str, float], sigma_scale: float = 1.0) -> float:
    base = profile["mean"] + sigma_scale * profile["std"]
    scale = profile["std"] + EPS
    return max(0.0, (float(value) - base) / scale)


def _low_mean_excess(value: float, profile: dict[str, float], sigma_scale: float = 1.0) -> float:
    base = profile["mean"] - sigma_scale * profile["std"]
    scale = profile["std"] + EPS
    return max(0.0, (base - float(value)) / scale)


def _profile_low_excess(value: float, profile: dict[str, float], anchor: str = "q25") -> float:
    base = profile.get(anchor, profile["q50"])
    return max(0.0, (base - float(value)) / (profile["iqr"] + EPS))


def _cube_abs_excess(value: float, reference: float, rel_tol: float, rel_scale: float, abs_scale: float = 1e-4) -> float:
    reference = float(reference)
    threshold = abs(reference) * rel_tol + abs_scale
    scale = abs(reference) * rel_scale + abs_scale
    return max(0.0, (abs(float(value) - reference) - threshold) / scale)


def _cube_high_excess(value: float, reference: float, rel_tol: float, rel_scale: float, abs_scale: float = 1e-4) -> float:
    reference = float(reference)
    threshold = reference * (1.0 + rel_tol) + abs_scale
    scale = abs(reference) * rel_scale + abs_scale
    return max(0.0, (float(value) - threshold) / scale)


def calibrated_material_score(metrics: dict, profiles: dict) -> tuple[float, float]:
    # Cube-level q95 statistics are compared against the clean q95 profile so
    # that the clean reference cube is not penalized by its own upper tail.
    rec_high = _robust_excess(metrics["rec_q95"], profiles["rec"], anchor="q95")
    sam_high = _robust_excess(metrics["sam_q95"], profiles["sam"], anchor="q95")

    # Several HSI degradations, especially spatial mixing and blur, can make
    # spectra easier to reconstruct while destroying material separability.
    # Low-tail deviations are therefore treated as evidence of material drift
    # rather than ignored as apparently better reconstruction.
    rec_low = _profile_low_excess(metrics["rec_q95"], profiles["rec"], anchor="q50")
    sam_low = _profile_low_excess(metrics["sam_q95"], profiles["sam"], anchor="q50")
    entropy_high = _mean_excess(metrics["abundance_entropy_mean"], profiles["entropy"], sigma_scale=0.5)
    tv_high = _mean_excess(metrics["abundance_tv_mean"], profiles["abundance_tv"], sigma_scale=1.0)
    tv_low = _low_mean_excess(metrics["abundance_tv_mean"], profiles["abundance_tv"], sigma_scale=0.7)

    clean_cube = profiles.get("cube", {})
    if clean_cube:
        rec_cube_shift = _cube_abs_excess(metrics["rec_q95"], clean_cube["rec_q95"], rel_tol=0.08, rel_scale=0.32)
        sam_cube_shift = _cube_abs_excess(metrics["sam_q95"], clean_cube["sam_q95"], rel_tol=0.06, rel_scale=0.28)
        ent_cube_high = _cube_high_excess(
            metrics["abundance_entropy_mean"], clean_cube["abundance_entropy_mean"], rel_tol=0.03, rel_scale=0.18
        )
        tv_cube_shift = _cube_abs_excess(
            metrics["abundance_tv_mean"], clean_cube["abundance_tv_mean"], rel_tol=0.08, rel_scale=0.35
        )
    else:
        rec_cube_shift = sam_cube_shift = ent_cube_high = tv_cube_shift = 0.0

    raw = (
        0.18 * rec_high
        + 0.16 * sam_high
        + 0.10 * rec_low
        + 0.10 * sam_low
        + 0.08 * entropy_high
        + 0.05 * tv_high
        + 0.05 * tv_low
        + 0.12 * rec_cube_shift
        + 0.12 * sam_cube_shift
        + 0.12 * ent_cube_high
        + 0.12 * tv_cube_shift
    )
    risk = float(1.0 - np.exp(-0.38 * raw))
    support = float(np.exp(-0.38 * raw))
    return risk, support


def save_artifact(path: str | Path, artifact: dict, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profiles = artifact["profiles"]
    np.savez_compressed(
        path,
        endmembers=artifact["endmembers"],
        selected_indices=artifact["selected_indices"],
        metadata=np.array([metadata or {}], dtype=object),
        profiles=np.array([profiles], dtype=object),
        n_endmembers=np.array([artifact["n_endmembers"]], dtype=np.int32),
        solver=np.array([artifact["solver"]], dtype=object),
        sum_weight=np.array([artifact["sum_weight"]], dtype=np.float32),
    )


def load_artifact(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "endmembers": data["endmembers"].astype(np.float32),
        "selected_indices": data["selected_indices"],
        "profiles": data["profiles"].item(),
        "n_endmembers": int(data["n_endmembers"][0]),
        "solver": str(data["solver"][0]),
        "sum_weight": float(data["sum_weight"][0]),
        "metadata": data["metadata"].item() if "metadata" in data else {},
    }


def fit_dataset_artifact(
    data_root: str | Path,
    dataset: str,
    output_path: str | Path,
    n_endmembers: int | None = None,
    seed: int = 0,
    max_vca_pixels: int = 50000,
    solver: str = "fcls",
    sum_weight: float = 10.0,
) -> dict:
    cube, meta = load_reference_cube(data_root, dataset)
    if n_endmembers is None:
        n_endmembers = int(meta["default_endmembers"])
    artifact = fit_material_reference(
        cube,
        n_endmembers=n_endmembers,
        seed=seed,
        max_vca_pixels=max_vca_pixels,
        solver=solver,
        sum_weight=sum_weight,
    )
    save_artifact(output_path, artifact, metadata=meta)
    return {**meta, "artifact": str(output_path), "n_endmembers": n_endmembers}


def score_path_with_artifact(
    cube_path: str | Path,
    artifact_path: str | Path,
    save_maps_path: str | Path | None = None,
) -> dict:
    cube, meta = load_cube_mat(cube_path)
    artifact = load_artifact(artifact_path)
    result = score_cube_material(
        cube,
        artifact["endmembers"],
        solver=artifact["solver"],
        sum_weight=artifact["sum_weight"],
    )
    risk, support = calibrated_material_score(result.metrics, artifact["profiles"])
    row = {
        **result.metrics,
        "material_risk": risk,
        "material_support": support,
        "source_path": str(cube_path),
        "height": meta["height"],
        "width": meta["width"],
        "bands": meta["bands"],
    }
    if save_maps_path is not None:
        save_map_npz(save_maps_path, **result.maps)
        row["maps_path"] = str(save_maps_path)
    return row
