"""Strict black-box weak-simplex-tangent region search.

The search consumes only clean and perturbed input/output queries.  It never
receives reference abundances, reference endmembers, model identities,
parameters, gradients, or intermediate features.  Clean returned endmember
sets define anonymous simplex-tangent candidates; returned abundance maps
provide the sole search signal.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
from scipy.ndimage import uniform_filter

from .cross_unmixer import QueryVictim, match_to_anchors
from .optimizer_registry import ALL_OPTIMIZERS, minimize_with_optimizer
from .response_guided_attack import _canonical_clean_outputs
from .query_accounting import QueryLedger


EPS = 1e-8
COVERAGE_THRESHOLDS = (0.02, 0.05, 0.10, 0.15, 0.20)


@dataclass
class WeakTangentConfig:
    attack_mode: str = "auto"
    target_source_material: int = 0
    target_destination_material: int = 1
    target_region: str = "whole"
    target_region_fraction: float = 0.20
    n_directions: int = 2
    candidate_directions_per_response: int = 2
    direction_strategy: str = "weak"
    anchor_pixels_per_material: int = 5
    anchor_dilation: int = 0
    linf_cap: float = 0.04
    support_fraction_cap: float = 1.0
    square_steps: int = 128
    refinement_steps: int = 0
    refinement_probes: int = 4
    initial_block_side: int = 64
    minimum_block_side: int = 2
    guidance_probability: float = 0.0
    guidance_temperature: float = 1.0
    coefficient_step: float = 0.35
    seed: int = 7
    baseline_residual_cap: float = 0.08
    minimum_active_services: int = 1
    pareto_archive_size: int = 40
    selection_effect_tolerance: float = 0.0
    directional_objective: bool = False
    directional_quantile: float = 0.25
    directional_gate_tolerance: float = 1e-6
    directional_positive_fraction_min: float = 0.60
    coverage_curve_objective: bool = False
    service_effect_quantile: float = 0.25
    minimum_successful_service_fraction: float = 0.70
    high_confidence_guidance: bool = False
    confidence_floor: float = 0.10
    proxy_reconstruction_sre_drop_cap_db: float = 1e6
    mean_sam_cap: float = 0.0
    endpoint_orthogonal_drift_cap: float = 1e6
    endpoint_total_drift_cap: float = 1e6
    residual_variability_branch: bool = False
    residual_variability_quantile: float = 0.80
    residual_coupling_cap: float = 0.35
    free_spectral_atoms: int = 0
    abundance_shift_quantile: float = 0.0
    maximum_candidate_queries: int = 0
    local_linf_projection: bool = False
    adaptive_operator_search: bool = False
    operator_candidates_per_step: int = 1
    moead_generations: int = 0
    moead_population_size: int = 20
    moead_neighbourhood_size: int = 5
    moead_neighbour_mating_probability: float = 0.90
    moead_maximum_replacements: int = 2
    moead_block_crossover_probability: float = 0.50
    moead_safe_selection: bool = True
    moead_structured_initialization: bool = True
    moead_structured_operators: bool = True
    moead_adaptive_operator_selection: bool = True
    refinement_optimizer: str = "proposed_moead"
    directional_hard_gate: bool = False


@dataclass
class WeakTangentMeasures:
    robust_abundance_shift: float
    mean_abundance_shift: float
    relative_energy: float
    linf: float
    support_fraction: float
    mean_endmember_drift: float
    per_response_shift: dict[str, float]
    robust_directional_transfer: float = 0.0
    mean_directional_transfer: float = 0.0
    fixed_directional_transfer: float = 0.0
    robust_directional_quantile: float = 0.0
    fixed_directional_quantile: float = 0.0
    minimum_positive_fraction: float = 0.0
    effective_coverage_001: float = 0.0
    effective_coverage_002: float = 0.0
    effective_coverage_005: float = 0.0
    effective_coverage_010: float = 0.0
    effective_coverage_015: float = 0.0
    effective_coverage_020: float = 0.0
    robust_coverage_auc: float = 0.0
    fixed_coverage_auc: float = 0.0
    successful_service_fraction: float = 1.0
    maximum_proxy_reconstruction_sre_drop_db: float = 0.0
    mean_proxy_reconstruction_sre_drop_db: float = 0.0
    directional_feasible: bool = True
    mean_sam: float = 0.0
    endpoint_parallel_progress: float = 0.0
    maximum_endpoint_orthogonal_ratio: float = 0.0
    maximum_endpoint_total_drift_ratio: float = 0.0
    per_response_directional_transfer: dict[str, float] | None = None
    per_response_directional_quantile: dict[str, float] | None = None
    per_response_positive_fraction: dict[str, float] | None = None
    per_response_coverage_auc: dict[str, float] | None = None
    per_response_reconstruction_sre_drop_db: dict[str, float] | None = None


@dataclass
class WeakTangentRecord:
    coefficients: np.ndarray
    objectives: np.ndarray
    measures: WeakTangentMeasures
    response_map: np.ndarray


@dataclass
class WeakTangentResult:
    attacked_cube: np.ndarray
    canonical_endmembers: np.ndarray
    consensus_abundances: np.ndarray
    directions: np.ndarray
    direction_footprints: np.ndarray
    direction_labels: list[str]
    residual_coupling: float
    clean_consensus_relative_residual: float
    anchor_mask: np.ndarray
    confidence_map: np.ndarray
    target_map: np.ndarray
    selected: WeakTangentRecord
    pareto: list[WeakTangentRecord]
    active_services: list[str]
    baseline_residuals: dict[str, float]
    candidate_queries: int
    service_queries: int
    cache_hits: int
    operator_attempts: dict[str, int]
    operator_successes: dict[str, int]
    moead_operator_attempts: dict[str, int]
    moead_operator_successes: dict[str, int]
    moead_effect_first: WeakTangentRecord | None
    moead_protected_baseline: WeakTangentRecord | None
    optimizer_evaluations: int
    optimizer_generations: int
    optimizer_implementation: str
    query_accounting: dict = field(default_factory=dict)


def simplex_tangent_basis(materials: int) -> np.ndarray:
    """Return an orthonormal basis of vectors whose entries sum to zero."""
    materials = int(materials)
    if materials < 2:
        raise ValueError("At least two materials are required.")
    raw = np.zeros((materials, materials - 1), dtype=np.float64)
    raw[:-1] = np.eye(materials - 1, dtype=np.float64)
    raw[-1] = -1.0
    basis, _ = np.linalg.qr(raw, mode="reduced")
    return basis


def robust_weak_directions(
    aligned_endmembers: dict[str, np.ndarray],
    count: int,
    candidates_per_response: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build anonymous cross-response weak tangent directions.

    Low-eigenvalue candidates are generated from every clean returned
    endmember set and their trace-normalised mean Gram matrix.  They are then
    ranked by their worst L-infinity spectral footprint across all responses.
    """
    if not aligned_endmembers:
        raise ValueError("At least one aligned endmember set is required.")
    matrices = [np.asarray(value, dtype=np.float64) for value in aligned_endmembers.values()]
    bands, materials = matrices[0].shape
    if any(value.shape != (bands, materials) for value in matrices):
        raise ValueError("Aligned endmember sets must share one shape.")
    if int(count) < 0:
        raise ValueError("Weak direction count cannot be negative.")
    if int(count) == 0:
        return (
            np.zeros((materials, 0), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    tangent = simplex_tangent_basis(materials)
    grams = []
    for value in matrices:
        gram = tangent.T @ value.T @ value @ tangent
        grams.append(gram / max(float(np.trace(gram)), EPS))
    source_grams = [*grams, np.mean(np.stack(grams, axis=0), axis=0)]
    requested = min(max(1, int(candidates_per_response)), materials - 1)
    candidates: list[np.ndarray] = []
    for gram in source_grams:
        _, vectors = np.linalg.eigh(gram)
        for column in range(requested):
            direction = tangent @ vectors[:, column]
            direction /= max(float(np.linalg.norm(direction)), EPS)
            pivot = int(np.argmax(np.abs(direction)))
            if direction[pivot] < 0.0:
                direction = -direction
            if any(abs(float(np.dot(direction, old))) > 1.0 - 1e-7 for old in candidates):
                continue
            candidates.append(direction)
    if not candidates:
        raise RuntimeError("Could not construct a weak simplex direction.")
    footprints = np.asarray(
        [max(float(np.max(np.abs(value @ direction))) for value in matrices) for direction in candidates],
        dtype=np.float64,
    )
    order = np.argsort(footprints)
    keep = order[: min(max(1, int(count)), len(order))]
    return (
        np.column_stack([candidates[index] for index in keep]).astype(np.float32),
        footprints[keep].astype(np.float32),
    )


def random_tangent_directions(
    aligned_endmembers: dict[str, np.ndarray],
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic, unit-norm random simplex-tangent control.

    This is an ablation control for :func:`robust_weak_directions`.  It keeps
    the same number of abundance-conserving search directions and the same
    downstream parameterisation, constraints and optimiser, but does not use
    low-footprint spectral geometry to choose the directions.
    """
    if not aligned_endmembers:
        raise ValueError("At least one aligned endmember set is required.")
    matrices = [
        np.asarray(value, dtype=np.float64)
        for value in aligned_endmembers.values()
    ]
    bands, materials = matrices[0].shape
    if any(value.shape != (bands, materials) for value in matrices):
        raise ValueError("Aligned endmember sets must share one shape.")
    count = int(count)
    if count < 0:
        raise ValueError("Direction count cannot be negative.")
    if count == 0:
        return (
            np.zeros((materials, 0), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    tangent = simplex_tangent_basis(materials)
    rng = np.random.default_rng(int(seed))
    directions: list[np.ndarray] = []
    while len(directions) < count:
        direction = tangent @ rng.normal(size=tangent.shape[1])
        direction /= max(float(np.linalg.norm(direction)), EPS)
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0.0:
            direction = -direction
        # Avoid effectively duplicating a previous control direction.  The
        # loop remains valid when count exceeds the tangent dimension because
        # non-orthogonal, non-duplicate directions are permitted.
        if any(
            abs(float(np.dot(direction, old))) > 1.0 - 1e-7
            for old in directions
        ):
            continue
        directions.append(direction)
    stacked = np.column_stack(directions)
    footprints = np.asarray(
        [
            max(float(np.max(np.abs(value @ stacked[:, index]))) for value in matrices)
            for index in range(stacked.shape[1])
        ],
        dtype=np.float32,
    )
    return stacked.astype(np.float32), footprints


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0:
        return np.asarray(mask, dtype=bool).copy()
    side = 2 * radius + 1
    return uniform_filter(mask.astype(np.float32), size=side, mode="constant") > 0.0


def anchor_mask_from_abundances(
    abundances: np.ndarray,
    pixels_per_material: int,
    dilation: int = 0,
) -> np.ndarray:
    """Freeze the highest-abundance clean-output pixels for every material."""
    abundances = np.asarray(abundances, dtype=np.float64)
    if abundances.ndim != 3:
        raise ValueError("abundances must have shape K x H x W.")
    materials, height, width = abundances.shape
    count = min(max(0, int(pixels_per_material)), height * width)
    mask = np.zeros(height * width, dtype=bool)
    if count:
        for material in range(materials):
            values = abundances[material].reshape(-1)
            selected = np.argpartition(values, -count)[-count:]
            mask[selected] = True
    return _dilate_mask(mask.reshape(height, width), dilation)


def fixed_dictionary_abundances(cube: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
    """Return a data-only fixed-dictionary abundance diagnostic.

    This branch is deliberately independent of any victim implementation.  It
    uses the clean anonymous consensus dictionary, an unconstrained least-
    squares solve, and a simplex projection.  It is a directional sanity gate,
    not a replacement for the formal held-out FCLS/SUnSAL evaluation.
    """
    cube = np.asarray(cube, dtype=np.float64)
    endmembers = np.asarray(endmembers, dtype=np.float64)
    height, width, bands = cube.shape
    if endmembers.shape[0] != bands:
        raise ValueError("cube and endmembers have incompatible band counts.")
    spectra = cube.reshape(-1, bands).T
    abundance = np.linalg.pinv(endmembers) @ spectra
    abundance = np.maximum(abundance, 0.0)
    abundance /= np.maximum(np.sum(abundance, axis=0, keepdims=True), EPS)
    return abundance.reshape(endmembers.shape[1], height, width).astype(np.float32)


def directional_transfer_score(
    planned_delta: np.ndarray,
    observed_delta: np.ndarray,
    quantile: float = 0.25,
) -> tuple[float, np.ndarray]:
    """Measure signed mass transfer along a candidate's planned simplex move.

    For each modified pixel, positive planned material components must increase
    and negative components must decrease.  The smaller of those two changes
    is the paired transfer.  A lower quantile prevents a few unstable pixels
    from dominating the black-box score.
    """
    planned = np.asarray(planned_delta, dtype=np.float64)
    observed = np.asarray(observed_delta, dtype=np.float64)
    if planned.shape != observed.shape or planned.ndim != 3:
        raise ValueError("planned_delta and observed_delta must share K x H x W shape.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1].")
    positive = np.maximum(planned, 0.0)
    negative = np.maximum(-planned, 0.0)
    positive_mass = np.sum(positive, axis=0)
    negative_mass = np.sum(negative, axis=0)
    active = (positive_mass > EPS) & (negative_mass > EPS)
    transfer_map = np.zeros(planned.shape[1:], dtype=np.float64)
    if not np.any(active):
        return 0.0, transfer_map.astype(np.float32)
    positive_weights = positive / np.maximum(positive_mass[None], EPS)
    negative_weights = negative / np.maximum(negative_mass[None], EPS)
    target_gain = np.sum(positive_weights * observed, axis=0)
    source_suppression = np.sum(negative_weights * (-observed), axis=0)
    transfer_map[active] = np.minimum(
        target_gain[active], source_suppression[active]
    )
    return (
        float(np.quantile(transfer_map[active], quantile)),
        transfer_map.astype(np.float32),
    )


def directional_transfer_statistics(
    planned_delta: np.ndarray,
    observed_delta: np.ndarray,
    quantile: float = 0.25,
) -> tuple[dict[str, float], np.ndarray]:
    """Return gate diagnostics and a coverage-aware whole-image effect.

    The signed lower quantile and positive fraction are feasibility gates.  The
    optimisation effect is the RMS of the positive paired transfer over *all*
    image pixels, so unmodified or unsuccessful pixels contribute zero instead
    of disappearing from the objective.
    """
    gate_quantile, transfer_map = directional_transfer_score(
        planned_delta, observed_delta, quantile
    )
    planned = np.asarray(planned_delta, dtype=np.float64)
    positive_mass = np.sum(np.maximum(planned, 0.0), axis=0)
    negative_mass = np.sum(np.maximum(-planned, 0.0), axis=0)
    active = (positive_mass > EPS) & (negative_mass > EPS)
    positive = np.maximum(np.asarray(transfer_map, dtype=np.float64), 0.0)
    if np.any(active):
        positive_fraction = float(np.mean(transfer_map[active] > 0.0))
    else:
        positive_fraction = 1.0
    coverages = {
        f"coverage_{int(round(threshold * 100)):03d}": float(
            np.mean(positive > threshold)
        )
        for threshold in COVERAGE_THRESHOLDS
    }
    statistics = {
        "gate_quantile": float(gate_quantile),
        "global_rms": float(np.sqrt(np.mean(positive**2))),
        "positive_fraction": positive_fraction,
        "coverage_001": float(np.mean(positive > 0.01)),
        **coverages,
        # Discrete area summary over the five explicitly reported thresholds.
        # Keeping the optimisation and paper diagnostics identical avoids a
        # hidden surrogate whose improvement may not transfer to evaluation.
        "coverage_auc": float(np.mean(list(coverages.values()))),
    }
    return statistics, transfer_map


def query_consensus_confidence(
    cube: np.ndarray,
    aligned_abundances: dict[str, np.ndarray],
    consensus: np.ndarray,
) -> np.ndarray:
    """Data-only confidence in locally coherent anonymous material regions.

    Confidence combines three observable properties: agreement between clean
    query responses, consensus dominance margin, and local spectral
    homogeneity.  It does not use reference endmembers, abundance truth, scene
    labels, or victim internals, so the same rule applies to synthetic and real
    images.
    """
    if not aligned_abundances:
        raise ValueError("At least one aligned abundance response is required.")
    cube64 = np.asarray(cube, dtype=np.float64)
    consensus64 = np.asarray(consensus, dtype=np.float64)
    if consensus64.ndim != 3 or cube64.shape[:2] != consensus64.shape[1:]:
        raise ValueError("cube and consensus have incompatible spatial shapes.")

    dominant = np.stack(
        [np.argmax(np.asarray(value), axis=0) for value in aligned_abundances.values()],
        axis=0,
    )
    agreement = np.zeros(consensus64.shape[1:], dtype=np.float64)
    for material in range(consensus64.shape[0]):
        agreement = np.maximum(agreement, np.mean(dominant == material, axis=0))

    ordered = np.sort(consensus64, axis=0)
    margin = ordered[-1] - ordered[-2]
    margin_scale = max(float(np.quantile(margin, 0.90)), EPS)
    margin = np.clip(margin / margin_scale, 0.0, 1.0)

    row_gradient = np.zeros(cube64.shape[:2], dtype=np.float64)
    column_gradient = np.zeros(cube64.shape[:2], dtype=np.float64)
    row_gradient[1:] = np.linalg.norm(cube64[1:] - cube64[:-1], axis=-1)
    column_gradient[:, 1:] = np.linalg.norm(cube64[:, 1:] - cube64[:, :-1], axis=-1)
    local_gradient = uniform_filter(
        row_gradient + column_gradient, size=3, mode="nearest"
    )
    gradient_scale = max(float(np.quantile(local_gradient, 0.90)), EPS)
    homogeneity = 1.0 - np.clip(local_gradient / gradient_scale, 0.0, 1.0)

    confidence = agreement * np.sqrt(np.maximum(margin * homogeneity, 0.0))
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def reconstruction_sre_db(cube: np.ndarray, reconstruction: np.ndarray) -> float:
    """Whole-cube reconstruction SRE used by the query-response hard gate."""
    reference = np.asarray(cube, dtype=np.float64)
    estimate = np.asarray(reconstruction, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError("cube and reconstruction must share one shape.")
    error = reference - estimate
    return float(
        20.0
        * np.log10(
            (float(np.linalg.norm(reference)) + EPS)
            / (float(np.linalg.norm(error)) + EPS)
        )
    )


def mean_spectral_angle(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean per-pixel spectral angle in radians over the complete image."""
    first = np.asarray(reference, dtype=np.float64).reshape(-1, reference.shape[-1])
    second = np.asarray(candidate, dtype=np.float64).reshape(-1, candidate.shape[-1])
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine = np.sum(first * second, axis=1) / np.maximum(denominator, EPS)
    return float(np.mean(np.arccos(np.clip(cosine, -1.0, 1.0))))


def data_spectral_atoms(cube: np.ndarray, count: int) -> np.ndarray:
    """Return data-only PCA atoms for unconstrained spectral block mutations.

    The columns are normalized to unit peak amplitude so the existing
    coefficient and physical-projection code can treat them exactly like the
    spectral atoms used by Square-HU.  No reference endmember or abundance is
    used.
    """
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError("cube must have shape H x W x B.")
    count = int(count)
    if count < 0:
        raise ValueError("count cannot be negative.")
    if count == 0:
        return np.zeros((cube.shape[-1], 0), dtype=np.float32)
    spectra = cube.reshape(-1, cube.shape[-1])
    centered = spectra - np.mean(spectra, axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    atoms = right[: min(count, right.shape[0])].T
    atoms /= np.maximum(np.max(np.abs(atoms), axis=0, keepdims=True), EPS)
    return atoms.astype(np.float32)


def targeted_abundance_map(
    consensus: np.ndarray,
    source_material: int,
    destination_material: int,
    region: str = "whole",
    region_fraction: float = 0.20,
) -> np.ndarray:
    """Construct a data-only source-to-destination abundance target map."""
    consensus = np.asarray(consensus, dtype=np.float64)
    if consensus.ndim != 3:
        raise ValueError("consensus must have shape K x H x W.")
    materials = consensus.shape[0]
    source = int(source_material)
    destination = int(destination_material)
    if not 0 <= source < materials or not 0 <= destination < materials:
        raise ValueError("Target material indices are outside the anonymous material set.")
    if source == destination:
        raise ValueError("Target source and destination materials must differ.")
    if region == "whole":
        mask = np.ones(consensus.shape[1:], dtype=bool)
    elif region == "source_dominant":
        mask = np.argmax(consensus, axis=0) == source
    elif region == "source_top_fraction":
        if not 0.0 < region_fraction <= 1.0:
            raise ValueError("target region fraction must be in (0, 1].")
        source_values = consensus[source].reshape(-1)
        count = max(1, int(np.ceil(float(region_fraction) * source_values.size)))
        selected = np.argpartition(source_values, source_values.size - count)[-count:]
        mask = np.zeros(source_values.size, dtype=bool)
        mask[selected] = True
        mask = mask.reshape(consensus.shape[1:])
    else:
        raise ValueError(
            "target region must be 'whole', 'source_dominant', or "
            "'source_top_fraction'."
        )
    if not np.any(mask):
        raise ValueError("The selected target region contains no pixels.")
    target = np.zeros_like(consensus, dtype=np.float32)
    target[source, mask] = -1.0
    target[destination, mask] = 1.0
    return target


def endpoint_total_drift_ratio(
    clean_endmembers: np.ndarray,
    attacked_endmembers: np.ndarray,
) -> float:
    """Maximum aligned endpoint displacement, normalised per clean endpoint."""
    clean = np.asarray(clean_endmembers, dtype=np.float64)
    attacked = np.asarray(attacked_endmembers, dtype=np.float64)
    if clean.shape != attacked.shape:
        raise ValueError("Endpoint arrays must share one shape.")
    displacement = np.linalg.norm(attacked - clean, axis=0)
    scale = np.maximum(np.linalg.norm(clean, axis=0), EPS)
    return float(np.max(displacement / scale))


def build_material_variability_genes(
    cube: np.ndarray,
    endmembers: np.ndarray,
    abundances: np.ndarray,
    directions: np.ndarray,
    quantile: float = 0.80,
    coupling_cap: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, list[str], float, float]:
    """Expand simplex genes with data-derived material-variability coupling.

    Only the submitted cube and anonymous clean-query decomposition are used.
    Every original simplex direction is retained.  Two additional genes couple
    it to the leading residual variation observed around its anonymous source
    and target material.  Coupling strength is the clean consensus residual
    ratio, capped by ``coupling_cap``; it therefore fades naturally on nearly
    linear scenes without any synthetic/real scene switch.
    """
    cube64 = np.asarray(cube, dtype=np.float64)
    endmembers64 = np.asarray(endmembers, dtype=np.float64)
    abundance64 = np.asarray(abundances, dtype=np.float64)
    directions64 = np.asarray(directions, dtype=np.float64)
    if not 0.0 <= quantile < 1.0:
        raise ValueError("quantile must lie in [0, 1).")
    if coupling_cap < 0.0:
        raise ValueError("coupling_cap cannot be negative.")
    bands = cube64.shape[-1]
    spectra = cube64.reshape(-1, bands).T
    reconstruction = endmembers64 @ abundance64.reshape(abundance64.shape[0], -1)
    residual = spectra - reconstruction
    relative_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(spectra), EPS))
    coupling = min(float(coupling_cap), relative_residual)

    material_atoms: list[np.ndarray] = []
    for material in range(abundance64.shape[0]):
        weights = abundance64[material].reshape(-1)
        threshold = float(np.quantile(weights, quantile))
        selected = weights >= threshold
        local = residual[:, selected].T
        local_weights = weights[selected]
        if local.shape[0] < 2 or float(np.linalg.norm(local)) <= EPS:
            vector = np.zeros(bands, dtype=np.float64)
        else:
            center = np.average(local, axis=0, weights=np.maximum(local_weights, EPS))
            weighted = (local - center[None]) * np.sqrt(
                np.maximum(local_weights, EPS)
            )[:, None]
            _, _, right = np.linalg.svd(weighted, full_matrices=False)
            vector = right[0]
            pivot = int(np.argmax(np.abs(vector)))
            if vector[pivot] < 0.0:
                vector = -vector
        material_atoms.append(vector)

    gene_directions: list[np.ndarray] = []
    residual_atoms: list[np.ndarray] = []
    labels: list[str] = []
    for index in range(directions64.shape[1]):
        direction = directions64[:, index]
        base = endmembers64 @ direction
        base_norm = max(float(np.linalg.norm(base)), EPS)
        source = int(np.argmin(direction))
        target = int(np.argmax(direction))
        variants = (
            ("mix", np.zeros(bands, dtype=np.float64)),
            (f"var_source_{source}", material_atoms[source]),
            (f"var_target_{target}", material_atoms[target]),
        )
        for label, atom in variants:
            atom_norm = float(np.linalg.norm(atom))
            scaled = (
                coupling * base_norm * atom / atom_norm
                if atom_norm > EPS and label != "mix"
                else np.zeros(bands, dtype=np.float64)
            )
            gene_directions.append(direction)
            residual_atoms.append(scaled)
            labels.append(f"d{index}_{label}")
    return (
        np.column_stack(gene_directions).astype(np.float32),
        np.column_stack(residual_atoms).astype(np.float32),
        labels,
        coupling,
        relative_residual,
    )


def endpoint_directional_decomposition(
    clean_endmembers: np.ndarray,
    attacked_endmembers: np.ndarray,
    planned_delta: np.ndarray,
) -> tuple[float, float]:
    """Split endpoint motion into planned parallel and unplanned orthogonal parts.

    The dominant negative and positive entries of the spatially aggregated
    planned abundance move define an anonymous source/target pair.  Only the
    source endpoint motion towards the target is considered planned.  Returned
    values are normalised by the clean source endpoint norm.
    """
    clean = np.asarray(clean_endmembers, dtype=np.float64)
    attacked = np.asarray(attacked_endmembers, dtype=np.float64)
    planned = np.asarray(planned_delta, dtype=np.float64)
    if clean.shape != attacked.shape or planned.shape[0] != clean.shape[1]:
        raise ValueError("Endpoint and planned abundance shapes are incompatible.")
    aggregate = np.sum(planned.reshape(planned.shape[0], -1), axis=1)
    source = int(np.argmin(aggregate))
    target = int(np.argmax(aggregate))
    if source == target or aggregate[target] - aggregate[source] <= EPS:
        return 0.0, 0.0
    direction = clean[:, target] - clean[:, source]
    direction /= max(float(np.linalg.norm(direction)), EPS)
    displacement = attacked[:, source] - clean[:, source]
    scale = max(float(np.linalg.norm(clean[:, source])), EPS)
    parallel = float(np.dot(displacement, direction) / scale)
    orthogonal = displacement - float(np.dot(displacement, direction)) * direction
    return parallel, float(np.linalg.norm(orthogonal) / scale)


class WeakTangentParameterization:
    """Map low-dimensional coefficient fields to simplex-feasible HSI cubes."""

    def __init__(
        self,
        cube: np.ndarray,
        endmembers: np.ndarray,
        abundances: np.ndarray,
        directions: np.ndarray,
        anchor_mask: np.ndarray,
        linf_cap: float,
        support_fraction_cap: float,
        mean_sam_cap: float = 0.0,
        residual_atoms: np.ndarray | None = None,
        local_linf_projection: bool = False,
    ) -> None:
        self.cube = np.asarray(cube, dtype=np.float32)
        self.endmembers = np.asarray(endmembers, dtype=np.float64)
        self.abundances = np.asarray(abundances, dtype=np.float64)
        self.directions = np.asarray(directions, dtype=np.float64)
        self.anchor_mask = np.asarray(anchor_mask, dtype=bool)
        self.linf_cap = float(linf_cap)
        self.support_fraction_cap = float(support_fraction_cap)
        self.mean_sam_cap = float(mean_sam_cap)
        self.local_linf_projection = bool(local_linf_projection)
        self.residual_atoms = (
            np.zeros((self.cube.shape[-1], self.directions.shape[1]), dtype=np.float64)
            if residual_atoms is None
            else np.asarray(residual_atoms, dtype=np.float64)
        )
        self.height, self.width, self.bands = self.cube.shape
        if self.abundances.shape[1:] != (self.height, self.width):
            raise ValueError("Abundances and cube have incompatible spatial shapes.")
        if self.directions.shape[0] != self.abundances.shape[0]:
            raise ValueError("Directions and abundances have incompatible material counts.")
        if self.residual_atoms.shape != (self.bands, self.directions.shape[1]):
            raise ValueError("Residual atoms must have shape B x D.")
        if self.anchor_mask.shape != (self.height, self.width):
            raise ValueError("anchor_mask has an incompatible shape.")
        if not 0.0 < self.linf_cap:
            raise ValueError("linf_cap must be positive.")
        if not 0.0 < self.support_fraction_cap <= 1.0:
            raise ValueError("support_fraction_cap must be in (0, 1].")
        if self.mean_sam_cap < 0.0:
            raise ValueError("mean_sam_cap cannot be negative.")
        self.maximum_support = max(
            1,
            int(np.floor(self.support_fraction_cap * self.height * self.width)),
        )
        # These arrays and limits are invariant across every candidate.  Keep
        # the exact original dtypes and layout so this only removes repeated
        # allocation; it does not change floating-point operation order.
        self._baseline_flat = self.abundances.reshape(self.abundances.shape[0], -1)
        self._spectra = self.cube.reshape(-1, self.bands).T.astype(np.float64)
        self._lower = float(np.min(self.cube))
        self._upper = float(np.max(self.cube))

    def empty_coefficients(self) -> np.ndarray:
        return np.zeros(
            (self.directions.shape[1], self.height, self.width), dtype=np.float32
        )

    def limit_support(
        self,
        coefficients: np.ndarray,
        preferred_mask: np.ndarray | None = None,
        response_map: np.ndarray | None = None,
    ) -> np.ndarray:
        coefficients = np.asarray(coefficients, dtype=np.float32).copy()
        coefficients[:, self.anchor_mask] = 0.0
        norms = np.linalg.norm(coefficients, axis=0)
        active = np.flatnonzero(norms.reshape(-1) > 1e-8)
        if len(active) <= self.maximum_support:
            return coefficients
        preferred = (
            np.asarray(preferred_mask, dtype=bool).reshape(-1)
            if preferred_mask is not None
            else np.zeros(self.height * self.width, dtype=bool)
        )
        response = (
            np.asarray(response_map, dtype=np.float64).reshape(-1)
            if response_map is not None
            else np.zeros(self.height * self.width, dtype=np.float64)
        )
        score = norms.reshape(-1)[active].astype(np.float64)
        score += preferred[active].astype(np.float64) * 1e3
        score += response[active] / max(float(np.max(response)), EPS)
        keep = active[np.argpartition(score, -self.maximum_support)[-self.maximum_support:]]
        retained = np.zeros(self.height * self.width, dtype=bool)
        retained[keep] = True
        coefficients[:, ~retained.reshape(self.height, self.width)] = 0.0
        return coefficients

    def apply(
        self,
        coefficients: np.ndarray,
        *,
        support_is_limited: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coefficients = (
            np.asarray(coefficients, dtype=np.float32)
            if support_is_limited
            else self.limit_support(coefficients)
        )
        flat = coefficients.reshape(coefficients.shape[0], -1).astype(np.float64)
        delta_abundance = self.directions @ flat
        baseline = self._baseline_flat

        scale = np.ones(delta_abundance.shape[1], dtype=np.float64)
        negative = delta_abundance < 0.0
        ratios = np.where(
            negative,
            baseline / np.maximum(-delta_abundance, EPS),
            np.inf,
        )
        scale = np.minimum(scale, np.min(ratios, axis=0))
        scale = np.clip(scale, 0.0, 1.0)
        flat *= scale[None, :]
        delta_abundance = self.directions @ flat
        delta_spectra = self.endmembers @ delta_abundance + self.residual_atoms @ flat
        spectra = self._spectra
        positive = delta_spectra > 0.0
        negative_spectra = delta_spectra < 0.0
        input_ratios = np.where(
            positive,
            (self._upper - spectra) / np.maximum(delta_spectra, EPS),
            np.where(
                negative_spectra,
                (spectra - self._lower) / np.maximum(-delta_spectra, EPS),
                np.inf,
            ),
        )
        input_scale = np.clip(np.min(input_ratios, axis=0), 0.0, 1.0)
        delta_abundance *= input_scale[None, :]
        delta_spectra *= input_scale[None, :]

        if self.local_linf_projection:
            pixel_peak = np.max(np.abs(delta_spectra), axis=0)
            local_scale = np.minimum(
                1.0,
                self.linf_cap / np.maximum(pixel_peak, EPS),
            )
            delta_abundance *= local_scale[None, :]
            delta_spectra *= local_scale[None, :]
        else:
            peak = float(np.max(np.abs(delta_spectra)))
            if peak > self.linf_cap:
                global_scale = self.linf_cap / peak
                delta_abundance *= global_scale
                delta_spectra *= global_scale
        attacked = (spectra + delta_spectra).T.reshape(self.cube.shape)
        if self.mean_sam_cap > 0.0 and mean_spectral_angle(self.cube, attacked) > self.mean_sam_cap:
            low, high = 0.0, 1.0
            for _ in range(18):
                middle = 0.5 * (low + high)
                candidate = (spectra + middle * delta_spectra).T.reshape(self.cube.shape)
                if mean_spectral_angle(self.cube, candidate) <= self.mean_sam_cap:
                    low = middle
                else:
                    high = middle
            delta_abundance *= low
            delta_spectra *= low
            attacked = (spectra + delta_spectra).T.reshape(self.cube.shape)
        return (
            attacked.astype(np.float32),
            delta_abundance.reshape(self.abundances.shape).astype(np.float32),
            coefficients,
        )


def _dominates(first: WeakTangentRecord, second: WeakTangentRecord) -> bool:
    return bool(
        np.all(first.objectives <= second.objectives + 1e-12)
        and np.any(first.objectives < second.objectives - 1e-12)
    )


def _archive_insert(
    archive: list[WeakTangentRecord],
    candidate: WeakTangentRecord,
    maximum_size: int,
) -> list[WeakTangentRecord]:
    if any(_dominates(record, candidate) for record in archive):
        return archive
    kept = [record for record in archive if not _dominates(candidate, record)]
    kept.append(candidate)
    if len(kept) <= maximum_size:
        return kept
    kept.sort(
        key=lambda record: (
            record.objectives[0],
            record.measures.relative_energy,
        )
    )
    effect_indices = np.linspace(0, len(kept) - 1, maximum_size).round().astype(int)
    return [kept[index] for index in np.unique(effect_indices)]


class WeakTangentParetoSearch:
    """Pareto archive with Square proposals and optional response refinement."""

    def __init__(self, services: dict[str, QueryVictim], config: WeakTangentConfig) -> None:
        if not services:
            raise ValueError("services cannot be empty.")
        if config.attack_mode not in {"auto", "untargeted", "targeted"}:
            raise ValueError("attack_mode must be auto, untargeted, or targeted.")
        if config.direction_strategy not in {"weak", "random_tangent"}:
            raise ValueError(
                "direction_strategy must be weak or random_tangent."
            )
        if config.target_region not in {
            "whole", "source_dominant", "source_top_fraction"
        }:
            raise ValueError(
                "target_region must be whole, source_dominant, or "
                "source_top_fraction."
            )
        if not 0.0 < config.target_region_fraction <= 1.0:
            raise ValueError("target_region_fraction must be in (0, 1].")
        if config.square_steps < 0 or config.refinement_steps < 0:
            raise ValueError("Search step counts cannot be negative.")
        if config.refinement_probes < 2 or config.refinement_probes % 2:
            raise ValueError("refinement_probes must be a positive even number.")
        if not 0.0 <= config.guidance_probability <= 1.0:
            raise ValueError("guidance_probability must be in [0, 1].")
        if not 0.0 <= config.directional_quantile <= 1.0:
            raise ValueError("directional_quantile must be in [0, 1].")
        if config.directional_gate_tolerance < 0.0:
            raise ValueError("directional_gate_tolerance cannot be negative.")
        if not 0.0 <= config.directional_positive_fraction_min <= 1.0:
            raise ValueError("directional_positive_fraction_min must be in [0, 1].")
        if not 0.0 <= config.service_effect_quantile <= 1.0:
            raise ValueError("service_effect_quantile must be in [0, 1].")
        if not 0.0 <= config.minimum_successful_service_fraction <= 1.0:
            raise ValueError("minimum_successful_service_fraction must be in [0, 1].")
        if not 0.0 <= config.confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in [0, 1].")
        if config.proxy_reconstruction_sre_drop_cap_db < 0.0:
            raise ValueError("proxy_reconstruction_sre_drop_cap_db cannot be negative.")
        if config.mean_sam_cap < 0.0:
            raise ValueError("mean_sam_cap cannot be negative.")
        if config.endpoint_orthogonal_drift_cap < 0.0:
            raise ValueError("endpoint_orthogonal_drift_cap cannot be negative.")
        if config.endpoint_total_drift_cap < 0.0:
            raise ValueError("endpoint_total_drift_cap cannot be negative.")
        if config.free_spectral_atoms < 0:
            raise ValueError("free_spectral_atoms cannot be negative.")
        if config.n_directions < 0:
            raise ValueError("n_directions cannot be negative.")
        if (
            config.n_directions == 0
            and config.free_spectral_atoms == 0
            and config.attack_mode != "targeted"
        ):
            raise ValueError(
                "Untargeted search needs at least one weak direction or free "
                "spectral atom."
            )
        if not 0.0 <= config.abundance_shift_quantile <= 1.0:
            raise ValueError("abundance_shift_quantile must be in [0, 1].")
        if config.maximum_candidate_queries < 0:
            raise ValueError("maximum_candidate_queries cannot be negative.")
        if config.operator_candidates_per_step < 1:
            raise ValueError("operator_candidates_per_step must be positive.")
        if config.moead_generations < 0:
            raise ValueError("moead_generations cannot be negative.")
        if config.moead_population_size < 4:
            raise ValueError("moead_population_size must be at least four.")
        if config.moead_neighbourhood_size < 2:
            raise ValueError("moead_neighbourhood_size must be at least two.")
        if not 0.0 <= config.moead_neighbour_mating_probability <= 1.0:
            raise ValueError("moead_neighbour_mating_probability must be in [0, 1].")
        if config.moead_maximum_replacements < 1:
            raise ValueError("moead_maximum_replacements must be positive.")
        if not 0.0 <= config.moead_block_crossover_probability <= 1.0:
            raise ValueError("moead_block_crossover_probability must be in [0, 1].")
        if config.refinement_optimizer not in ALL_OPTIMIZERS:
            raise ValueError(
                f"refinement_optimizer must be one of {ALL_OPTIMIZERS}."
            )
        self.services = dict(services)
        self.config = config

    @staticmethod
    def config_dict(config: WeakTangentConfig) -> dict:
        return asdict(config)

    def run(
        self,
        cube: np.ndarray,
        progress: Callable[[str], None] | None = print,
        *,
        geometry_service_names: tuple[str, ...] | None = None,
        use_target_direction: bool = True,
        alignment_anchors: np.ndarray | None = None,
        target_map_override: np.ndarray | None = None,
    ) -> WeakTangentResult:
        cube = np.asarray(cube, dtype=np.float32)
        ledger = QueryLedger()
        clean_outputs = {
            name: ledger.call("reference", name, service.query, cube)
            for name, service in self.services.items()
        }
        residuals = {name: float(value.relative_residual) for name, value in clean_outputs.items()}
        active = sorted(
            name for name, residual in residuals.items()
            if residual <= self.config.baseline_residual_cap
        )
        if len(active) < self.config.minimum_active_services:
            active = [
                name for name, _ in sorted(residuals.items(), key=lambda item: item[1])[
                    : self.config.minimum_active_services
                ]
            ]
        active_outputs = {name: clean_outputs[name] for name in active}
        canonical, baseline_abundances, aligned_endmembers = _canonical_clean_outputs(active_outputs)
        # Optional revision-study controls. Default calls preserve the original
        # operations; geometry-only ablations never remove scoring services.
        if alignment_anchors is not None:
            from scipy.optimize import linear_sum_assignment
            from .cross_unmixer import spectral_angle_matrix
            if np.asarray(alignment_anchors).shape != canonical.shape:
                raise ValueError("Alignment anchors must match the canonical dictionary.")
            rows, cols = linear_sum_assignment(spectral_angle_matrix(alignment_anchors, canonical))
            order = cols[np.argsort(rows)]
            canonical = canonical[:, order]
            baseline_abundances = {k: v[order] for k, v in baseline_abundances.items()}
            aligned_endmembers = {k: v[:, order] for k, v in aligned_endmembers.items()}
        consensus = np.median(np.stack(list(baseline_abundances.values()), axis=0), axis=0)
        consensus = np.maximum(consensus, 0.0)
        consensus /= np.maximum(np.sum(consensus, axis=0, keepdims=True), EPS)
        confidence_map = query_consensus_confidence(
            cube, baseline_abundances, consensus
        )
        target_map = (
            targeted_abundance_map(
                consensus,
                self.config.target_source_material,
                self.config.target_destination_material,
                self.config.target_region,
                self.config.target_region_fraction,
            )
            if self.config.attack_mode == "targeted"
            else np.zeros_like(consensus, dtype=np.float32)
        )
        if target_map_override is not None:
            target_map = np.asarray(target_map_override, dtype=np.float32).copy()
            if target_map.shape != consensus.shape or not np.all(np.isfinite(target_map)):
                raise ValueError("Invalid frozen target map.")
            if not np.allclose(target_map.sum(axis=0), 0.0, atol=1e-6):
                raise ValueError("Frozen target map must conserve abundance.")
        geometry = aligned_endmembers
        if geometry_service_names is not None:
            if not geometry_service_names or not set(geometry_service_names) <= set(active):
                raise ValueError("Geometry services must be a nonempty subset of active services.")
            geometry = {name: aligned_endmembers[name] for name in geometry_service_names}
        if self.config.direction_strategy == "random_tangent":
            base_directions, footprints = random_tangent_directions(
                geometry,
                self.config.n_directions,
                self.config.seed + 65537,
            )
        else:
            base_directions, footprints = robust_weak_directions(
                geometry,
                self.config.n_directions,
                self.config.candidate_directions_per_response,
            )
        anchors = anchor_mask_from_abundances(
            consensus,
            self.config.anchor_pixels_per_material,
            self.config.anchor_dilation,
        )
        clean_consensus_residual = float(
            np.linalg.norm(
                cube.reshape(-1, cube.shape[-1]).T
                - canonical @ consensus.reshape(consensus.shape[0], -1)
            )
            / max(float(np.linalg.norm(cube)), EPS)
        )
        if self.config.residual_variability_branch:
            directions, residual_atoms, direction_labels, residual_coupling, clean_consensus_residual = (
                build_material_variability_genes(
                    cube,
                    canonical,
                    consensus,
                    base_directions,
                    self.config.residual_variability_quantile,
                    self.config.residual_coupling_cap,
                )
            )
            footprints = np.asarray(
                [
                    float(np.max(np.abs(canonical @ directions[:, index] + residual_atoms[:, index])))
                    for index in range(directions.shape[1])
                ],
                dtype=np.float32,
            )
        else:
            directions = base_directions
            residual_atoms = np.zeros((cube.shape[-1], directions.shape[1]), dtype=np.float32)
            direction_labels = [f"d{index}_mix" for index in range(directions.shape[1])]
            residual_coupling = 0.0
        if self.config.attack_mode == "targeted" and use_target_direction:
            target_direction = np.zeros((directions.shape[0], 1), dtype=np.float32)
            target_direction[self.config.target_source_material, 0] = -1.0 / np.sqrt(2.0)
            target_direction[self.config.target_destination_material, 0] = 1.0 / np.sqrt(2.0)
            directions = np.concatenate([directions, target_direction], axis=1)
            residual_atoms = np.concatenate(
                [
                    residual_atoms,
                    np.zeros((cube.shape[-1], 1), dtype=np.float32),
                ],
                axis=1,
            )
            direction_labels.append(
                f"target_{self.config.target_source_material}_to_"
                f"{self.config.target_destination_material}"
            )
            footprints = np.concatenate(
                [
                    footprints,
                    np.asarray(
                        [float(np.max(np.abs(canonical @ target_direction[:, 0])))],
                        dtype=np.float32,
                    ),
                ]
            )
        free_atoms = data_spectral_atoms(cube, self.config.free_spectral_atoms)
        if free_atoms.shape[1]:
            free_count = free_atoms.shape[1]
            directions = np.concatenate(
                [
                    directions,
                    np.zeros((directions.shape[0], free_count), dtype=np.float32),
                ],
                axis=1,
            )
            residual_atoms = np.concatenate([residual_atoms, free_atoms], axis=1)
            direction_labels.extend(
                f"pca{index}_free_square" for index in range(free_count)
            )
            footprints = np.concatenate(
                [footprints, np.max(np.abs(free_atoms), axis=0).astype(np.float32)]
            )
        parameterization = WeakTangentParameterization(
            cube,
            canonical,
            consensus,
            directions,
            anchors,
            self.config.linf_cap,
            self.config.support_fraction_cap,
            self.config.mean_sam_cap,
            residual_atoms,
            self.config.local_linf_projection,
        )
        rng = np.random.default_rng(self.config.seed)
        cache: dict[bytes, WeakTangentRecord] = {}
        candidate_queries = 0
        service_queries = 0
        cache_hits = 0
        budget_fallback: list[WeakTangentRecord] = []
        cube_float64 = cube.astype(np.float64)
        abundance_pixel_scale = np.sqrt(cube.shape[0] * cube.shape[1])
        cube_value_scale = np.sqrt(cube.size)
        fixed_clean_abundance = fixed_dictionary_abundances(cube, canonical)
        ledger.local["q0_reference_evaluations"] += 1
        feasibility = self._feasibility_map(
            consensus,
            directions,
            anchors,
            confidence_map if self.config.high_confidence_guidance else None,
            self.config.confidence_floor,
        )
        initial_side = min(
            max(self.config.minimum_block_side, self.config.initial_block_side),
            cube.shape[0],
            cube.shape[1],
            max(1, int(np.floor(np.sqrt(parameterization.maximum_support)))),
        )

        def evaluate(coefficients: np.ndarray, *, accounting_phase: str = "search") -> WeakTangentRecord:
            nonlocal candidate_queries, service_queries, cache_hits
            ledger.requests[accounting_phase] += 1
            attacked, planned_delta, feasible_coefficients = parameterization.apply(
                coefficients, support_is_limited=True
            )
            key = hashlib.blake2b(
                memoryview(np.ascontiguousarray(attacked)).cast("B"), digest_size=16
            ).digest()
            if key in cache:
                cache_hits += 1
                ledger.cache_hits[accounting_phase] += 1
                return cache[key]
            if (
                self.config.maximum_candidate_queries > 0
                and candidate_queries >= self.config.maximum_candidate_queries
            ):
                if not budget_fallback:
                    raise RuntimeError(
                        "The candidate-query budget must permit the clean baseline query."
                    )
                ledger.budget_returns[accounting_phase] += 1
                return budget_fallback[0]
            shifts: dict[str, float] = {}
            maps: list[np.ndarray] = []
            drifts: list[float] = []
            directional_scores: dict[str, float] = {}
            directional_quantiles: dict[str, float] = {}
            positive_fractions: dict[str, float] = {}
            directional_coverages: list[dict[str, float]] = []
            coverage_aucs: dict[str, float] = {}
            directional_maps: list[np.ndarray] = []
            endpoint_parallel: list[float] = []
            endpoint_orthogonal: list[float] = []
            endpoint_total: list[float] = []
            reconstruction_drops: dict[str, float] = {}
            for name in active:
                service = self.services[name]
                compact = getattr(service, "query_compact", None)
                output = ledger.call(
                    accounting_phase, name, compact if callable(compact) else service.query, attacked
                )
                service_queries += 1
                aligned_abundance, attacked_endmembers, drift = match_to_anchors(
                    aligned_endmembers[name], output
                )
                difference = aligned_abundance - baseline_abundances[name]
                shifts[name] = float(
                    np.linalg.norm(difference) / abundance_pixel_scale
                )
                maps.append(np.linalg.norm(difference, axis=0))
                drifts.append(float(drift))
                scoring_delta = (
                    target_map
                    if self.config.attack_mode == "targeted"
                    else planned_delta
                )
                directional_stats, directional_map = directional_transfer_statistics(
                    scoring_delta,
                    difference,
                    self.config.directional_quantile,
                )
                directional_scores[name] = directional_stats["global_rms"]
                directional_quantiles[name] = directional_stats["gate_quantile"]
                positive_fractions[name] = directional_stats["positive_fraction"]
                coverage_aucs[name] = directional_stats["coverage_auc"]
                directional_coverages.append(directional_stats)
                directional_maps.append(directional_map)
                parallel, orthogonal = endpoint_directional_decomposition(
                    aligned_endmembers[name], attacked_endmembers, scoring_delta
                )
                endpoint_parallel.append(parallel)
                endpoint_orthogonal.append(orthogonal)
                endpoint_total.append(
                    endpoint_total_drift_ratio(aligned_endmembers[name], attacked_endmembers)
                )
                reconstruction = (
                    attacked_endmembers.astype(np.float64)
                    @ aligned_abundance.reshape(aligned_abundance.shape[0], -1).astype(
                        np.float64
                    )
                ).T.reshape(attacked.shape)
                attacked_sre = reconstruction_sre_db(attacked, reconstruction)
                reconstruction_drops[name] = max(
                    0.0,
                    float(clean_outputs[name].reconstruction_sre_db) - attacked_sre,
                )
            fixed_attacked_abundance = fixed_dictionary_abundances(attacked, canonical)
            ledger.local["q0_candidate_evaluations"] += 1
            scoring_delta = (
                target_map
                if self.config.attack_mode == "targeted"
                else planned_delta
            )
            fixed_stats, fixed_directional_map = directional_transfer_statistics(
                scoring_delta,
                fixed_attacked_abundance - fixed_clean_abundance,
                self.config.directional_quantile,
            )
            fixed_directional = fixed_stats["global_rms"]
            all_directional = [*directional_scores.values(), fixed_directional]
            all_quantiles = [*directional_quantiles.values(), fixed_stats["gate_quantile"]]
            all_positive_fractions = [*positive_fractions.values(), fixed_stats["positive_fraction"]]
            directional_coverages.append(fixed_stats)
            all_coverage_aucs = [*coverage_aucs.values(), fixed_stats["coverage_auc"]]
            successful = [
                quantile >= -self.config.directional_gate_tolerance
                and fraction >= self.config.directional_positive_fraction_min
                for quantile, fraction in zip(all_quantiles, all_positive_fractions)
            ]
            successful_fraction = float(np.mean(successful))
            if self.config.coverage_curve_objective:
                raw_robust_directional = float(
                    np.quantile(all_directional, self.config.service_effect_quantile)
                )
                robust_coverage_auc = float(
                    np.quantile(all_coverage_aucs, self.config.service_effect_quantile)
                )
                robust_quantile = float(
                    np.quantile(all_quantiles, self.config.service_effect_quantile)
                )
                minimum_positive_fraction = float(
                    np.quantile(all_positive_fractions, self.config.service_effect_quantile)
                )
            else:
                raw_robust_directional = float(min(all_directional))
                robust_coverage_auc = float(min(all_coverage_aucs))
                robust_quantile = float(min(all_quantiles))
                minimum_positive_fraction = float(min(all_positive_fractions))
            maximum_orthogonal = float(max(endpoint_orthogonal, default=0.0))
            maximum_total = float(max(endpoint_total, default=0.0))
            maximum_reconstruction_drop = float(
                max(reconstruction_drops.values(), default=0.0)
            )
            directional_consensus = bool(
                (
                    successful_fraction
                    >= self.config.minimum_successful_service_fraction
                    if self.config.coverage_curve_objective
                    else (
                        robust_quantile >= -self.config.directional_gate_tolerance
                        and minimum_positive_fraction
                        >= self.config.directional_positive_fraction_min
                    )
                )
            )
            physical_feasible = bool(
                maximum_orthogonal <= self.config.endpoint_orthogonal_drift_cap
                and maximum_total <= self.config.endpoint_total_drift_cap
                and maximum_reconstruction_drop
                <= self.config.proxy_reconstruction_sre_drop_cap_db
            )
            directional_feasible = directional_consensus and physical_feasible
            robust_directional = (
                raw_robust_directional
                if directional_feasible
                else min(0.0, raw_robust_directional)
            )
            delta = attacked.astype(np.float64) - cube_float64
            modified = np.any(np.abs(delta) > 1e-7, axis=-1)
            robust = float(
                np.quantile(
                    np.asarray(list(shifts.values()), dtype=np.float64),
                    self.config.abundance_shift_quantile,
                )
            )
            energy = float(np.linalg.norm(delta) / cube_value_scale)
            measures = WeakTangentMeasures(
                robust_abundance_shift=robust,
                mean_abundance_shift=float(np.mean(list(shifts.values()))),
                relative_energy=energy,
                linf=float(np.max(np.abs(delta))),
                support_fraction=float(np.mean(modified)),
                mean_sam=mean_spectral_angle(cube, attacked),
                mean_endmember_drift=float(np.mean(drifts)),
                per_response_shift=shifts,
                robust_directional_transfer=robust_directional,
                mean_directional_transfer=float(np.mean(all_directional)),
                fixed_directional_transfer=float(fixed_directional),
                robust_directional_quantile=robust_quantile,
                fixed_directional_quantile=float(fixed_stats["gate_quantile"]),
                minimum_positive_fraction=minimum_positive_fraction,
                effective_coverage_001=float(
                    np.quantile(
                        [item["coverage_001"] for item in directional_coverages],
                        self.config.service_effect_quantile,
                    )
                ),
                effective_coverage_002=float(
                    np.quantile(
                        [item["coverage_002"] for item in directional_coverages],
                        self.config.service_effect_quantile,
                    )
                ),
                effective_coverage_005=float(
                    np.quantile(
                        [item["coverage_005"] for item in directional_coverages],
                        self.config.service_effect_quantile,
                    )
                ),
                effective_coverage_010=float(
                    np.quantile(
                        [item["coverage_010"] for item in directional_coverages],
                        self.config.service_effect_quantile,
                    )
                ),
                effective_coverage_015=float(
                    np.quantile(
                        [item["coverage_015"] for item in directional_coverages],
                        self.config.service_effect_quantile,
                    )
                ),
                effective_coverage_020=float(
                    np.quantile(
                        [item["coverage_020"] for item in directional_coverages],
                        self.config.service_effect_quantile,
                    )
                ),
                # Directional agreement is a graded robustness signal rather
                # than a hard constraint.  Keeping the measured coverage when
                # only part of the proxy pool agrees prevents the zero-input
                # solution from winning solely because every nonzero candidate
                # narrowly missed an all-or-nothing consensus gate.  Physical
                # stealth/reconstruction constraints remain hard.
                robust_coverage_auc=(
                    robust_coverage_auc
                    if physical_feasible
                    and (
                        not self.config.directional_hard_gate
                        or directional_consensus
                    )
                    else 0.0
                ),
                fixed_coverage_auc=float(fixed_stats["coverage_auc"]),
                successful_service_fraction=successful_fraction,
                maximum_proxy_reconstruction_sre_drop_db=maximum_reconstruction_drop,
                mean_proxy_reconstruction_sre_drop_db=float(
                    np.mean(list(reconstruction_drops.values()))
                ),
                directional_feasible=directional_feasible,
                endpoint_parallel_progress=float(min(endpoint_parallel))
                if endpoint_parallel
                else 0.0,
                maximum_endpoint_orthogonal_ratio=maximum_orthogonal,
                maximum_endpoint_total_drift_ratio=maximum_total,
                per_response_directional_transfer=directional_scores,
                per_response_directional_quantile=directional_quantiles,
                per_response_positive_fraction=positive_fractions,
                per_response_coverage_auc=coverage_aucs,
                per_response_reconstruction_sre_drop_db=reconstruction_drops,
            )
            if self.config.attack_mode == "untargeted":
                effect = robust
            elif self.config.attack_mode == "targeted":
                raw_effect = (
                    measures.robust_coverage_auc
                    if self.config.coverage_curve_objective
                    else robust_directional
                )
                effect = raw_effect * (
                    0.25 + 0.75 * measures.successful_service_fraction
                )
            elif self.config.coverage_curve_objective:
                effect = measures.robust_coverage_auc
            elif self.config.directional_objective:
                effect = robust_directional
            else:
                effect = robust
            record = WeakTangentRecord(
                coefficients=feasible_coefficients.copy(),
                objectives=np.asarray([-effect, energy], dtype=np.float64),
                measures=measures,
                response_map=(
                    np.maximum(
                        np.median(
                            np.stack(
                                [
                                    *directional_maps,
                                    fixed_directional_map,
                                ],
                                axis=0,
                            ),
                            axis=0,
                        ),
                        0.0,
                    ).astype(np.float32)
                    if self.config.directional_objective
                    or self.config.attack_mode == "targeted"
                    else np.median(np.stack(maps, axis=0), axis=0).astype(np.float32)
                ),
            )
            candidate_queries += 1
            ledger.evaluations[accounting_phase] += 1
            cache[key] = record
            return record

        zero = evaluate(parameterization.empty_coefficients())
        budget_fallback.append(zero)
        archive = [zero]
        incumbent = zero
        operator_arms = [
            (direction, sign, amplitude)
            for direction in range(directions.shape[1])
            for sign in (-1.0, 1.0)
            for amplitude in (0.35, 0.60, 1.00)
        ]
        operator_attempts = np.zeros(len(operator_arms), dtype=np.int64)
        operator_successes = np.zeros(len(operator_arms), dtype=np.int64)

        for direction in range(directions.shape[1]):
            for sign in (-1.0, 1.0):
                coefficients = parameterization.empty_coefficients()
                center = int(np.argmax(feasibility[direction, 0 if sign < 0 else 1]))
                block = self._square_mask(center, initial_side, cube.shape[:2]) & ~anchors
                coefficients[direction, block] = np.float32(sign)
                coefficients = parameterization.limit_support(coefficients, block)
                record = evaluate(coefficients)
                archive = _archive_insert(archive, record, self.config.pareto_archive_size)
                if self._is_stronger(record, incumbent):
                    incumbent = record

        for step in range(self.config.square_steps):
            side = self._scheduled_side(step, self.config.square_steps, initial_side)
            center = self._sample_center(
                rng,
                incumbent.response_map,
                feasibility,
                anchors,
                side,
            )
            block = self._square_mask(center, side, cube.shape[:2]) & ~anchors
            if self.config.adaptive_operator_search:
                draws = rng.beta(
                    operator_successes.astype(np.float64) + 1.0,
                    operator_attempts.astype(np.float64)
                    - operator_successes.astype(np.float64)
                    + 1.0,
                )
                arm_indices = np.argsort(draws)[::-1][
                    : min(self.config.operator_candidates_per_step, len(operator_arms))
                ]
            else:
                arm_indices = rng.choice(
                    len(operator_arms),
                    size=min(self.config.operator_candidates_per_step, len(operator_arms)),
                    replace=False,
                )
            proposals: list[tuple[int, WeakTangentRecord]] = []
            for arm_index_value in np.atleast_1d(arm_indices):
                arm_index = int(arm_index_value)
                direction, sign, amplitude = operator_arms[arm_index]
                proposal = incumbent.coefficients.copy()
                proposal[:, block] = 0.0
                proposal[direction, block] = np.float32(sign * amplitude)
                proposal = parameterization.limit_support(
                    proposal, preferred_mask=block, response_map=incumbent.response_map
                )
                record = evaluate(proposal)
                archive = _archive_insert(
                    archive, record, self.config.pareto_archive_size
                )
                proposals.append((arm_index, record))
            arm_index, record = max(
                proposals,
                key=lambda item: (
                    self._effect(item[1]),
                    -item[1].measures.relative_energy,
                ),
            )
            improved = self._is_stronger(record, incumbent)
            operator_attempts[arm_index] += 1
            if improved:
                operator_successes[arm_index] += 1
                incumbent = record
            if progress is not None and (step + 1) % max(1, self.config.square_steps // 4) == 0:
                progress(
                    f"[WT-PARS square] {step + 1}/{self.config.square_steps} "
                        f"effect={self._effect(incumbent):.6f}"
                )

        for step in range(self.config.refinement_steps):
            side = max(
                self.config.minimum_block_side,
                self._scheduled_side(step, self.config.refinement_steps, max(initial_side // 2, 2)),
            )
            center = self._sample_center(
                rng,
                incumbent.response_map,
                feasibility,
                anchors,
                side,
                force_guidance=True,
            )
            block = self._square_mask(center, side, cube.shape[:2]) & ~anchors
            gradient = np.zeros(directions.shape[1], dtype=np.float64)
            pairs = self.config.refinement_probes // 2
            sigma = self.config.coefficient_step * (0.5 ** (step // max(1, self.config.refinement_steps // 3)))
            for _ in range(pairs):
                probe = rng.choice(np.asarray([-1.0, 1.0]), size=directions.shape[1])
                plus = incumbent.coefficients.copy()
                minus = incumbent.coefficients.copy()
                plus[:, block] += (sigma * probe)[:, None]
                minus[:, block] -= (sigma * probe)[:, None]
                plus = np.clip(plus, -1.0, 1.0)
                minus = np.clip(minus, -1.0, 1.0)
                plus = parameterization.limit_support(plus, block, incumbent.response_map)
                minus = parameterization.limit_support(minus, block, incumbent.response_map)
                plus_record = evaluate(plus)
                minus_record = evaluate(minus)
                archive = _archive_insert(archive, plus_record, self.config.pareto_archive_size)
                archive = _archive_insert(archive, minus_record, self.config.pareto_archive_size)
                gradient += (
                    self._effect(plus_record) - self._effect(minus_record)
                ) * probe
                if self._is_stronger(plus_record, incumbent):
                    incumbent = plus_record
                if self._is_stronger(minus_record, incumbent):
                    incumbent = minus_record
            if np.any(np.abs(gradient) > EPS):
                proposal = incumbent.coefficients.copy()
                proposal[:, block] += (
                    self.config.coefficient_step * np.sign(gradient)
                )[:, None]
                proposal = np.clip(proposal, -1.0, 1.0)
                proposal = parameterization.limit_support(proposal, block, incumbent.response_map)
                record = evaluate(proposal)
                archive = _archive_insert(archive, record, self.config.pareto_archive_size)
                if self._is_stronger(record, incumbent):
                    incumbent = record
            if progress is not None:
                progress(
                    f"[WT-PARS refine] {step + 1}/{self.config.refinement_steps} "
                    f"effect={self._effect(incumbent):.6f}"
                )

        # Optional MOEA/D refinement.  The complete WT-PARS archive remains in
        # ``archive`` and the incumbent is explicitly included in the final
        # selection pool.  Consequently this stage can improve the black-box
        # objective but cannot replace the established result with a weaker
        # one.  Standard MOEA/D decomposition/neighbour replacement is paired
        # with the same spatial Square operator used above, rather than a dense
        # polynomial mutation that would destroy HSI spatial coherence.
        # Protect the exact endpoint that the unchanged WT-PARS selector would
        # have saved, rather than merely the last accepted incumbent.
        protected_incumbent = self._select(archive)
        incumbent = protected_incumbent
        moead_attempts = np.zeros(len(operator_arms), dtype=np.int64)
        moead_successes = np.zeros(len(operator_arms), dtype=np.int64)
        optimizer_evaluations = 0
        optimizer_generations = 0
        optimizer_implementation = "disabled"
        if self.config.moead_generations > 0:
            coefficient_shape = incumbent.coefficients.shape
            n_variables = int(np.prod(coefficient_shape))
            moead_rng = np.random.default_rng(self.config.seed + 104729)
            last_arm = {-1: 0}

            def encode(coefficients: np.ndarray) -> np.ndarray:
                return np.clip(
                    0.5 * (np.asarray(coefficients, dtype=np.float64).reshape(-1) + 1.0),
                    0.0,
                    1.0,
                )

            def decode(genome: np.ndarray) -> np.ndarray:
                return (2.0 * np.asarray(genome, dtype=np.float64) - 1.0).reshape(
                    coefficient_shape
                ).astype(np.float32)

            def repair(genome: np.ndarray) -> np.ndarray:
                coefficients = parameterization.limit_support(decode(genome))
                return encode(coefficients)

            def moead_evaluate(genome: np.ndarray) -> np.ndarray:
                nonlocal archive, incumbent
                record = evaluate(parameterization.limit_support(decode(genome)))
                archive = _archive_insert(
                    archive, record, self.config.pareto_archive_size
                )
                if self._is_stronger(record, incumbent):
                    incumbent = record
                return record.objectives.copy()

            def offspring_operator(
                first: np.ndarray,
                second: np.ndarray,
                generation: int,
                subproblem: int,
                rng_value: np.random.Generator,
            ) -> np.ndarray:
                first_coefficients = decode(first)
                second_coefficients = decode(second)
                side = self._scheduled_side(
                    generation - 1,
                    max(self.config.moead_generations, 1),
                    initial_side,
                )
                center = self._sample_center(
                    rng_value,
                    np.zeros(cube.shape[:2], dtype=np.float32),
                    feasibility,
                    anchors,
                    side,
                )
                block = self._square_mask(center, side, cube.shape[:2]) & ~anchors
                child = first_coefficients.copy()
                if rng_value.random() < self.config.moead_block_crossover_probability:
                    child[:, block] = second_coefficients[:, block]
                if self.config.moead_adaptive_operator_selection:
                    draws = rng_value.beta(
                        moead_successes.astype(np.float64) + 1.0,
                        moead_attempts.astype(np.float64)
                        - moead_successes.astype(np.float64)
                        + 1.0,
                    )
                    arm_index = int(np.argmax(draws))
                else:
                    arm_index = int(rng_value.integers(len(operator_arms)))
                direction, sign, amplitude = operator_arms[arm_index]
                child[:, block] = 0.0
                child[direction, block] = np.float32(sign * amplitude)
                child = parameterization.limit_support(child, preferred_mask=block)
                last_arm[subproblem] = arm_index
                return encode(child)

            def offspring_feedback(
                _child: np.ndarray,
                _objective: np.ndarray,
                replacements: int,
                _generation: int,
                subproblem: int,
            ) -> None:
                arm_index = last_arm.get(subproblem)
                if arm_index is None:
                    return
                moead_attempts[arm_index] += 1
                if replacements > 0:
                    moead_successes[arm_index] += 1

            # Seed all subproblems with feasible, spatially structured records.
            # The incumbent and zero solution are always present; the remaining
            # slots use Pareto records and inexpensive coefficient mutations.
            initial_genomes = None
            if self.config.moead_structured_initialization:
                initial_coefficients: list[np.ndarray] = [
                    protected_incumbent.coefficients.copy(),
                    parameterization.empty_coefficients(),
                ]
                initial_coefficients.extend(
                    record.coefficients.copy() for record in archive
                )
                while len(initial_coefficients) < self.config.moead_population_size:
                    base = initial_coefficients[
                        int(moead_rng.integers(len(initial_coefficients)))
                    ].copy()
                    side = int(
                        moead_rng.integers(
                            self.config.minimum_block_side,
                            max(
                                self.config.minimum_block_side + 1,
                                initial_side + 1,
                            ),
                        )
                    )
                    center = self._sample_center(
                        moead_rng,
                        np.zeros(cube.shape[:2], dtype=np.float32),
                        feasibility,
                        anchors,
                        side,
                    )
                    block = (
                        self._square_mask(center, side, cube.shape[:2]) & ~anchors
                    )
                    arm_index = int(moead_rng.integers(len(operator_arms)))
                    direction, sign, amplitude = operator_arms[arm_index]
                    base[:, block] = 0.0
                    base[direction, block] = np.float32(sign * amplitude)
                    initial_coefficients.append(
                        parameterization.limit_support(base, preferred_mask=block)
                    )
                initial_genomes = np.stack(
                    [
                        encode(value)
                        for value in initial_coefficients[
                            : self.config.moead_population_size
                        ]
                    ],
                    axis=0,
                )
            def moead_progress(
                generation: int,
                _genomes: np.ndarray,
                _objectives: np.ndarray,
            ) -> None:
                if progress is not None:
                    progress(
                        f"[WT-{self.config.refinement_optimizer}] "
                        f"{generation}/{self.config.moead_generations} "
                        f"effect={self._effect(incumbent):.6f}"
                    )

            optimizer_result = minimize_with_optimizer(
                self.config.refinement_optimizer,
                moead_evaluate,
                n_variables=n_variables,
                population_size=self.config.moead_population_size,
                max_evaluations=(
                    self.config.moead_population_size
                    * (self.config.moead_generations + 1)
                ),
                seed=self.config.seed + 104729,
                repair=repair,
                neighbourhood_size=self.config.moead_neighbourhood_size,
                neighbour_mating_probability=self.config.moead_neighbour_mating_probability,
                maximum_replacements=self.config.moead_maximum_replacements,
                proposed_offspring_operator=(
                    offspring_operator
                    if self.config.moead_structured_operators
                    else None
                ),
                # Feedback is always collected for structured operators so the
                # uniform-selection ablation has identical diagnostics.  It
                # changes future choices only when adaptive selection is on.
                proposed_offspring_feedback=(
                    offspring_feedback
                    if self.config.moead_structured_operators
                    else None
                ),
                callback=moead_progress,
                initial_genomes=initial_genomes,
            )
            optimizer_evaluations = optimizer_result.evaluations
            optimizer_generations = optimizer_result.generations
            optimizer_implementation = optimizer_result.implementation

        # Even if archive truncation removes an endpoint, include the protected
        # WT-PARS result explicitly.  This is the non-degradation safeguard.
        archive = _archive_insert(
            archive, protected_incumbent, self.config.pareto_archive_size
        )

        effect_first = self._select(archive)
        if self.config.moead_generations > 0 and self.config.moead_safe_selection:
            # Deployment-safe output: an MOEA/D descendant may replace the
            # established WT-PARS result only when it does not trade additional
            # input energy for its proxy-effect gain.  The unrestricted
            # effect-first endpoint is retained for diagnostics and ablation.
            protected_effect = self._effect(protected_incumbent)
            protected_energy = protected_incumbent.measures.relative_energy
            safe_pool = [
                record
                for record in [*archive, protected_incumbent]
                if self._effect(record) >= protected_effect - 1e-12
                and record.measures.relative_energy <= protected_energy + 1e-12
            ]
            selected = self._select(safe_pool)
        else:
            selected = effect_first
        selected_record = evaluate(selected.coefficients, accounting_phase="selection")
        selected_cube, _, _ = parameterization.apply(
            selected_record.coefficients, support_is_limited=True
        )
        return WeakTangentResult(
            attacked_cube=selected_cube,
            canonical_endmembers=canonical,
            consensus_abundances=consensus.astype(np.float32),
            directions=directions,
            direction_footprints=footprints,
            direction_labels=direction_labels,
            residual_coupling=float(residual_coupling),
            clean_consensus_relative_residual=float(clean_consensus_residual),
            anchor_mask=anchors,
            confidence_map=confidence_map,
            target_map=target_map,
            selected=selected_record,
            pareto=archive,
            active_services=active,
            baseline_residuals=residuals,
            candidate_queries=candidate_queries,
            service_queries=service_queries,
            cache_hits=cache_hits,
            operator_attempts={
                self._operator_name(arm): int(operator_attempts[index])
                for index, arm in enumerate(operator_arms)
            },
            operator_successes={
                self._operator_name(arm): int(operator_successes[index])
                for index, arm in enumerate(operator_arms)
            },
            moead_operator_attempts={
                self._operator_name(arm): int(moead_attempts[index])
                for index, arm in enumerate(operator_arms)
            },
            moead_operator_successes={
                self._operator_name(arm): int(moead_successes[index])
                for index, arm in enumerate(operator_arms)
            },
            moead_effect_first=(
                effect_first if self.config.moead_generations > 0 else None
            ),
            moead_protected_baseline=(
                protected_incumbent if self.config.moead_generations > 0 else None
            ),
            optimizer_evaluations=optimizer_evaluations,
            optimizer_generations=optimizer_generations,
            optimizer_implementation=optimizer_implementation,
            query_accounting=ledger.snapshot(),
        )

    @staticmethod
    def _operator_name(arm: tuple[int, float, float]) -> str:
        direction, sign, amplitude = arm
        return f"d{direction}_s{int(sign):+d}_a{amplitude:.2f}"

    def _effect(self, record: WeakTangentRecord) -> float:
        if self.config.attack_mode == "untargeted":
            return record.measures.robust_abundance_shift
        if self.config.attack_mode == "targeted":
            raw = (
                record.measures.robust_coverage_auc
                if self.config.coverage_curve_objective
                else record.measures.robust_directional_transfer
            )
            # Reward cross-proxy agreement continuously.  A partial but
            # genuinely directional transfer remains searchable; full proxy
            # agreement still receives the largest score.
            agreement = 0.25 + 0.75 * record.measures.successful_service_fraction
            return raw * agreement
        if self.config.coverage_curve_objective:
            return record.measures.robust_coverage_auc
        return (
            record.measures.robust_directional_transfer
            if self.config.directional_objective
            else record.measures.robust_abundance_shift
        )

    def _is_stronger(self, first: WeakTangentRecord, second: WeakTangentRecord) -> bool:
        first_effect = self._effect(first)
        second_effect = self._effect(second)
        return (
            first_effect > second_effect + 1e-12
            or (
                abs(first_effect - second_effect) <= 1e-12
                and first.measures.relative_energy < second.measures.relative_energy
            )
        )

    def _select(self, archive: list[WeakTangentRecord]) -> WeakTangentRecord:
        best = max(self._effect(record) for record in archive)
        tolerance = float(self.config.selection_effect_tolerance)
        plateau = [
            record for record in archive
            if self._effect(record) >= (1.0 - tolerance) * best - 1e-12
        ]
        return min(
            plateau,
            key=lambda record: (
                record.measures.relative_energy,
                -record.measures.mean_abundance_shift,
            ),
        )

    @staticmethod
    def _feasibility_map(
        abundances: np.ndarray,
        directions: np.ndarray,
        anchors: np.ndarray,
        confidence: np.ndarray | None = None,
        confidence_floor: float = 0.10,
    ) -> np.ndarray:
        materials, height, width = abundances.shape
        output = np.zeros((directions.shape[1], 2, height, width), dtype=np.float64)
        flat_abundance = abundances.reshape(materials, -1)
        for index in range(directions.shape[1]):
            for sign_index, sign in enumerate((-1.0, 1.0)):
                direction = sign * directions[:, index]
                negative = direction < 0.0
                if np.any(negative):
                    feasible = np.min(
                        flat_abundance[negative] / np.maximum(-direction[negative, None], EPS),
                        axis=0,
                    )
                else:
                    feasible = np.ones(height * width, dtype=np.float64)
                if confidence is not None:
                    confidence_weight = confidence_floor + (
                        1.0 - confidence_floor
                    ) * np.asarray(confidence, dtype=np.float64).reshape(-1)
                    # Prefer pixels whose clean consensus is dominated by the
                    # material that the planned direction will suppress.
                    dominant = np.argmax(flat_abundance, axis=0)
                    source = int(np.argmin(direction))
                    source_alignment = np.maximum(
                        -direction[dominant], 0.0
                    ) / max(float(-np.min(direction)), EPS)
                    feasible *= confidence_weight * (
                        confidence_floor + (1.0 - confidence_floor) * source_alignment
                    )
                output[index, sign_index] = feasible.reshape(height, width)
        output[:, :, anchors] = 0.0
        return output

    def _scheduled_side(self, step: int, total: int, maximum: int) -> int:
        if total <= 1:
            return max(self.config.minimum_block_side, int(maximum))
        fraction = step / max(total - 1, 1)
        value = maximum * (
            max(self.config.minimum_block_side, 1) / max(maximum, 1)
        ) ** fraction
        return max(self.config.minimum_block_side, int(round(value)))

    @staticmethod
    def _square_mask(center: int, side: int, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        row, column = divmod(int(center), width)
        half = max(0, int(side) // 2)
        top = min(max(0, row - half), max(0, height - side))
        left = min(max(0, column - half), max(0, width - side))
        mask = np.zeros((height, width), dtype=bool)
        mask[top : min(height, top + side), left : min(width, left + side)] = True
        return mask

    def _sample_center(
        self,
        rng: np.random.Generator,
        response_map: np.ndarray,
        feasibility: np.ndarray,
        anchors: np.ndarray,
        side: int,
        force_guidance: bool = False,
    ) -> int:
        height, width = anchors.shape
        eligible = ~anchors
        use_guidance = force_guidance or rng.random() < self.config.guidance_probability
        weights = np.ones((height, width), dtype=np.float64)
        if use_guidance and float(np.max(response_map)) > EPS:
            local = uniform_filter(
                np.asarray(response_map, dtype=np.float64),
                size=max(1, int(side)),
                mode="nearest",
            )
            local = (local - np.mean(local)) / (np.std(local) + EPS)
            weights = np.exp(
                np.clip(local / max(self.config.guidance_temperature, EPS), -20.0, 20.0)
            )
        feasible = np.max(feasibility, axis=(0, 1))
        feasible /= max(float(np.max(feasible)), EPS)
        weights *= 0.1 + feasible
        weights[~eligible] = 0.0
        probabilities = weights.reshape(-1)
        if float(np.sum(probabilities)) <= EPS:
            probabilities = eligible.reshape(-1).astype(np.float64)
        probabilities /= np.sum(probabilities)
        return int(rng.choice(height * width, p=probabilities))
