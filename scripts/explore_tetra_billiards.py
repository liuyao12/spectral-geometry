#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - scipy is optional for candidate generation.
    least_squares = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import tetrahedron_exhaustive_closed_paths as exact
except Exception:  # pragma: no cover - the public repo may not carry the verifier yet.
    exact = None


FACE_LABELS = "ABCD"
FACE_INDEX = {face: i for i, face in enumerate(FACE_LABELS)}
VERTICES = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=float,
)
LAMBDA_GRADS = np.array(
    [
        [0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
    ],
    dtype=float,
)
LAMBDA_OFFSETS = np.array([0.5, 0.5, 0.5, -0.5], dtype=float)


@dataclass
class BounceState:
    face: str
    point: np.ndarray
    direction: np.ndarray


@dataclass
class Candidate:
    word: str
    period: int
    score: float
    point_error: float
    direction_error: float
    trial: int
    start_index: int
    end_index: int
    refined_score: Optional[float] = None
    refined_point_error: Optional[float] = None
    refined_direction_error: Optional[float] = None
    exact_reason: Optional[str] = None


def lambdas(point: np.ndarray) -> np.ndarray:
    return LAMBDA_GRADS @ point + LAMBDA_OFFSETS


def point_from_barycentric(weights: np.ndarray) -> np.ndarray:
    return weights @ VERTICES


def sample_face_point(rng: np.random.Generator, face: str, edge_margin: float) -> np.ndarray:
    idx = FACE_INDEX[face]
    others = [j for j in range(4) if j != idx]
    raw = rng.dirichlet(np.ones(3))
    if edge_margin > 0:
        raw = edge_margin + (1.0 - 3.0 * edge_margin) * raw
    weights = np.zeros(4)
    weights[others] = raw
    return point_from_barycentric(weights)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def sample_outgoing_direction(rng: np.random.Generator, face: str) -> np.ndarray:
    idx = FACE_INDEX[face]
    direction = unit(rng.normal(size=3))
    if float(LAMBDA_GRADS[idx] @ direction) <= 0:
        direction = -direction
    if abs(float(LAMBDA_GRADS[idx] @ direction)) < 1e-9:
        direction = unit(direction + LAMBDA_GRADS[idx])
    return direction


def reflect_direction(direction: np.ndarray, face: str) -> np.ndarray:
    normal = LAMBDA_GRADS[FACE_INDEX[face]]
    return direction - 2.0 * float(direction @ normal) / float(normal @ normal) * normal


def next_bounce(point: np.ndarray, direction: np.ndarray, eps: float) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
    values = lambdas(point)
    derivs = LAMBDA_GRADS @ direction
    hits: List[Tuple[float, int]] = []
    for idx in range(4):
        if abs(derivs[idx]) < eps:
            continue
        t = -values[idx] / derivs[idx]
        if t > eps:
            hits.append((float(t), idx))
    if not hits:
        return None
    t_min = min(t for t, _ in hits)
    hit_indices = [idx for t, idx in hits if abs(t - t_min) <= 1e-8]
    if len(hit_indices) != 1:
        return None
    face = FACE_LABELS[hit_indices[0]]
    hit = point + t_min * direction
    hit_lambdas = lambdas(hit)
    if np.any(np.delete(hit_lambdas, hit_indices[0]) <= eps):
        return None
    return face, hit, unit(reflect_direction(direction, face))


def shoot_trial(
    rng: np.random.Generator,
    max_bounces: int,
    min_period: int,
    max_period: int,
    threshold: float,
    edge_margin: float,
    eps: float,
    trial: int,
) -> List[Candidate]:
    start_face = str(rng.choice(list(FACE_LABELS)))
    point = sample_face_point(rng, start_face, edge_margin)
    direction = sample_outgoing_direction(rng, start_face)
    states = [BounceState(start_face, point, direction)]
    candidates: List[Candidate] = []

    for _ in range(max_bounces):
        nxt = next_bounce(states[-1].point, states[-1].direction, eps)
        if nxt is None:
            break
        face, hit, new_direction = nxt
        states.append(BounceState(face, hit, new_direction))
        end = len(states) - 1
        low = max(0, end - max_period)
        high = end - min_period
        for start in range(low, high + 1):
            if states[start].face != face:
                continue
            period = end - start
            word = "".join(state.face for state in states[start:end])
            if any(word[i] == word[(i + 1) % len(word)] for i in range(len(word))):
                continue
            point_error = float(np.linalg.norm(states[end].point - states[start].point))
            direction_error = float(np.linalg.norm(unit(states[end].direction) - unit(states[start].direction)))
            score = math.hypot(point_error, direction_error)
            if score <= threshold:
                candidates.append(
                    Candidate(
                        word=word,
                        period=period,
                        score=score,
                        point_error=point_error,
                        direction_error=direction_error,
                        trial=trial,
                        start_index=start,
                        end_index=end,
                    )
                )
    return candidates


def barycentric_logits_for_face(point: np.ndarray, face: str) -> np.ndarray:
    values = np.maximum(lambdas(point), 1e-12)
    idx = FACE_INDEX[face]
    return np.log(np.delete(values, idx))


def point_from_face_logits(logits: np.ndarray, face: str) -> np.ndarray:
    idx = FACE_INDEX[face]
    exp_values = np.exp(logits - np.max(logits))
    weights3 = exp_values / np.sum(exp_values)
    weights = np.zeros(4)
    weights[[j for j in range(4) if j != idx]] = weights3
    return point_from_barycentric(weights)


def fixed_word_residual(params: np.ndarray, word: str) -> np.ndarray:
    start_face = word[0]
    p0 = point_from_face_logits(params[:3], start_face)
    d0 = unit(params[3:6])
    residuals: List[float] = []
    inward = float(LAMBDA_GRADS[FACE_INDEX[start_face]] @ d0)
    residuals.append(min(0.0, inward) * 10.0)

    point = p0.copy()
    direction = d0.copy()
    failed = False
    for next_face in word[1:] + word[:1]:
        if failed:
            residuals.extend([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
            continue

        target = FACE_INDEX[next_face]
        values = lambdas(point)
        derivs = LAMBDA_GRADS @ direction
        denom = derivs[target]
        if abs(float(denom)) < 1e-12:
            residuals.extend([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
            failed = True
            continue

        t_target = -values[target] / denom
        if t_target <= 1e-10:
            residuals.extend([10.0 + float(abs(t_target)), 10.0, 10.0, 10.0, 10.0, 10.0])
            failed = True
            continue

        early_hit_penalty = 0.0
        for idx in range(4):
            if idx == target or abs(float(derivs[idx])) < 1e-12:
                continue
            t = -values[idx] / derivs[idx]
            if 1e-10 < t < t_target:
                early_hit_penalty = max(early_hit_penalty, float((t_target - t) / max(t_target, 1e-12)))

        point = point + float(t_target) * direction
        hit_values = lambdas(point)
        residuals.append(0.0)
        residuals.append(early_hit_penalty)
        for idx, value in enumerate(hit_values):
            if idx == target:
                residuals.append(float(value))
            else:
                residuals.append(min(0.0, float(value)) * 10.0)
        direction = unit(reflect_direction(direction, next_face))

    residuals.extend((point - p0).tolist())
    residuals.extend((unit(direction) - d0).tolist())
    return np.array(residuals, dtype=float)


def refine_candidate(candidate: Candidate, max_nfev: int) -> Candidate:
    if least_squares is None:
        return candidate
    # The near-return shooting state is not retained, so start from a stable
    # interior point on the first face and let the fixed-word residual relax it.
    rng = np.random.default_rng(abs(hash(candidate.word)) % (2**32))
    start = sample_face_point(rng, candidate.word[0], edge_margin=0.12)
    direction = sample_outgoing_direction(rng, candidate.word[0])
    x0 = np.concatenate([barycentric_logits_for_face(start, candidate.word[0]), direction])
    result = least_squares(
        lambda x: fixed_word_residual(x, candidate.word),
        x0,
        max_nfev=max_nfev,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    residual = fixed_word_residual(result.x, candidate.word)
    refined_score = float(np.linalg.norm(residual[-6:]))
    candidate.refined_score = refined_score
    candidate.refined_point_error = float(np.linalg.norm(residual[-6:-3]))
    candidate.refined_direction_error = float(np.linalg.norm(residual[-3:]))
    return candidate


def canonical_word(word: str, equivalence: str) -> str:
    if exact is None:
        return word
    parsed = exact.parse_word(word)
    return exact.word_name(exact.canonical_by_mode(parsed, equivalence))


def primitive_word_text(word: str) -> bool:
    if exact is not None:
        return exact.primitive_word(exact.parse_word(word))
    n = len(word)
    for d in range(1, n):
        if n % d == 0 and word == word[:d] * (n // d):
            return False
    return True


def exactify_word(word: str, equivalence: str) -> Tuple[Optional[Dict[str, object]], str]:
    if exact is None:
        return None, "exact_verifier_unavailable"
    parsed = exact.parse_word(word)
    canon = exact.canonical_by_mode(parsed, equivalence)
    result = exact.verify_word_exact(canon)
    if result.orbit is None:
        return None, result.reason
    record = exact.orbit_to_record(result.orbit, equivalence=equivalence)
    record["provenance"] = "exploratory_exactified"
    record["discovery_method"] = "random_shoot_near_return"
    record["kind"] = "ordinary"
    return record, result.reason


def candidate_to_json(candidate: Candidate) -> Dict[str, object]:
    return {
        "word": candidate.word,
        "period": candidate.period,
        "score": candidate.score,
        "point_error": candidate.point_error,
        "direction_error": candidate.direction_error,
        "trial": candidate.trial,
        "start_index": candidate.start_index,
        "end_index": candidate.end_index,
        "refined_score": candidate.refined_score,
        "refined_point_error": candidate.refined_point_error,
        "refined_direction_error": candidate.refined_direction_error,
        "exact_reason": candidate.exact_reason,
    }


def run_search(args: argparse.Namespace) -> Dict[str, object]:
    rng = np.random.default_rng(args.seed)
    by_word: Dict[str, Candidate] = {}
    started = time.time()
    for trial in range(args.trials):
        for candidate in shoot_trial(
            rng,
            max_bounces=args.max_bounces,
            min_period=args.min_period,
            max_period=args.max_period,
            threshold=args.near_threshold,
            edge_margin=args.edge_margin,
            eps=args.eps,
            trial=trial,
        ):
            key = canonical_word(candidate.word, args.equivalence)
            old = by_word.get(key)
            if old is None or candidate.score < old.score:
                candidate.word = key
                by_word[key] = candidate
        if args.report_every and (trial + 1) % args.report_every == 0:
            print(f"[explore] trials={trial + 1} candidates={len(by_word)} elapsed={time.time() - started:.1f}s", flush=True)

    candidates = sorted(by_word.values(), key=lambda item: (item.score, item.period, item.word))
    if args.refine:
        for candidate in candidates[: args.refine]:
            refine_candidate(candidate, args.refine_max_nfev)
        candidates.sort(key=lambda item: (item.refined_score if item.refined_score is not None else item.score, item.period, item.word))

    records: Dict[str, Dict[str, object]] = {}
    for candidate in candidates[: args.exactify]:
        if not args.include_imprimitive and not primitive_word_text(candidate.word):
            candidate.exact_reason = "imprimitive_skip"
            continue
        record, reason = exactify_word(candidate.word, args.equivalence)
        candidate.exact_reason = reason
        if record is not None:
            record["exploratory_score"] = candidate.score
            if candidate.refined_score is not None:
                record["exploratory_refined_score"] = candidate.refined_score
            records[str(record["id"])] = record

    return {
        "schema": "spectral.tetra_billiards_exploratory.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "description": "Non-exhaustive random billiard shooting with near-return detection; exactified rows are verified separately.",
        "parameters": {
            "trials": args.trials,
            "max_bounces": args.max_bounces,
            "min_period": args.min_period,
            "max_period": args.max_period,
            "near_threshold": args.near_threshold,
            "edge_margin": args.edge_margin,
            "seed": args.seed,
            "equivalence": args.equivalence,
            "exact_verifier_available": exact is not None,
        },
        "summary": {
            "near_return_candidates": len(candidates),
            "exactified_orbits": len(records),
            "elapsed_seconds": time.time() - started,
        },
        "candidates": [candidate_to_json(candidate) for candidate in candidates[: args.keep_candidates]],
        "orbits": sorted(records.values(), key=lambda row: (int(row["period"]), str(row["word"]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory non-exhaustive search for closed regular-tetrahedron billiards.")
    parser.add_argument("--trials", type=int, default=2000, help="Random shooting trials.")
    parser.add_argument("--max-bounces", type=int, default=90, help="Maximum bounces per random trajectory.")
    parser.add_argument("--min-period", type=int, default=4, help="Minimum near-return period to report.")
    parser.add_argument("--max-period", type=int, default=80, help="Maximum near-return period to report.")
    parser.add_argument("--near-threshold", type=float, default=0.08, help="Phase-space near-return threshold.")
    parser.add_argument("--edge-margin", type=float, default=0.035, help="Initial face sampling margin from edges.")
    parser.add_argument("--eps", type=float, default=1e-10, help="Numerical hit tolerance.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--equivalence", choices=["none", "cyclic", "full"], default="full", help="Canonicalization used before exactification.")
    parser.add_argument("--include-imprimitive", action="store_true", help="Exactify repeated words too. Defaults to primitive words only.")
    parser.add_argument("--refine", type=int, default=25, help="Number of best candidates to refine numerically.")
    parser.add_argument("--refine-max-nfev", type=int, default=1200, help="Least-squares budget per refined candidate.")
    parser.add_argument("--exactify", type=int, default=200, help="Number of best candidates to pass to the exact verifier.")
    parser.add_argument("--keep-candidates", type=int, default=200, help="Candidate rows to keep in the output JSON.")
    parser.add_argument("--report-every", type=int, default=200, help="Progress print interval in trials; 0 disables.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "tetra_exploratory_paths.json", help="Output JSON path.")
    args = parser.parse_args()

    payload = run_search(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"wrote {args.out} with {payload['summary']['near_return_candidates']} candidates "
        f"and {payload['summary']['exactified_orbits']} exactified orbit(s)"
    )


if __name__ == "__main__":
    main()
