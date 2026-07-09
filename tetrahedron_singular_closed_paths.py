from __future__ import annotations

"""
Exhaustive singular normal-cone billiard search for the regular tetrahedron.

The ordinary tetrahedron search unfolds face words.  That is not enough for
singular paths: at an edge or vertex the reflection normal can be any vector in
the normal cone.  This script instead enumerates cyclic words in boundary
strata, minimizes the closed polygonal length for each word, and keeps exactly
the relative-interior minimizers whose velocity jump lies in the tetrahedron
normal cone.

Tokens are active-face sets in barycentric coordinates.  For example:

    A     means the face opposite vertex A;
    AB    means the edge F_A cap F_B;
    ABC   means the vertex F_A cap F_B cap F_C, namely vertex D.

The tetrahedron vertices are A=(1,0,0), B=(0,1,0), C=(0,0,1), D=(1,1,1).
"""

import argparse
import itertools
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import sympy as sp


VecF = Tuple[float, float, float]
VecQ = Tuple[Fraction, Fraction, Fraction]
MaskWord = Tuple[int, ...]


REPO_ROOT = Path(__file__).resolve().parent
FACE_LABELS = ("A", "B", "C", "D")

VERTICES_FLOAT: Tuple[VecF, ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
)
VERTICES_EXACT: Tuple[VecQ, ...] = (
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(0), Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(0), Fraction(1)),
    (Fraction(1), Fraction(1), Fraction(1)),
)

# Twice the outward face normals for barycentric coordinates A, B, C, D.
# Interior points have positive barycentric coordinate, so outward is -grad.
OUTWARD_NORMALS: Tuple[Tuple[int, int, int], ...] = (
    (-1, 1, 1),
    (1, -1, 1),
    (1, 1, -1),
    (-1, -1, -1),
)

ALL_FACE_PERMUTATIONS: Tuple[Tuple[int, ...], ...] = tuple(itertools.permutations(range(4)))
ALL_STRATUM_MASKS: Tuple[int, ...] = tuple(
    sum(1 << i for i in combo)
    for size in (1, 2, 3)
    for combo in itertools.combinations(range(4), size)
)


@dataclass(frozen=True)
class FloatCandidate:
    word: MaskWord
    barycentric: Tuple[Tuple[float, float, float, float], ...]
    length: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class ExactCandidate:
    word: MaskWord
    barycentric: Tuple[Tuple[Fraction, Fraction, Fraction, Fraction], ...]
    length_expr: sp.Expr
    length_numeric: float
    family_dimension: int


def popcount(mask: int) -> int:
    return bin(mask).count("1")


def mask_label(mask: int) -> str:
    return "".join(FACE_LABELS[i] for i in range(4) if mask & (1 << i))


def word_label(word: Sequence[int]) -> str:
    return " ".join(mask_label(mask) for mask in word)


def token_id(mask: int) -> str:
    return mask_label(mask)


def word_id(word: Sequence[int]) -> str:
    return "_".join(token_id(mask) for mask in word)


def frac_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def decimal_text(value: float, digits: int = 15) -> str:
    if abs(value) < 5e-16:
        return "0"
    return f"{value:.{digits}g}"


def bary_to_xyz_float(row: Sequence[float]) -> VecF:
    return (row[0] + row[3], row[1] + row[3], row[2] + row[3])


def bary_to_xyz_exact(row: Sequence[Fraction]) -> VecQ:
    return (row[0] + row[3], row[1] + row[3], row[2] + row[3])


def xyz_to_bary_float(point: VecF) -> Tuple[float, float, float, float]:
    x, y, z = point
    return (
        (x - y - z + 1.0) / 2.0,
        (-x + y - z + 1.0) / 2.0,
        (-x - y + z + 1.0) / 2.0,
        (x + y + z - 1.0) / 2.0,
    )


def add_float(a: VecF, b: VecF) -> VecF:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub_float(a: VecF, b: VecF) -> VecF:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale_float(c: float, a: VecF) -> VecF:
    return (c * a[0], c * a[1], c * a[2])


def dot_float(a: VecF, b: VecF) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm_float(a: VecF) -> float:
    return math.sqrt(max(0.0, dot_float(a, a)))


def dist_float(a: VecF, b: VecF) -> float:
    return norm_float(sub_float(a, b))


def path_length_float(points: Sequence[VecF]) -> float:
    return sum(dist_float(points[i], points[(i + 1) % len(points)]) for i in range(len(points)))


def centroid_bary(mask: int) -> Tuple[float, float, float, float]:
    free = [i for i in range(4) if not (mask & (1 << i))]
    row = [0.0, 0.0, 0.0, 0.0]
    for index in free:
        row[index] = 1.0 / len(free)
    return tuple(row)  # type: ignore[return-value]


def edge_minimizer(mask: int, previous: VecF, next_point: VecF) -> Tuple[float, float, float, float]:
    free = [i for i in range(4) if not (mask & (1 << i))]
    if len(free) != 2:
        raise ValueError(f"{mask_label(mask)} is not an edge stratum")

    left, right = free
    a = VERTICES_FLOAT[left]
    b = VERTICES_FLOAT[right]
    edge = sub_float(b, a)

    def point_at(t: float) -> VecF:
        return add_float(a, scale_float(t, edge))

    def derivative(t: float) -> float:
        point = point_at(t)
        total = 0.0
        for endpoint in (previous, next_point):
            delta = sub_float(point, endpoint)
            length = norm_float(delta)
            if length > 1e-15:
                total += dot_float(edge, delta) / length
        return total

    d0 = derivative(0.0)
    d1 = derivative(1.0)
    if d0 >= 0.0:
        t = 0.0
    elif d1 <= 0.0:
        t = 1.0
    else:
        lo = 0.0
        hi = 1.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if derivative(mid) > 0.0:
                hi = mid
            else:
                lo = mid
        t = (lo + hi) / 2.0

    row = [0.0, 0.0, 0.0, 0.0]
    row[left] = 1.0 - t
    row[right] = t
    return tuple(row)  # type: ignore[return-value]


def face_minimizer(mask: int, previous: VecF, next_point: VecF) -> Tuple[float, float, float, float]:
    active = [i for i in range(4) if mask & (1 << i)]
    if len(active) != 1:
        raise ValueError(f"{mask_label(mask)} is not a face stratum")
    face = active[0]

    candidates: List[Tuple[float, float, float, float]] = []

    # Reflect next_point across the active face plane lambda_face=0.  The
    # minimizer over the full plane is where previous--reflected(next) crosses
    # the plane.  If that point misses the triangle, fall back to its edges.
    lambda_next = xyz_to_bary_float(next_point)[face]
    grad = tuple(-OUTWARD_NORMALS[face][axis] / 2.0 for axis in range(3))
    grad_norm_squared = 3.0 / 4.0
    reflected_next = tuple(
        next_point[axis] - 2.0 * lambda_next / grad_norm_squared * grad[axis]
        for axis in range(3)
    )
    lambda_previous = xyz_to_bary_float(previous)[face]
    lambda_reflected = xyz_to_bary_float(reflected_next)[face]
    denominator = lambda_reflected - lambda_previous
    if abs(denominator) > 1e-14:
        t = -lambda_previous / denominator
        point = add_float(previous, scale_float(t, sub_float(reflected_next, previous)))
        row = xyz_to_bary_float(point)
        if -1e-9 <= t <= 1.0 + 1e-9 and all(value >= -1e-9 for value in row):
            clipped = [max(0.0, value) if i != face else 0.0 for i, value in enumerate(row)]
            total = sum(clipped)
            if total > 0.0:
                candidates.append(tuple(value / total for value in clipped))  # type: ignore[arg-type]

    free = [i for i in range(4) if i != face]
    for left, right in itertools.combinations(free, 2):
        edge_mask = sum(1 << j for j in range(4) if j not in (left, right))
        candidates.append(edge_minimizer(edge_mask, previous, next_point))
    for vertex in free:
        row = [0.0, 0.0, 0.0, 0.0]
        row[vertex] = 1.0
        candidates.append(tuple(row))  # type: ignore[arg-type]

    return min(
        candidates,
        key=lambda row: dist_float(bary_to_xyz_float(row), previous)
        + dist_float(bary_to_xyz_float(row), next_point),
    )


def stratum_minimizer(mask: int, previous: VecF, next_point: VecF) -> Tuple[float, float, float, float]:
    codimension = popcount(mask)
    if codimension == 3:
        return centroid_bary(mask)
    if codimension == 2:
        return edge_minimizer(mask, previous, next_point)
    if codimension == 1:
        return face_minimizer(mask, previous, next_point)
    raise ValueError(f"unsupported stratum mask {mask}")


def solve_word_float(
    word: MaskWord,
    max_iterations: int = 5000,
    tolerance: float = 1e-12,
) -> FloatCandidate:
    rows = [centroid_bary(mask) for mask in word]
    previous_length = float("inf")
    converged = False
    iteration = 0

    for iteration in range(max_iterations):
        max_change = 0.0
        for i, mask in enumerate(word):
            points = [bary_to_xyz_float(row) for row in rows]
            updated = stratum_minimizer(mask, points[(i - 1) % len(word)], points[(i + 1) % len(word)])
            max_change = max(max_change, max(abs(updated[j] - rows[i][j]) for j in range(4)))
            rows[i] = updated

        points = [bary_to_xyz_float(row) for row in rows]
        current_length = path_length_float(points)
        if abs(previous_length - current_length) < tolerance and max_change < 10.0 * tolerance:
            converged = True
            break
        previous_length = current_length

    points = [bary_to_xyz_float(row) for row in rows]
    return FloatCandidate(
        word=word,
        barycentric=tuple(rows),
        length=path_length_float(points),
        iterations=iteration + 1,
        converged=converged,
    )


def solve_linear_float(
    matrix: Sequence[Sequence[float]],
    rhs: Sequence[float],
) -> Optional[List[float]]:
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(rows[row][col]))
        if abs(rows[pivot][col]) < 1e-11:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        pivot_value = rows[col][col]
        for j in range(col, n + 1):
            rows[col][j] /= pivot_value
        for r in range(n):
            if r == col:
                continue
            factor = rows[r][col]
            for j in range(col, n + 1):
                rows[r][j] -= factor * rows[col][j]
    return [rows[i][n] for i in range(n)]


def normal_coefficients_float(mask: int, vector: VecF) -> Optional[List[float]]:
    active = [i for i in range(4) if mask & (1 << i)]
    codimension = len(active)
    best: Optional[Tuple[float, List[float]]] = None
    for rows in itertools.combinations(range(3), codimension):
        matrix = [[float(OUTWARD_NORMALS[face][axis]) for face in active] for axis in rows]
        rhs = [vector[axis] for axis in rows]
        coefficients = solve_linear_float(matrix, rhs)
        if coefficients is None:
            continue
        reconstructed = tuple(
            sum(coefficients[j] * OUTWARD_NORMALS[active[j]][axis] for j in range(codimension))
            for axis in range(3)
        )
        error = norm_float(sub_float(reconstructed, vector))
        if best is None or error < best[0]:
            best = (error, coefficients)
    if best is None or best[0] > 2e-5:
        return None
    return best[1]


def float_candidate_valid(
    candidate: FloatCandidate,
    relative_tol: float = 1e-6,
    cone_tol: float = 2e-6,
) -> bool:
    if candidate.length <= 1e-8:
        return False
    word = candidate.word
    rows = candidate.barycentric
    points = [bary_to_xyz_float(row) for row in rows]
    period = len(word)

    for i, mask in enumerate(word):
        for face in range(4):
            if mask & (1 << face):
                if abs(rows[i][face]) > relative_tol:
                    return False
            elif rows[i][face] < relative_tol:
                return False

        previous = (i - 1) % period
        previous_length = dist_float(points[i], points[previous])
        next_length = dist_float(points[(i + 1) % period], points[i])
        if previous_length <= 1e-10 or next_length <= 1e-10:
            return False

        incoming = tuple((points[i][axis] - points[previous][axis]) / previous_length for axis in range(3))
        outgoing = tuple((points[(i + 1) % period][axis] - points[i][axis]) / next_length for axis in range(3))
        jump = sub_float(incoming, outgoing)
        coefficients = normal_coefficients_float(mask, jump)
        if coefficients is None or min(coefficients) < -cone_tol:
            return False

    return True


def rationalize_row(
    mask: int,
    row: Sequence[float],
    max_denominator: int,
) -> Tuple[Fraction, Fraction, Fraction, Fraction]:
    free = [i for i in range(4) if not (mask & (1 << i))]
    out = [Fraction(0), Fraction(0), Fraction(0), Fraction(0)]
    if len(free) == 1:
        out[free[0]] = Fraction(1)
    elif len(free) == 2:
        first = Fraction(row[free[0]]).limit_denominator(max_denominator)
        if first <= 0:
            first = Fraction(1, max_denominator)
        if first >= 1:
            first = Fraction(max_denominator - 1, max_denominator)
        out[free[0]] = first
        out[free[1]] = 1 - first
    elif len(free) == 3:
        first = Fraction(row[free[0]]).limit_denominator(max_denominator)
        second = Fraction(row[free[1]]).limit_denominator(max_denominator)
        third = 1 - first - second
        if first <= 0 or second <= 0 or third <= 0:
            limited = [max(0.0, row[index]) for index in free]
            total = sum(limited)
            if total <= 0:
                limited = [1.0, 1.0, 1.0]
                total = 3.0
            fractions = [
                Fraction(value / total).limit_denominator(max_denominator)
                for value in limited[:2]
            ]
            first, second = fractions
            third = 1 - first - second
        out[free[0]] = first
        out[free[1]] = second
        out[free[2]] = third
    else:
        raise ValueError(f"invalid free coordinate count for {mask_label(mask)}")
    return tuple(out)  # type: ignore[return-value]


def squared_distance_exact(a: VecQ, b: VecQ) -> Fraction:
    return sum((a[i] - b[i]) ** 2 for i in range(3))


def fraction_to_sympy(value: Fraction) -> sp.Rational:
    return sp.Rational(value.numerator, value.denominator)


def exact_normal_coefficients(
    mask: int,
    jump: Sequence[sp.Expr],
) -> Optional[List[sp.Expr]]:
    active = [i for i in range(4) if mask & (1 << i)]
    matrix = sp.Matrix([[OUTWARD_NORMALS[face][axis] for face in active] for axis in range(3)])
    rhs = sp.Matrix(jump)
    try:
        solution, params = matrix.gauss_jordan_solve(rhs)
    except ValueError:
        return None
    if params.rows:
        return None
    reconstructed = matrix * solution
    residual = [sp.simplify(reconstructed[i] - rhs[i]) for i in range(3)]
    if any(value != 0 for value in residual):
        return None
    return [sp.simplify(solution[i]) for i in range(solution.rows)]


def verify_exact_candidate(
    word: MaskWord,
    rows: Sequence[Sequence[Fraction]],
    positivity_tol: float = 1e-9,
) -> Optional[sp.Expr]:
    points = [bary_to_xyz_exact(row) for row in rows]
    period = len(word)

    for i, mask in enumerate(word):
        for face in range(4):
            if mask & (1 << face):
                if rows[i][face] != 0:
                    return None
            elif rows[i][face] <= 0:
                return None
        if word[i] & word[(i + 1) % period]:
            return None

    leg_squares = [
        squared_distance_exact(points[i], points[(i + 1) % period])
        for i in range(period)
    ]
    if any(value <= 0 for value in leg_squares):
        return None

    for i, mask in enumerate(word):
        previous = (i - 1) % period
        incoming = [
            fraction_to_sympy(points[i][axis] - points[previous][axis])
            / sp.sqrt(fraction_to_sympy(leg_squares[previous]))
            for axis in range(3)
        ]
        outgoing = [
            fraction_to_sympy(points[(i + 1) % period][axis] - points[i][axis])
            / sp.sqrt(fraction_to_sympy(leg_squares[i]))
            for axis in range(3)
        ]
        jump = [sp.simplify(incoming[axis] - outgoing[axis]) for axis in range(3)]
        coefficients = exact_normal_coefficients(mask, jump)
        if coefficients is None:
            return None
        if min(float(sp.N(value, 30)) for value in coefficients) < -positivity_tol:
            return None

    return sp.simplify(sum(sp.sqrt(fraction_to_sympy(value)) for value in leg_squares))


def exactify_candidate(
    candidate: FloatCandidate,
    max_denominator: int,
) -> Optional[ExactCandidate]:
    rows = tuple(
        rationalize_row(mask, row, max_denominator)
        for mask, row in zip(candidate.word, candidate.barycentric)
    )
    length_expr = verify_exact_candidate(candidate.word, rows)
    if length_expr is None:
        return None
    return ExactCandidate(
        word=candidate.word,
        barycentric=rows,
        length_expr=length_expr,
        length_numeric=float(sp.N(length_expr, 30)),
        family_dimension=family_dimension(candidate.word, rows),
    )


def free_parameters_from_rows(
    word: MaskWord,
    rows: Sequence[Sequence[Fraction]],
) -> Tuple[List[Tuple[int, List[int]]], List[float]]:
    specs: List[Tuple[int, List[int]]] = []
    values: List[float] = []
    for i, mask in enumerate(word):
        free = [face for face in range(4) if not (mask & (1 << face))]
        if len(free) >= 2:
            specs.append((i, free))
            values.extend(float(rows[i][face]) for face in free[:-1])
    return specs, values


def rows_from_parameters(
    word: MaskWord,
    specs: Sequence[Tuple[int, List[int]]],
    values: Sequence[float],
) -> List[Tuple[float, float, float, float]]:
    rows = [list(centroid_bary(mask)) for mask in word]
    cursor = 0
    spec_by_index = {index: free for index, free in specs}
    for i, mask in enumerate(word):
        free = [face for face in range(4) if not (mask & (1 << face))]
        rows[i] = [0.0, 0.0, 0.0, 0.0]
        if len(free) == 1:
            rows[i][free[0]] = 1.0
            continue
        spec_free = spec_by_index[i]
        used = 0.0
        for face in spec_free[:-1]:
            value = values[cursor]
            cursor += 1
            rows[i][face] = value
            used += value
        rows[i][spec_free[-1]] = 1.0 - used
    return [tuple(row) for row in rows]  # type: ignore[return-value]


def length_from_parameters(
    word: MaskWord,
    specs: Sequence[Tuple[int, List[int]]],
    values: Sequence[float],
) -> float:
    rows = rows_from_parameters(word, specs, values)
    if any(min(row) < -1e-5 for row in rows):
        return 1e6 + sum(abs(min(0.0, value)) for row in rows for value in row)
    return path_length_float([bary_to_xyz_float(row) for row in rows])


def matrix_rank_float(matrix: List[List[float]], tolerance: float = 1e-5) -> int:
    if not matrix or not matrix[0]:
        return 0
    rows = [row[:] for row in matrix]
    height = len(rows)
    width = len(rows[0])
    rank = 0
    scale = max(1.0, max(abs(value) for row in rows for value in row))
    threshold = tolerance * scale
    for col in range(width):
        pivot = max(range(rank, height), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) <= threshold:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][col]
        for j in range(col, width):
            rows[rank][j] /= pivot_value
        for r in range(height):
            if r == rank:
                continue
            factor = rows[r][col]
            for j in range(col, width):
                rows[r][j] -= factor * rows[rank][j]
        rank += 1
        if rank == height:
            break
    return rank


def family_dimension(word: MaskWord, rows: Sequence[Sequence[Fraction]]) -> int:
    specs, values = free_parameters_from_rows(word, rows)
    return family_dimension_from_parameters(word, specs, values)


def free_parameters_from_float_rows(
    word: MaskWord,
    rows: Sequence[Sequence[float]],
) -> Tuple[List[Tuple[int, List[int]]], List[float]]:
    specs: List[Tuple[int, List[int]]] = []
    values: List[float] = []
    for i, mask in enumerate(word):
        free = [face for face in range(4) if not (mask & (1 << face))]
        if len(free) >= 2:
            specs.append((i, free))
            values.extend(float(rows[i][face]) for face in free[:-1])
    return specs, values


def family_dimension_float(word: MaskWord, rows: Sequence[Sequence[float]]) -> int:
    specs, values = free_parameters_from_float_rows(word, rows)
    return family_dimension_from_parameters(word, specs, values)


def family_dimension_from_parameters(
    word: MaskWord,
    specs: Sequence[Tuple[int, List[int]]],
    values: Sequence[float],
) -> int:
    n = len(values)
    if n == 0:
        return 0

    step = 1e-5
    hessian = [[0.0 for _ in range(n)] for _ in range(n)]
    base = length_from_parameters(word, specs, values)
    for i in range(n):
        plus = values[:]
        minus = values[:]
        plus[i] += step
        minus[i] -= step
        hessian[i][i] = (
            length_from_parameters(word, specs, plus)
            - 2.0 * base
            + length_from_parameters(word, specs, minus)
        ) / (step * step)
        for j in range(i + 1, n):
            pp = values[:]
            pm = values[:]
            mp = values[:]
            mm = values[:]
            pp[i] += step
            pp[j] += step
            pm[i] += step
            pm[j] -= step
            mp[i] -= step
            mp[j] += step
            mm[i] -= step
            mm[j] -= step
            value = (
                length_from_parameters(word, specs, pp)
                - length_from_parameters(word, specs, pm)
                - length_from_parameters(word, specs, mp)
                + length_from_parameters(word, specs, mm)
            ) / (4.0 * step * step)
            hessian[i][j] = value
            hessian[j][i] = value

    rank = matrix_rank_float(hessian, tolerance=5e-5)
    return max(0, n - rank)


def primitive_word(word: MaskWord) -> bool:
    period = len(word)
    for divisor in range(1, period):
        if period % divisor == 0 and word == word[:divisor] * (period // divisor):
            return False
    return True


def rotations(word: Sequence[int]) -> Iterator[MaskWord]:
    w = tuple(word)
    for shift in range(len(w)):
        yield w[shift:] + w[:shift]


def permute_mask(mask: int, permutation: Sequence[int]) -> int:
    out = 0
    for face in range(4):
        if mask & (1 << face):
            out |= 1 << permutation[face]
    return out


def permute_word(word: Sequence[int], permutation: Sequence[int]) -> MaskWord:
    return tuple(permute_mask(mask, permutation) for mask in word)


def canonical_stratum_word(word: Sequence[int]) -> MaskWord:
    variants: List[MaskWord] = []
    for permutation in ALL_FACE_PERMUTATIONS:
        relabeled = permute_word(word, permutation)
        variants.extend(rotations(relabeled))
        variants.extend(rotations(tuple(reversed(relabeled))))
    return min(variants)


def canonical_cyclic_stratum_word(word: Sequence[int]) -> MaskWord:
    variants = list(rotations(word))
    variants.extend(rotations(tuple(reversed(tuple(word)))))
    return min(variants)


def candidate_words(period: int) -> Iterator[MaskWord]:
    seen: set[MaskWord] = set()

    def rec(prefix: List[int]) -> Iterator[MaskWord]:
        if len(prefix) == period:
            if prefix[-1] & prefix[0]:
                return
            word = tuple(prefix)
            if not any(popcount(mask) > 1 for mask in word):
                return
            if not primitive_word(word):
                return
            canonical = canonical_stratum_word(word)
            if canonical in seen:
                return
            seen.add(canonical)
            yield canonical
            return

        for mask in ALL_STRATUM_MASKS:
            if not (prefix[-1] & mask):
                yield from rec(prefix + [mask])

    for mask in ALL_STRATUM_MASKS:
        yield from rec([mask])


def lcm(values: Iterable[int]) -> int:
    out = 1
    for value in values:
        out = out * value // math.gcd(out, value)
    return out


def canonical_matrix_key(rows: Sequence[Sequence[Fraction]]) -> Tuple[Tuple[str, ...], ...]:
    return tuple(tuple(frac_text(value) for value in row) for row in rows)


def cyclic_matrix_keys(rows: Sequence[Sequence[Fraction]]) -> List[Tuple[Tuple[str, ...], ...]]:
    matrix = [tuple(row) for row in rows]
    keys = []
    for source in (matrix, list(reversed(matrix))):
        for shift in range(len(source)):
            keys.append(canonical_matrix_key(source[shift:] + source[:shift]))
    return keys


def copy_count(rows: Sequence[Sequence[Fraction]]) -> int:
    seen = set()
    for permutation in ALL_FACE_PERMUTATIONS:
        relabeled = [
            tuple(row[permutation[i]] for i in range(4))
            for row in rows
        ]
        seen.add(min(cyclic_matrix_keys(relabeled)))
    return len(seen)


def stratum_word_copy_count(word: Sequence[int]) -> int:
    seen = set()
    for permutation in ALL_FACE_PERMUTATIONS:
        seen.add(canonical_cyclic_stratum_word(permute_word(word, permutation)))
    return len(seen)


def center_metrics(points: Sequence[Sequence[Fraction]]) -> Dict[str, object]:
    center = [
        sum(point[axis] for point in points) / len(points)
        for axis in range(3)
    ]
    tetra_centroid = (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2))
    distance_squared = sum((center[axis] - tetra_centroid[axis]) ** 2 for axis in range(3))
    return {
        "path_center_xyz_exact": [frac_text(value) for value in center],
        "path_center_xyz": [float(value) for value in center],
        "centroid_distance_squared_exact": frac_text(distance_squared),
        "centroid_distance_numeric": math.sqrt(float(distance_squared)),
    }


def record_from_candidate(candidate: ExactCandidate) -> Dict[str, object]:
    denominator = lcm(value.denominator for row in candidate.barycentric for value in row)
    barycentric_points = [
        [str(value.numerator * (denominator // value.denominator)) for value in row]
        for row in candidate.barycentric
    ]
    barycentric_exact = [[frac_text(value) for value in row] for row in candidate.barycentric]
    points_exact = [bary_to_xyz_exact(row) for row in candidate.barycentric]
    points_xyz_exact = [[frac_text(coord) for coord in point] for point in points_exact]
    points_xyz = [[float(coord) for coord in point] for point in points_exact]
    positive = [value for row in candidate.barycentric for value in row if value > 0]
    height = max([denominator] + [abs(int(value)) for row in barycentric_points for value in row])
    copies = copy_count(candidate.barycentric)

    record: Dict[str, object] = {
        "id": f"s{len(candidate.word):02d}_{word_id(candidate.word)}",
        "kind": "singular_normal_cone",
        "singular": True,
        "period": len(candidate.word),
        "word": word_label(candidate.word),
        "stratum_word": [mask_label(mask) for mask in candidate.word],
        "length_exact": str(candidate.length_expr),
        "length_numeric": candidate.length_numeric,
        "barycentric_denominator": str(denominator),
        "barycentric_points": barycentric_points,
        "barycentric_exact": barycentric_exact,
        "points_xyz_exact": points_xyz_exact,
        "points_xyz": points_xyz,
        "height": str(height),
        "boundary_margin": frac_text(min(positive)) if positive else "",
        "coordinate_kind": "exact_rational",
        "provenance": "exhaustive_singular_normal_cone",
        "discovery_method": "canonical stratum-word convex search",
    }
    if candidate.family_dimension > 0:
        record["family_dimension"] = candidate.family_dimension
        record["family_copy_count"] = copies
        record["representative_of_family"] = True
    else:
        record["copy_count"] = copies
    record.update(center_metrics(points_exact))
    return record


def record_from_float_candidate(candidate: FloatCandidate) -> Dict[str, object]:
    rows = candidate.barycentric
    points = [bary_to_xyz_float(row) for row in rows]
    positive = [value for row in rows for value in row if value > 1e-14]
    family_dim = family_dimension_float(candidate.word, rows)
    copies = stratum_word_copy_count(candidate.word)
    record: Dict[str, object] = {
        "id": f"s{len(candidate.word):02d}_{word_id(candidate.word)}",
        "kind": "singular_normal_cone",
        "singular": True,
        "period": len(candidate.word),
        "word": word_label(candidate.word),
        "stratum_word": [mask_label(mask) for mask in candidate.word],
        "length_exact": f"~{decimal_text(candidate.length, 13)}",
        "length_numeric": candidate.length,
        "barycentric_denominator": "1",
        "barycentric_points": [
            [decimal_text(value) for value in row]
            for row in rows
        ],
        "barycentric_exact": [
            [decimal_text(value) for value in row]
            for row in rows
        ],
        "points_xyz_exact": [
            [decimal_text(coord) for coord in point]
            for point in points
        ],
        "points_xyz": [[float(coord) for coord in point] for point in points],
        "height": "numeric",
        "boundary_margin": decimal_text(min(positive)) if positive else "",
        "coordinate_kind": "numeric",
        "numeric_residual": "normal-cone checked to floating tolerance",
        "provenance": "exhaustive_singular_normal_cone_numeric",
        "discovery_method": "canonical stratum-word convex search",
    }
    if family_dim > 0:
        record["family_dimension"] = family_dim
        record["family_copy_count"] = copies
        record["representative_of_family"] = True
    else:
        record["copy_count"] = copies

    center = [
        sum(point[axis] for point in points) / len(points)
        for axis in range(3)
    ]
    record["path_center_xyz_exact"] = [decimal_text(value) for value in center]
    record["path_center_xyz"] = center
    record["centroid_distance_squared_exact"] = decimal_text(
        sum((center[axis] - 0.5) ** 2 for axis in range(3))
    )
    record["centroid_distance_numeric"] = math.sqrt(
        sum((center[axis] - 0.5) ** 2 for axis in range(3))
    )
    return record


def build_inventory(
    max_period: int,
    max_denominator: int,
    quiet: bool = False,
) -> Dict[str, object]:
    records: Dict[str, Dict[str, object]] = {}
    candidate_counts: Dict[str, int] = {}
    rejected_after_float = 0
    numeric_after_exact = 0

    for period in range(2, max_period + 1):
        period_candidates = list(candidate_words(period))
        candidate_counts[str(period)] = len(period_candidates)
        hits = 0
        if not quiet:
            print(f"[period {period}] canonical candidates={len(period_candidates)}", flush=True)
        for word in period_candidates:
            float_candidate = solve_word_float(word)
            if not float_candidate.converged or not float_candidate_valid(float_candidate):
                rejected_after_float += 1
                continue
            exact_candidate = exactify_candidate(float_candidate, max_denominator=max_denominator)
            if exact_candidate is None:
                record = record_from_float_candidate(float_candidate)
                records[str(record["id"])] = record
                numeric_after_exact += 1
                hits += 1
                continue
            record = record_from_candidate(exact_candidate)
            records[str(record["id"])] = record
            hits += 1
        if not quiet:
            print(f"[period {period}] singular hits={hits}", flush=True)

    orbits = sorted(records.values(), key=lambda row: (int(row["period"]), float(row["length_numeric"]), str(row["word"])))
    period_counts = Counter(int(record["period"]) for record in orbits)
    family_count = sum(1 for record in orbits if int(record.get("family_dimension", 0)) > 0)
    exact_count = sum(1 for record in orbits if record.get("coordinate_kind") == "exact_rational")
    numeric_count = len(orbits) - exact_count
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return {
        "schema": "spectral.tetra_singular_normal_cones.v2",
        "generated_at": generated_at,
        "title": "Singular normal-cone billiard paths in the regular tetrahedron",
        "source_algorithm": "tetrahedron_singular_closed_paths.py",
        "source_notebook": "https://observablehq.com/@liuyao12/billiard-in-tetrahedra",
        "tetrahedron": {
            "vertex_order": list(FACE_LABELS),
            "vertices": {name: list(VERTICES_FLOAT[i]) for i, name in enumerate(FACE_LABELS)},
            "face_convention": "A, B, C, D denote the faces opposite the corresponding vertices.",
        },
        "notation": {
            "A": "face hit on F_A",
            "AB": "edge hit on F_A cap F_B",
            "ABC": "vertex hit on F_A cap F_B cap F_C",
        },
        "summary": {
            "total_orbits": len(orbits),
            "max_period": max((int(record["period"]) for record in orbits), default=0),
            "exhaustive_checked_through": max_period,
            "period_counts": {str(period): period_counts[period] for period in sorted(period_counts)},
            "canonical_candidates_by_period": candidate_counts,
            "family_orbits": family_count,
            "exact_rational_orbits": exact_count,
            "numeric_orbits": numeric_count,
            "rejected_after_float": rejected_after_float,
            "numeric_after_exact": numeric_after_exact,
            "equivalence": "cyclic, reversal, and full tetrahedral relabeling",
            "method": (
                "Enumerate primitive cyclic active-face stratum words with disjoint "
                "adjacent active sets; minimize polygonal length over the product of "
                "closed strata; retain relative-interior minimizers satisfying the "
                "normal-cone velocity-jump condition."
            ),
        },
        "orbits": orbits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find singular normal-cone closed billiard paths in the regular tetrahedron."
    )
    parser.add_argument("--max-period", type=int, default=7)
    parser.add_argument("--max-denominator", type=int, default=10000)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs" / "data" / "tetra" / "singular_normal_cones.json",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.max_period < 2:
        raise SystemExit("--max-period must be at least 2")
    if args.max_denominator < 2:
        raise SystemExit("--max-denominator must be at least 2")

    inventory = build_inventory(
        max_period=args.max_period,
        max_denominator=args.max_denominator,
        quiet=args.quiet,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out} with {inventory['summary']['total_orbits']} singular orbit classes "
        f"through period {args.max_period}"
    )


if __name__ == "__main__":
    main()
