from __future__ import annotations

"""
Exact exhaustive search for closed billiard paths in the regular tetrahedron.

This script is intentionally separate from the geometry-first dashboard.  The
dashboard is useful for exploration, but its prefix filter samples directions on
S^2.  Here the contract is stricter:

* final orbit verification is exact rational arithmetic;
* repeated adjacent faces are never generated;
* by default, no nontrivial geometric forbidden-subword lemma is assumed;
  optional lemmas can be enabled explicitly after they have a proof/certificate;
* output is one representative per requested equivalence class.

Face labels use the convention from the older scripts:
    D = face ABC, A = face BCD, B = face ACD, C = face ABD.
"""

import argparse
import itertools
import json
import math
import multiprocessing as mp
import mmap
import sys
import threading
import time
from array import array
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Callable, DefaultDict, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import sympy as sp


VecI = Tuple[int, int, int]
MatI = Tuple[VecI, VecI, VecI]
VecQ = Tuple[Fraction, Fraction, Fraction]
TriangleQ = Tuple[VecQ, VecQ, VecQ]
InequalityId = Tuple[str, int]


# Internal face order: D, A, B, C.
FACE_NAMES = ("D", "A", "B", "C")
NAME_TO_FACE = {name: i for i, name in enumerate(FACE_NAMES)}
CANONICAL_FORBIDDEN_CACHE_SIZE = 65536

FACE_NORMALS: Tuple[VecI, ...] = (
    (1, 1, 1),
    (1, -1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
)
FACE_CONSTANTS = (1, -1, -1, -1)
FACE_OFFSETS = (-1, 1, 1, 1)

ALL_FACE_PERMUTATIONS: Tuple[Tuple[int, ...], ...] = tuple(itertools.permutations(range(4)))
IDENTITY_MATRIX: MatI = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
ZERO_VECTOR: VecI = (0, 0, 0)


def _reflection_data() -> Tuple[Tuple[MatI, VecI], ...]:
    out: List[Tuple[MatI, VecI]] = []
    for n, c in zip(FACE_NORMALS, FACE_CONSTANTS):
        mat: MatI = tuple(
            tuple((3 if i == j else 0) - 2 * n[i] * n[j] for j in range(3))
            for i in range(3)
        )  # type: ignore[assignment]
        shift: VecI = tuple(2 * c * n[i] for i in range(3))  # type: ignore[assignment]
        out.append((mat, shift))
    return tuple(out)


# Reflection f is (R_f x + b_f) / 3.
REFLECTIONS: Tuple[Tuple[MatI, VecI], ...] = _reflection_data()

VERTICES: Tuple[VecQ, VecQ, VecQ, VecQ] = (
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(0), Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(0), Fraction(1)),
    (Fraction(1), Fraction(1), Fraction(1)),
)

FACE_TRIANGLE_VERTEX_INDICES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),  # D = ABC
    (1, 2, 3),  # A = BCD
    (0, 2, 3),  # B = ACD
    (0, 1, 3),  # C = ABD
)


def mat_mul(A: MatI, B: MatI) -> MatI:
    return tuple(
        tuple(A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def mat_vec(A: MatI, v: VecI) -> VecI:
    return (
        A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
        A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
        A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
    )


def dot_int(a: VecI, b: VecI) -> int:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross_int(a: VecI, b: VecI) -> VecI:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def gcd_many(values: Iterable[int]) -> int:
    g = 0
    for value in values:
        g = math.gcd(g, abs(value))
    return g


def normalize_int_vector(v: VecI) -> VecI:
    g = gcd_many(v)
    if g > 1:
        v = (v[0] // g, v[1] // g, v[2] // g)
    for x in v:
        if x:
            if x < 0:
                return (-v[0], -v[1], -v[2])
            return v
    return v


def det3(A: MatI) -> int:
    return (
        A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
    )


def rank3(A: MatI) -> int:
    if not any(x for row in A for x in row):
        return 0
    if det3(A) != 0:
        return 3
    for r1, r2 in ((0, 1), (0, 2), (1, 2)):
        for c1, c2 in ((0, 1), (0, 2), (1, 2)):
            if A[r1][c1] * A[r2][c2] - A[r1][c2] * A[r2][c1]:
                return 2
    return 1


def null_vector_rank2(A: MatI) -> Optional[VecI]:
    for r1, r2 in ((0, 1), (0, 2), (1, 2)):
        v = cross_int(A[r1], A[r2])
        if any(v):
            return normalize_int_vector(v)
    return None


def append_face_to_return_map(M: MatI, b: VecI, q: int, face: int) -> Tuple[MatI, VecI, int]:
    """
    If T(x) = (M x + b) / q, append a face on the right:
        T_new = T o reflection_face.
    """
    R, shift = REFLECTIONS[face]
    m_shift = mat_vec(M, shift)
    new_b = (m_shift[0] + 3 * b[0], m_shift[1] + 3 * b[1], m_shift[2] + 3 * b[2])
    return mat_mul(M, R), new_b, q * 3


def apply_scaled_affine_to_fraction_point(M: MatI, b: VecI, q: int, p: VecQ) -> VecQ:
    return (
        (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * p[2] + b[0]) / q,
        (M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * p[2] + b[1]) / q,
        (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * p[2] + b[2]) / q,
    )


def unfolded_face_triangles_exact(word: Sequence[int]) -> Tuple[TriangleQ, ...]:
    """
    Return the open face triangles hit by a linear word in the unfolded stack.
    """
    M = IDENTITY_MATRIX
    b = ZERO_VECTOR
    q = 1
    out: List[TriangleQ] = []
    for face in word:
        tri = tuple(
            apply_scaled_affine_to_fraction_point(M, b, q, VERTICES[idx])
            for idx in FACE_TRIANGLE_VERTEX_INDICES[face]
        )
        out.append(tri)  # type: ignore[arg-type]
        M, b, q = append_face_to_return_map(M, b, q, face)
    return tuple(out)


def unfolded_oriented_exit_triangles_exact(word: Sequence[int]) -> Tuple[TriangleQ, ...]:
    """
    Return unfolded hit triangles oriented by the exit normal of each current
    tetrahedron in the stack.

    A ray following the word from left to right exits the current unfolded
    tetrahedron through each listed face, so all three Plucker edge brackets
    are positive for a valid ordered transversal with that ray orientation.
    """
    M = IDENTITY_MATRIX
    b = ZERO_VECTOR
    q = 1
    out: List[TriangleQ] = []
    for face in word:
        tri = tuple(
            apply_scaled_affine_to_fraction_point(M, b, q, VERTICES[idx])
            for idx in FACE_TRIANGLE_VERTEX_INDICES[face]
        )
        inward = mat_vec(M, FACE_NORMALS[face])
        outward: VecQ = tuple(Fraction(-x) for x in inward)  # type: ignore[assignment]
        normal = cross_frac(sub_frac_vec(tri[1], tri[0]), sub_frac_vec(tri[2], tri[0]))
        if dot_frac(normal, outward) < 0:
            tri = (tri[0], tri[2], tri[1])
        out.append(tri)  # type: ignore[arg-type]
        M, b, q = append_face_to_return_map(M, b, q, face)
    return tuple(out)


def return_map_for_word(word: Sequence[int]) -> Tuple[MatI, VecI, int]:
    M = IDENTITY_MATRIX
    b = ZERO_VECTOR
    q = 1
    for face in word:
        M, b, q = append_face_to_return_map(M, b, q, face)
    return M, b, q


def solve_fraction_system(A: Sequence[Sequence[int]], rhs: Sequence[int]) -> Optional[Tuple[Fraction, ...]]:
    n = len(rhs)
    aug = [[Fraction(A[i][j]) for j in range(n)] + [Fraction(rhs[i])] for i in range(n)]

    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            aug[row], aug[pivot] = aug[pivot], aug[row]

        pivot_value = aug[row][col]
        aug[row] = [x / pivot_value for x in aug[row]]
        for r in range(n):
            if r == row:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            aug[r] = [aug[r][c] - factor * aug[row][c] for c in range(n + 1)]
        row += 1

    if row != n:
        return None
    return tuple(aug[i][n] for i in range(n))


def face_value(face: int, p: VecQ) -> Fraction:
    n = FACE_NORMALS[face]
    return n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + FACE_OFFSETS[face]


def face_derivative(face: int, d: VecI) -> int:
    return dot_int(FACE_NORMALS[face], d)


def add_scaled(p: VecQ, t: Fraction, d: VecI) -> VecQ:
    return (p[0] + t * d[0], p[1] + t * d[1], p[2] + t * d[2])


def scale_vec(v: VecI, c: int) -> VecI:
    return (c * v[0], c * v[1], c * v[2])


def negate_vec(v: VecI) -> VecI:
    return (-v[0], -v[1], -v[2])


def barycentric_abcd(p: VecQ) -> Tuple[Fraction, Fraction, Fraction, Fraction]:
    x, y, z = p
    return (
        (x - y - z + 1) / 2,
        (-x + y - z + 1) / 2,
        (-x - y + z + 1) / 2,
        (x + y + z - 1) / 2,
    )


def frac_to_json(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator}"


def frac_to_sympy(x: Fraction) -> sp.Rational:
    return sp.Rational(x.numerator, x.denominator)


@dataclass(frozen=True)
class Z3ChartPrefixState:
    fixed_axis: int
    fixed_sign: int
    active_inequalities: Tuple[InequalityId, ...]
    status: str


@dataclass(frozen=True)
class Z3PrefixState:
    word: Tuple[int, ...]
    charts: Tuple[Z3ChartPrefixState, ...]


@dataclass(frozen=True)
class GeometricValidityResult:
    status: str
    word: Tuple[int, ...]
    reason: str
    elapsed_seconds: float = 0.0
    z3_state: Optional[Z3PrefixState] = None
    chart_checks: int = 0
    active_chart_states: int = 0
    active_inequalities: int = 0
    reduction_checks: int = 0
    redundant_inequalities: int = 0
    reduction_unknown: int = 0
    linear_chart_checks: int = 0

    @property
    def feasible(self) -> Optional[bool]:
        if self.status == "sat":
            return True
        if self.status == "unsat":
            return False
        return None


def _z3_rational(z3_module, value: Fraction):
    if value.denominator == 1:
        return z3_module.RealVal(str(value.numerator))
    return z3_module.RealVal(f"{value.numerator}/{value.denominator}")


def start_slow_check_timer(
    kind: str,
    word: Sequence[int],
    warn_after_seconds: float,
) -> Tuple[threading.Event, Optional[threading.Timer]]:
    warned = threading.Event()
    timer: Optional[threading.Timer] = None
    if warn_after_seconds > 0:
        def warn_still_running() -> None:
            warned.set()
            print(
                f"[{kind}] still checking {word_name(word)} after {warn_after_seconds:.1f}s",
                file=sys.stderr,
                flush=True,
            )

        timer = threading.Timer(warn_after_seconds, warn_still_running)
        timer.daemon = True
        timer.start()
    return warned, timer


def sub_frac_vec(a: VecQ, b: VecQ) -> VecQ:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross_frac(a: VecQ, b: VecQ) -> VecQ:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot_frac(a: VecQ, b: VecQ) -> Fraction:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def triangle_plane_exact(tri: TriangleQ) -> Tuple[VecQ, Fraction]:
    a, b, c = tri
    normal = cross_frac(sub_frac_vec(b, a), sub_frac_vec(c, a))
    offset = -dot_frac(normal, a)
    return normal, offset


def barycentric_from_point_expr(z3_module, tri: TriangleQ, q_exprs: Sequence[object]):
    """
    Return alpha, beta for q = A + alpha(B-A) + beta(C-A).

    The returned expressions are linear in q.  We choose a non-singular 2x2
    coordinate minor exactly, so this stays in rational arithmetic.
    """
    a, b, c = tri
    e1 = sub_frac_vec(b, a)
    e2 = sub_frac_vec(c, a)
    for r, s in ((0, 1), (0, 2), (1, 2)):
        det = e1[r] * e2[s] - e1[s] * e2[r]
        if det == 0:
            continue
        rhs_r = q_exprs[r] - _z3_rational(z3_module, a[r])
        rhs_s = q_exprs[s] - _z3_rational(z3_module, a[s])
        det_z3 = _z3_rational(z3_module, det)
        alpha = (rhs_r * _z3_rational(z3_module, e2[s]) - rhs_s * _z3_rational(z3_module, e2[r])) / det_z3
        beta = (_z3_rational(z3_module, e1[r]) * rhs_s - _z3_rational(z3_module, e1[s]) * rhs_r) / det_z3
        return alpha, beta
    raise ValueError("degenerate triangle")


def z3_dot_fraction_expr(z3_module, coeffs: VecQ, exprs: Sequence[object]):
    return sum(_z3_rational(z3_module, coeffs[i]) * exprs[i] for i in range(3))


def z3_plucker_edge_bracket(z3_module, a: VecQ, b: VecQ, d: Sequence[object], m: Sequence[object]):
    """
    Linear Plucker side operator between an oriented line (d, m) and edge ab.

    For an actual line with moment m = p x d, a transverse intersection with
    the open triangle has all three oriented edge brackets strictly same-signed.
    We intentionally omit d.m = 0 here: the resulting linear formula is a
    relaxation, so UNSAT is still a certified no-line result.
    """
    cross = cross_frac(a, b)
    edge = sub_frac_vec(b, a)
    return sum(_z3_rational(z3_module, cross[i]) * d[i] for i in range(3)) + sum(
        _z3_rational(z3_module, edge[i]) * m[i] for i in range(3)
    )


def certify_stack_transversal_plucker_lra(
    word: Sequence[int],
    timeout_ms: int = 0,
    warn_after_seconds: float = 10.0,
) -> GeometricValidityResult:
    """
    Fast exact linear necessary condition for a line stabbing all open triangles.

    This is a relaxation in Plucker coordinates.  If every nonzero chart is
    UNSAT in QF_LRA, no true line exists and the word can be pruned.  SAT is
    not trusted as feasibility, because the Plucker quadratic d.m = 0 was
    deliberately omitted to keep this screen linear and fast.
    """
    word = tuple(word)
    if len(word) <= 1:
        return GeometricValidityResult("unknown", word, "plucker_length_at_most_one")

    try:
        import z3  # type: ignore
    except ImportError as exc:
        return GeometricValidityResult("unknown", word, f"z3_unavailable:{exc}")

    triangles = unfolded_face_triangles_exact(word)
    start = time.perf_counter()
    _, timer = start_slow_check_timer("plucker-lra", word, warn_after_seconds)
    chart_checks = 0
    any_unknown = False

    try:
        for fixed_coord in range(6):
            for fixed_sign in (1, -1):
                solver = z3_make_solver(z3, timeout_ms, logic="QF_LRA")
                variables = [z3.Real(name) for name in ("dx", "dy", "dz", "mx", "my", "mz")]
                solver.add(variables[fixed_coord] == fixed_sign)
                d = variables[:3]
                m = variables[3:]
                for tri in triangles:
                    a, b, c = tri
                    brackets = (
                        z3_plucker_edge_bracket(z3, a, b, d, m),
                        z3_plucker_edge_bracket(z3, b, c, d, m),
                        z3_plucker_edge_bracket(z3, c, a, d, m),
                    )
                    solver.add(
                        z3.Or(
                            z3.And(*(value > 0 for value in brackets)),
                            z3.And(*(value < 0 for value in brackets)),
                        )
                    )
                answer = solver.check()
                chart_checks += 1
                if answer == z3.sat:
                    return GeometricValidityResult(
                        "unknown",
                        word,
                        "plucker_relaxation_sat",
                        time.perf_counter() - start,
                        linear_chart_checks=chart_checks,
                    )
                if answer != z3.unsat:
                    return GeometricValidityResult(
                        "unknown",
                        word,
                        f"plucker_lra_{answer}",
                        time.perf_counter() - start,
                        linear_chart_checks=chart_checks,
                    )
    except Exception as exc:
        return GeometricValidityResult(
            "unknown",
            word,
            f"plucker_error:{type(exc).__name__}:{exc}",
            time.perf_counter() - start,
            linear_chart_checks=chart_checks,
        )
    finally:
        if timer is not None:
            timer.cancel()

    return GeometricValidityResult(
        "unknown" if any_unknown else "unsat",
        word,
        "plucker_unknown" if any_unknown else "plucker_unsat",
        time.perf_counter() - start,
        linear_chart_checks=chart_checks,
    )


def certify_stack_transversal_plucker_nra(
    word: Sequence[int],
    timeout_ms: int = 0,
    warn_after_seconds: float = 10.0,
) -> GeometricValidityResult:
    """
    Exact Plucker necessary condition with the true line relation d.m = 0.

    This still ignores the order of the triangle hits, so SAT is not enough to
    accept a prefix.  UNSAT is a certified prune and is often much faster than
    the ordered stack formulation because the variable count stays fixed.
    """
    word = tuple(word)
    if len(word) <= 1:
        return GeometricValidityResult("unknown", word, "plucker_length_at_most_one")

    try:
        import z3  # type: ignore
    except ImportError as exc:
        return GeometricValidityResult("unknown", word, f"z3_unavailable:{exc}")

    triangles = unfolded_oriented_exit_triangles_exact(word)
    start = time.perf_counter()
    _, timer = start_slow_check_timer("plucker-nra", word, warn_after_seconds)
    chart_checks = 0
    any_unknown = False

    try:
        for fixed_coord in range(6):
            for fixed_sign in (1, -1):
                solver = z3_make_solver(z3, timeout_ms)
                variables = [z3.Real(name) for name in ("dx", "dy", "dz", "mx", "my", "mz")]
                solver.add(variables[fixed_coord] == fixed_sign)
                d = variables[:3]
                m = variables[3:]
                solver.add(sum(d[i] * m[i] for i in range(3)) == 0)
                for tri in triangles:
                    a, b, c = tri
                    solver.add(z3_plucker_edge_bracket(z3, a, b, d, m) > 0)
                    solver.add(z3_plucker_edge_bracket(z3, b, c, d, m) > 0)
                    solver.add(z3_plucker_edge_bracket(z3, c, a, d, m) > 0)
                answer = solver.check()
                chart_checks += 1
                if answer == z3.sat:
                    return GeometricValidityResult(
                        "unknown",
                        word,
                        "plucker_quadratic_sat",
                        time.perf_counter() - start,
                        linear_chart_checks=chart_checks,
                    )
                if answer != z3.unsat:
                    return GeometricValidityResult(
                        "unknown",
                        word,
                        f"plucker_quadratic_{answer}",
                        time.perf_counter() - start,
                        linear_chart_checks=chart_checks,
                    )
    except Exception as exc:
        return GeometricValidityResult(
            "unknown",
            word,
            f"plucker_quadratic_error:{type(exc).__name__}:{exc}",
            time.perf_counter() - start,
            linear_chart_checks=chart_checks,
        )
    finally:
        if timer is not None:
            timer.cancel()

    return GeometricValidityResult(
        "unknown" if any_unknown else "unsat",
        word,
        "plucker_quadratic_unknown" if any_unknown else "plucker_quadratic_unsat",
        time.perf_counter() - start,
        linear_chart_checks=chart_checks,
    )


def plucker_prune_worker(
    payload: Tuple[Tuple[int, ...], int, bool, bool, float]
) -> Tuple[Tuple[int, ...], GeometricValidityResult, Optional[GeometricValidityResult]]:
    word, timeout_ms, use_lra, use_nra, warn_after_seconds = payload
    if use_lra:
        lra = certify_stack_transversal_plucker_lra(
            word,
            timeout_ms=timeout_ms,
            warn_after_seconds=warn_after_seconds,
        )
        if lra.status == "unsat" or not use_nra:
            return word, lra, None
    else:
        lra = GeometricValidityResult("unknown", word, "plucker_lra_disabled")

    nra = (
        certify_stack_transversal_plucker_nra(
            word,
            timeout_ms=timeout_ms,
            warn_after_seconds=warn_after_seconds,
        )
        if use_nra
        else None
    )
    return word, lra, nra


def z3_transversal_chart_constraints(
    z3_module,
    triangles: Sequence[TriangleQ],
    fixed_axis: int,
    fixed_sign: int,
) -> Tuple[List[object], Dict[InequalityId, object], Tuple[InequalityId, ...]]:
    hard_constraints: List[object] = []
    inequalities: Dict[InequalityId, object] = {}
    inequality_order: List[InequalityId] = []

    def add_inequality(key: InequalityId, expr: object) -> None:
        inequalities[key] = expr
        inequality_order.append(key)

    first = triangles[0]
    u = z3_module.Real("u")
    v = z3_module.Real("v")
    add_inequality(("u", 0), u > 0)
    add_inequality(("v", 0), v > 0)
    add_inequality(("uv", 0), u + v < 1)

    a0, b0, c0 = first
    e1 = sub_frac_vec(b0, a0)
    e2 = sub_frac_vec(c0, a0)
    p = [
        _z3_rational(z3_module, a0[i])
        + u * _z3_rational(z3_module, e1[i])
        + v * _z3_rational(z3_module, e2[i])
        for i in range(3)
    ]

    d = []
    for axis in range(3):
        if axis == fixed_axis:
            d.append(_z3_rational(z3_module, Fraction(fixed_sign)))
        else:
            var = z3_module.Real(f"d_{axis}")
            d.append(var)

    previous_t = z3_module.RealVal("0")
    for i, tri in enumerate(triangles[1:], start=1):
        t = z3_module.Real(f"t_{i}")
        add_inequality(("t", i), previous_t < t)

        normal, offset = triangle_plane_exact(tri)
        q = [p[j] + t * d[j] for j in range(3)]
        hard_constraints.append(
            z3_dot_fraction_expr(z3_module, normal, q) + _z3_rational(z3_module, offset) == 0
        )

        alpha, beta = barycentric_from_point_expr(z3_module, tri, q)
        add_inequality(("alpha", i), alpha > 0)
        add_inequality(("beta", i), beta > 0)
        add_inequality(("alphabeta", i), alpha + beta < 1)
        previous_t = t

    return hard_constraints, inequalities, tuple(inequality_order)


def z3_make_solver(z3_module, timeout_ms: int, logic: str = "QF_NRA"):
    solver = z3_module.SolverFor(logic)
    if timeout_ms > 0:
        solver.set(timeout=int(timeout_ms))
    return solver


def z3_check_transversal_chart(z3_module, triangles: Sequence[TriangleQ], fixed_axis: int, fixed_sign: int):
    solver = z3_make_solver(z3_module, 0)
    hard_constraints, inequalities, inequality_order = z3_transversal_chart_constraints(
        z3_module,
        triangles,
        fixed_axis,
        fixed_sign,
    )
    solver.add(*hard_constraints)
    solver.add(*(inequalities[key] for key in inequality_order))
    return solver


def z3_reduce_active_inequalities(
    z3_module,
    hard_constraints: Sequence[object],
    inequalities: Dict[InequalityId, object],
    active_inequalities: Sequence[InequalityId],
    candidate_inequalities: Sequence[InequalityId],
    timeout_ms: int,
) -> Tuple[Tuple[InequalityId, ...], int, int, int]:
    active = list(dict.fromkeys(active_inequalities))
    active_set = set(active)
    checks = 0
    removed = 0
    unknown = 0

    for key in candidate_inequalities:
        if key not in active_set:
            continue
        solver = z3_make_solver(z3_module, timeout_ms)
        solver.add(*hard_constraints)
        solver.add(*(inequalities[other] for other in active if other != key))
        solver.add(z3_module.Not(inequalities[key]))
        answer = solver.check()
        checks += 1
        if answer == z3_module.unsat:
            active.remove(key)
            active_set.remove(key)
            removed += 1
        elif answer != z3_module.sat:
            unknown += 1

    return tuple(active), checks, removed, unknown


def certify_stack_transversal_z3(
    word: Sequence[int],
    timeout_ms: int = 0,
    warn_after_seconds: float = 10.0,
) -> GeometricValidityResult:
    """
    Decide whether the unfolded stack has a straight line through all requested
    open hit faces in order.

    The formula is quantifier-free nonlinear real arithmetic with exact rational
    literals.  The answer is used one-sided in the search: only UNSAT prunes.
    SAT keeps the word, and UNKNOWN keeps the word as well.
    """
    word = tuple(word)
    if len(word) <= 1:
        return GeometricValidityResult("sat", word, "length_at_most_one")

    try:
        import z3  # type: ignore
    except ImportError as exc:
        return GeometricValidityResult("unknown", word, f"z3_unavailable:{exc}")

    triangles = unfolded_face_triangles_exact(word)

    start = time.perf_counter()
    warned = threading.Event()
    timer: Optional[threading.Timer] = None
    if warn_after_seconds > 0:
        def warn_still_running() -> None:
            warned.set()
            print(
                f"[z3] still checking {word_name(word)} after {warn_after_seconds:.1f}s",
                file=sys.stderr,
                flush=True,
            )

        timer = threading.Timer(warn_after_seconds, warn_still_running)
        timer.daemon = True
        timer.start()

    unknown_reason = None
    any_unknown = False
    answer = z3.unsat
    try:
        for fixed_axis in range(3):
            for fixed_sign in (1, -1):
                solver = z3_check_transversal_chart(z3, triangles, fixed_axis, fixed_sign)
                if timeout_ms > 0:
                    solver.set(timeout=int(timeout_ms))
                chart_answer = solver.check()
                if chart_answer == z3.sat:
                    answer = z3.sat
                    raise StopIteration
                if chart_answer != z3.unsat:
                    any_unknown = True
                    unknown_reason = f"z3_{chart_answer}"
        if any_unknown:
            answer = z3.unknown
    except StopIteration:
        pass
    except Exception as exc:
        elapsed = time.perf_counter() - start
        if timer is not None:
            timer.cancel()
        return GeometricValidityResult("unknown", word, f"z3_error:{type(exc).__name__}:{exc}", elapsed)
    finally:
        if timer is not None:
            timer.cancel()

    elapsed = time.perf_counter() - start
    if warned.is_set():
        print(
            f"[z3] finished {word_name(word)} with {answer} after {elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    if answer == z3.sat:
        return GeometricValidityResult("sat", word, "z3_sat", elapsed)
    if answer == z3.unsat:
        return GeometricValidityResult("unsat", word, "z3_unsat", elapsed)
    return GeometricValidityResult("unknown", word, unknown_reason or f"z3_{answer}", elapsed)


def certify_stack_transversal_z3_reduced(
    word: Sequence[int],
    parent_state: Optional[Z3PrefixState],
    timeout_ms: int = 0,
    warn_after_seconds: float = 10.0,
    reduce_inequalities: bool = True,
    reduction_scope: str = "new",
) -> GeometricValidityResult:
    """
    Certified prefix check that propagates a reduced strict-inequality set.

    For each direction chart, a parent SAT/UNKNOWN chart gives a set of active
    inequalities known to be sufficient for the parent prefix.  The child check
    adds only the new inequalities for the appended face.  A SAT child chart is
    greedily reduced by deleting an inequality only when Z3 proves the remaining
    constraints imply it.  UNKNOWN charts are carried forward conservatively.
    """
    word = tuple(word)
    if len(word) <= 1:
        state = Z3PrefixState(
            word,
            tuple(
                Z3ChartPrefixState(axis, sign, tuple(), "sat")
                for axis in range(3)
                for sign in (1, -1)
            ),
        )
        return GeometricValidityResult("sat", word, "length_at_most_one", z3_state=state)

    try:
        import z3  # type: ignore
    except ImportError as exc:
        return GeometricValidityResult("unknown", word, f"z3_unavailable:{exc}")

    triangles = unfolded_face_triangles_exact(word)
    parent_matches = parent_state is not None and parent_state.word == word[:-1]
    if parent_matches:
        seeds = list(parent_state.charts)
        parent_length = len(parent_state.word)
    else:
        seeds = [
            Z3ChartPrefixState(axis, sign, tuple(), "unknown")
            for axis in range(3)
            for sign in (1, -1)
        ]
        parent_length = 0

    start = time.perf_counter()
    warned = threading.Event()
    timer: Optional[threading.Timer] = None
    if warn_after_seconds > 0:
        def warn_still_running() -> None:
            warned.set()
            print(
                f"[z3] still checking {word_name(word)} after {warn_after_seconds:.1f}s",
                file=sys.stderr,
                flush=True,
            )

        timer = threading.Timer(warn_after_seconds, warn_still_running)
        timer.daemon = True
        timer.start()

    chart_checks = 0
    reduction_checks = 0
    redundant_inequalities = 0
    reduction_unknown = 0
    retained: List[Z3ChartPrefixState] = []
    tried_charts: set[Tuple[int, int]] = set()

    try:
        while True:
            for seed in seeds:
                chart_key = (seed.fixed_axis, seed.fixed_sign)
                if chart_key in tried_charts:
                    continue
                tried_charts.add(chart_key)
                hard_constraints, inequalities, inequality_order = z3_transversal_chart_constraints(
                    z3,
                    triangles,
                    seed.fixed_axis,
                    seed.fixed_sign,
                )
                if parent_matches:
                    active = list(seed.active_inequalities)
                    active_set = set(active)
                    new_keys: List[InequalityId] = []
                    for key in inequality_order:
                        if key[1] >= parent_length and key not in active_set:
                            active.append(key)
                            active_set.add(key)
                            new_keys.append(key)
                else:
                    active = list(inequality_order)
                    new_keys = list(inequality_order)

                solver = z3_make_solver(z3, timeout_ms)
                solver.add(*hard_constraints)
                solver.add(*(inequalities[key] for key in active))
                answer = solver.check()
                chart_checks += 1

                if answer == z3.unsat:
                    continue
                if answer == z3.sat:
                    if reduce_inequalities:
                        if reduction_scope == "all":
                            candidate_inequalities = list(active)
                        elif reduction_scope == "new":
                            candidate_inequalities = new_keys
                        else:
                            raise ValueError(f"unknown Z3 reduction scope: {reduction_scope}")
                        active_tuple, checks, removed, unknown = z3_reduce_active_inequalities(
                            z3,
                            hard_constraints,
                            inequalities,
                            active,
                            candidate_inequalities,
                            timeout_ms,
                        )
                        reduction_checks += checks
                        redundant_inequalities += removed
                        reduction_unknown += unknown
                    else:
                        active_tuple = tuple(active)
                    retained.append(
                        Z3ChartPrefixState(
                            seed.fixed_axis,
                            seed.fixed_sign,
                            active_tuple,
                            "sat",
                        )
                    )
                    break

                retained.append(
                    Z3ChartPrefixState(
                        seed.fixed_axis,
                        seed.fixed_sign,
                        tuple(active),
                        f"z3_{answer}",
                    )
                )
                break

            if retained or not parent_matches or len(tried_charts) >= 6:
                break
            seeds = [
                Z3ChartPrefixState(axis, sign, tuple(), "unknown")
                for axis in range(3)
                for sign in (1, -1)
            ]
            parent_matches = False
            parent_length = 0
    except Exception as exc:
        elapsed = time.perf_counter() - start
        if timer is not None:
            timer.cancel()
        return GeometricValidityResult(
            "unknown",
            word,
            f"z3_error:{type(exc).__name__}:{exc}",
            elapsed,
            chart_checks=chart_checks,
            reduction_checks=reduction_checks,
            redundant_inequalities=redundant_inequalities,
            reduction_unknown=reduction_unknown,
        )
    finally:
        if timer is not None:
            timer.cancel()

    elapsed = time.perf_counter() - start
    answer_text = "unsat"
    if any(chart.status == "sat" for chart in retained):
        answer_text = "sat"
    elif retained:
        answer_text = "unknown"

    if warned.is_set():
        print(
            f"[z3] finished {word_name(word)} with {answer_text} after {elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    state = Z3PrefixState(word, tuple(retained)) if retained else None
    active_inequalities = sum(len(chart.active_inequalities) for chart in retained)

    if any(chart.status == "sat" for chart in retained):
        return GeometricValidityResult(
            "sat",
            word,
            "z3_sat_reduced",
            elapsed,
            z3_state=state,
            chart_checks=chart_checks,
            active_chart_states=len(retained),
            active_inequalities=active_inequalities,
            reduction_checks=reduction_checks,
            redundant_inequalities=redundant_inequalities,
            reduction_unknown=reduction_unknown,
        )
    if retained:
        return GeometricValidityResult(
            "unknown",
            word,
            "z3_unknown_reduced",
            elapsed,
            z3_state=state,
            chart_checks=chart_checks,
            active_chart_states=len(retained),
            active_inequalities=active_inequalities,
            reduction_checks=reduction_checks,
            redundant_inequalities=redundant_inequalities,
            reduction_unknown=reduction_unknown,
        )
    return GeometricValidityResult(
        "unsat",
        word,
        "z3_unsat_reduced",
        elapsed,
        chart_checks=chart_checks,
        reduction_checks=reduction_checks,
        redundant_inequalities=redundant_inequalities,
        reduction_unknown=reduction_unknown,
    )


@dataclass(frozen=True)
class ExactOrbit:
    word: Tuple[int, ...]
    points: Tuple[VecQ, ...]
    axis_direction: VecI
    initial_direction: VecI

    @property
    def period(self) -> int:
        return len(self.word)

    @property
    def leg_vectors(self) -> Tuple[VecQ, ...]:
        out: List[VecQ] = []
        for i, p in enumerate(self.points):
            q = self.points[(i + 1) % len(self.points)]
            out.append((q[0] - p[0], q[1] - p[1], q[2] - p[2]))
        return tuple(out)

    @property
    def leg_length_squared(self) -> Tuple[Fraction, ...]:
        out: List[Fraction] = []
        for v in self.leg_vectors:
            out.append(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        return tuple(out)

    @property
    def total_length_sympy(self) -> sp.Expr:
        return sp.simplify(sum(sp.sqrt(frac_to_sympy(x)) for x in self.leg_length_squared))


@dataclass(frozen=True)
class VerificationResult:
    orbit: Optional[ExactOrbit]
    reason: str


def verify_word_exact(
    word: Sequence[int],
    M: Optional[MatI] = None,
    b: Optional[VecI] = None,
    q: Optional[int] = None,
) -> VerificationResult:
    """
    Exact closed-orbit check for one cyclic face word.

    The affine return map is T(x) = (M x + b) / q.  A closed path must lie on
    an invariant axis of T, and the folded ray must hit exactly the requested
    next face at every step.  Strict positivity of the other face inequalities
    excludes edge and vertex hits.
    """
    word = tuple(word)
    if len(word) < 2:
        return VerificationResult(None, "too_short")
    if M is None or b is None or q is None:
        M, b, q = return_map_for_word(word)

    B: MatI = tuple(
        tuple(M[i][j] - (q if i == j else 0) for j in range(3))
        for i in range(3)
    )  # type: ignore[assignment]
    if rank3(B) != 2:
        return VerificationResult(None, "return_map_has_no_unique_axis")

    axis = null_vector_rank2(B)
    if axis is None:
        return VerificationResult(None, "return_axis_nullspace_failed")

    start_face = word[0]
    n = FACE_NORMALS[start_face]
    rows = [
        [B[0][0], B[0][1], B[0][2], -axis[0]],
        [B[1][0], B[1][1], B[1][2], -axis[1]],
        [B[2][0], B[2][1], B[2][2], -axis[2]],
        [n[0], n[1], n[2], 0],
    ]
    rhs = [-b[0], -b[1], -b[2], -FACE_OFFSETS[start_face]]
    solved = solve_fraction_system(rows, rhs)
    if solved is None:
        return VerificationResult(None, "axis_start_intersection_failed")
    p0: VecQ = (solved[0], solved[1], solved[2])

    start_reflection = REFLECTIONS[start_face][0]
    last_reason = "start_direction_points_out"

    for sign in (1, -1):
        d0 = mat_vec(start_reflection, axis)
        if sign < 0:
            d0 = negate_vec(d0)
        if face_derivative(start_face, d0) <= 0:
            continue

        p = p0
        d = d0
        points: List[VecQ] = [p0]
        ok = True

        for k in range(len(word)):
            expected_next = word[(k + 1) % len(word)]
            candidates: List[Tuple[Fraction, int]] = []
            for face in range(4):
                du = face_derivative(face, d)
                if du == 0:
                    continue
                t = -face_value(face, p) / du
                if t > 0:
                    candidates.append((t, face))
            if not candidates:
                ok = False
                last_reason = "no_forward_face_hit"
                break

            t_min = min(t for t, _ in candidates)
            hit_faces = [face for t, face in candidates if t == t_min]
            if len(hit_faces) != 1:
                ok = False
                last_reason = "edge_or_vertex_tie"
                break

            hit_face = hit_faces[0]
            if hit_face != expected_next:
                ok = False
                last_reason = "wrong_next_face"
                break

            p = add_scaled(p, t_min, d)
            for face in range(4):
                value = face_value(face, p)
                if face == hit_face:
                    if value != 0:
                        ok = False
                        last_reason = "hit_not_on_expected_face"
                        break
                elif value <= 0:
                    ok = False
                    last_reason = "edge_or_vertex_hit"
                    break
            if not ok:
                break

            d = mat_vec(REFLECTIONS[hit_face][0], d)
            if k < len(word) - 1:
                points.append(p)

        if ok and p == p0 and d == scale_vec(d0, 3 ** len(word)):
            return VerificationResult(
                ExactOrbit(
                    word=word,
                    points=tuple(points),
                    axis_direction=axis,
                    initial_direction=d0,
                ),
                "closed",
            )
        if ok:
            last_reason = "does_not_return_to_start_state"

    return VerificationResult(None, last_reason)


def word_name(word: Sequence[int]) -> str:
    return "".join(FACE_NAMES[i] for i in word)


def parse_word(text: str) -> Tuple[int, ...]:
    try:
        return tuple(NAME_TO_FACE[ch] for ch in text.strip().upper())
    except KeyError as exc:
        raise ValueError(f"invalid face label {exc}; use only A, B, C, D") from exc


def primitive_word(word: Sequence[int]) -> bool:
    m = len(word)
    w = tuple(word)
    for d in range(1, m):
        if m % d == 0 and w == w[:d] * (m // d):
            return False
    return True


def rotations(word: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
    for k in range(len(word)):
        yield word[k:] + word[:k]


def canonical_cyclic_reversal(word: Tuple[int, ...]) -> Tuple[int, ...]:
    rev = tuple(reversed(word))
    return min(itertools.chain(rotations(word), rotations(rev)))


def permute_word(word: Sequence[int], perm: Sequence[int]) -> Tuple[int, ...]:
    return tuple(perm[x] for x in word)


def canonical_relabel_linear(word: Sequence[int]) -> Tuple[int, ...]:
    """
    Return the lexicographically least relabeling of one linear word under S4.

    For a full symmetric group on labels, the least relabeling is obtained by
    naming the first new symbol 0, the next new symbol 1, and so on.
    """
    mapping: Dict[int, int] = {}
    next_label = 0
    out = []
    for face in word:
        if face not in mapping:
            mapping[face] = next_label
            next_label += 1
        out.append(mapping[face])
    return tuple(out)


def canonical_full_symmetry(word: Tuple[int, ...]) -> Tuple[int, ...]:
    rev = tuple(reversed(word))
    return min(
        itertools.chain(
            (canonical_relabel_linear(rot) for rot in rotations(word)),
            (canonical_relabel_linear(rot) for rot in rotations(rev)),
        )
    )


def canonical_by_mode(word: Tuple[int, ...], mode: str) -> Tuple[int, ...]:
    if mode == "none":
        return word
    if mode == "cyclic":
        return canonical_cyclic_reversal(word)
    if mode == "full":
        return canonical_full_symmetry(word)
    raise ValueError(f"unknown equivalence mode: {mode}")


@lru_cache(maxsize=4096)
def stabilizer_of_prefix(prefix: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
    if not prefix:
        return ALL_FACE_PERMUTATIONS
    return tuple(perm for perm in ALL_FACE_PERMUTATIONS if permute_word(prefix, perm) == prefix)


def canonical_child_under_stabilizer(prefix: Tuple[int, ...], child: Tuple[int, ...]) -> Tuple[int, ...]:
    return min(permute_word(child, perm) for perm in stabilizer_of_prefix(prefix))


@lru_cache(maxsize=CANONICAL_FORBIDDEN_CACHE_SIZE)
def canonical_forbidden_subword(word: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Canonicalize a linear forbidden subword under tetrahedron symmetries and
    reversal.  Cyclic shifts are intentionally not used for linear subwords.
    """
    rev = tuple(reversed(word))
    return min(canonical_relabel_linear(word), canonical_relabel_linear(rev))


def has_xyxy_suffix(word: Tuple[int, ...]) -> bool:
    if len(word) < 4:
        return False
    a, b, c, d = word[-4:]
    return a == c and b == d and a != b


class ForbiddenSubwordCache:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.patterns: set[Tuple[int, ...]] = set()
        self.lengths: set[int] = set()
        self.packed_variants_by_length: DefaultDict[int, set[int]] = defaultdict(set)
        self.new_patterns = 0

    def _add_packed_variants(self, pattern: Tuple[int, ...]) -> None:
        variants = self.packed_variants_by_length[len(pattern)]
        rev = tuple(reversed(pattern))
        for perm in ALL_FACE_PERMUTATIONS:
            variants.add(pack_word(permute_word(pattern, perm)))
            variants.add(pack_word(permute_word(rev, perm)))

    def register(self, word: Sequence[int]) -> bool:
        if not self.enabled:
            return False
        pattern = canonical_forbidden_subword(tuple(word))
        before = len(self.patterns)
        self.patterns.add(pattern)
        self.lengths.add(len(pattern))
        if len(self.patterns) != before:
            self._add_packed_variants(pattern)
            self.new_patterns += 1
            return True
        return False

    def load_patterns_from_file(self, path: Path) -> int:
        data = load_json_without_frontier_words(path)
        if isinstance(data, dict) and "summary" in data:
            data = data["summary"].get("forbidden_subwords", {})
        elif isinstance(data, dict) and "forbidden_subwords" in data:
            data = data["forbidden_subwords"]
        if isinstance(data, dict) and "patterns" in data:
            patterns = data["patterns"]
        elif isinstance(data, list):
            patterns = data
        else:
            raise ValueError(f"cannot find forbidden patterns in {path}")

        before = len(self.patterns)
        for text in patterns:
            self.register(parse_word(str(text)))
        return len(self.patterns) - before

    def write_patterns_to_file(self, path: Path, level: int, stats: Optional["SearchStats"] = None) -> None:
        payload: Dict[str, object] = {
            "level": level,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "forbidden_subwords": self.to_json(),
        }
        if stats is not None:
            payload["level_reports"] = {str(k): v for k, v in sorted(stats.level_reports.items())}
            payload["elapsed_seconds"] = stats.elapsed()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def register_xyxy_suffix_if_present(self, word: Tuple[int, ...]) -> bool:
        if has_xyxy_suffix(word):
            return self.register(word[-4:])
        return False

    def hits_linear_suffix(self, word: Tuple[int, ...]) -> bool:
        if not self.enabled or not self.lengths:
            return False
        m = len(word)
        for length in self.lengths:
            if length <= m and canonical_forbidden_subword(word[-length:]) in self.patterns:
                return True
        return False

    def hits_linear_suffix_packed(self, code: int, word_length: int) -> bool:
        if not self.enabled or not self.lengths:
            return False
        for length in self.lengths:
            if length > word_length:
                continue
            variants = self.packed_variants_by_length.get(length)
            if not variants:
                continue
            suffix = code & ((1 << (2 * length)) - 1)
            if suffix in variants:
                return True
        return False

    def hits_linear_subword(self, word: Tuple[int, ...]) -> bool:
        if not self.enabled or not self.lengths:
            return False
        m = len(word)
        for length in self.lengths:
            if length > m:
                continue
            for start in range(m - length + 1):
                if canonical_forbidden_subword(word[start:start + length]) in self.patterns:
                    return True
        return False

    def hits_linear_subword_packed(self, code: int, word_length: int) -> bool:
        if not self.enabled or not self.lengths:
            return False
        for length in self.lengths:
            if length > word_length:
                continue
            variants = self.packed_variants_by_length.get(length)
            if not variants:
                continue
            mask = (1 << (2 * length)) - 1
            max_shift = 2 * (word_length - length)
            for shift in range(max_shift, -1, -2):
                if ((code >> shift) & mask) in variants:
                    return True
        return False

    def hits_cyclic_word(self, word: Tuple[int, ...]) -> bool:
        if not self.enabled or not self.lengths:
            return False
        m = len(word)
        doubled = word + word
        for length in self.lengths:
            if length > m:
                continue
            for start in range(m):
                sub = doubled[start:start + length]
                if canonical_forbidden_subword(sub) in self.patterns:
                    return True
        return False

    def to_json(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "count": len(self.patterns),
            "patterns": [word_name(p) for p in sorted(self.patterns, key=lambda w: (len(w), w))],
            "note": "linear patterns canonicalized under tetrahedron face relabeling and reversal",
        }


FrontierItem = Tuple[Tuple[int, ...], Optional[MatI], Optional[VecI], Optional[int], Optional[Z3PrefixState]]
PackedFrontierItem = int
SearchFrontierItem = Union[FrontierItem, PackedFrontierItem]
FrontierContainer = Union[List[SearchFrontierItem], array]


def pack_word(word: Sequence[int]) -> int:
    code = 0
    for face in word:
        code = (code << 2) | int(face)
    return code


def unpack_word(code: int, length: int) -> Tuple[int, ...]:
    return tuple((code >> (2 * (length - 1 - i))) & 3 for i in range(length))


def append_packed_face(code: int, face: int) -> int:
    return (code << 2) | face


PACKED_FACE_FROM_BYTE = {
    ord("D"): 0,
    ord("A"): 1,
    ord("B"): 2,
    ord("C"): 3,
}


def pack_word_bytes(raw: bytes) -> int:
    code = 0
    for byte in raw:
        code = (code << 2) | PACKED_FACE_FROM_BYTE[byte]
    return code


def parse_word_bytes(raw: bytes) -> Tuple[int, ...]:
    return tuple(PACKED_FACE_FROM_BYTE[byte] for byte in raw)


def packed_face_at(code: int, length: int, index: int) -> int:
    return (code >> (2 * (length - 1 - index))) & 3


def packed_can_be_full_canonical(code: int, length: int) -> bool:
    if length <= 0:
        return True
    if packed_face_at(code, length, 0) != 0:
        return False
    if length >= 2 and packed_face_at(code, length, 1) != 1:
        return False
    return True


def search_frontier_word(item: SearchFrontierItem, length: int) -> Tuple[int, ...]:
    if isinstance(item, int):
        return unpack_word(item, length)
    return item[0]


def search_frontier_components(
    item: SearchFrontierItem,
    length: int,
) -> Tuple[Tuple[int, ...], Optional[MatI], Optional[VecI], Optional[int], Optional[Z3PrefixState]]:
    if isinstance(item, int):
        return unpack_word(item, length), None, None, None, None
    return item


def frontier_item_for_word(word: Tuple[int, ...], lazy_map: bool = False) -> SearchFrontierItem:
    if lazy_map:
        return pack_word(word)
    M, b, q = return_map_for_word(word)
    return (word, M, b, q, None)


def packed_frontier_container(level: int) -> FrontierContainer:
    # Packed words through level 32 fit in 64 bits.  array('Q') is much lighter
    # than a list of Python ints for the very large level-30/32 frontiers.
    if level <= 32:
        return array("Q")
    return []


def empty_frontier_container(level: int, lazy_map: bool) -> FrontierContainer:
    if lazy_map:
        return packed_frontier_container(level)
    return []


def materialize_return_map(
    word: Tuple[int, ...],
    M: Optional[MatI],
    b: Optional[VecI],
    q: Optional[int],
) -> Tuple[MatI, VecI, int]:
    if M is not None and b is not None and q is not None:
        return M, b, q
    return return_map_for_word(word)


def frontier_words(frontier: Sequence[SearchFrontierItem], level: int) -> List[str]:
    return [word_name(search_frontier_word(item, level)) for item in frontier]


def checkpoint_frontier_words(data: Dict[str, object], path: Path) -> Tuple[int, List[str]]:
    frontier = data.get("frontier")
    if isinstance(frontier, dict):
        raw_level = frontier.get("level")
        raw_words = frontier.get("words")
    else:
        raw_level = data.get("frontier_level", data.get("level"))
        raw_words = data.get("frontier_words")

    if raw_level is None or not isinstance(raw_words, list):
        raise ValueError(f"{path} is not a resumable frontier checkpoint")

    level = int(raw_level)
    words = [str(text).strip().upper() for text in raw_words]
    for text in words:
        word = parse_word(text)
        if len(word) != level:
            raise ValueError(f"frontier word {text} in {path} has length {len(word)}, expected {level}")
    return level, words


def checkpoint_or_forbidden_level(data: Dict[str, object]) -> Optional[int]:
    frontier = data.get("frontier")
    raw_level = None
    if isinstance(frontier, dict):
        raw_level = frontier.get("level")
    if raw_level is None:
        raw_level = data.get("frontier_level", data.get("level"))
    if raw_level is None:
        return None
    return int(raw_level)


def load_json_without_frontier_words(path: Path) -> Dict[str, object]:
    """
    Load a checkpoint JSON while replacing the huge frontier.words array by [].
    The level/count metadata, summary, forbidden cache, and orbit records remain.
    """
    if path.stat().st_size == 0:
        raise ValueError(f"{path} is empty")

    words_marker = b'"words":['
    with path.open("rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            marker_pos = mm.find(words_marker)
            if marker_pos < 0:
                return json.loads(mm[:].decode("utf-8"))
            words_start = marker_pos + len(words_marker)
            words_end = mm.find(b"]", words_start)
            if words_end < 0:
                raise ValueError(f"cannot find end of frontier words array in {path}")
            compact = bytes(mm[:words_start]) + b"]" + bytes(mm[words_end + 1 :])
    return json.loads(compact.decode("utf-8"))


def load_checkpoint_frontier_items(path: Path, lazy_map: bool) -> Tuple[int, FrontierContainer]:
    """
    Stream frontier words from a checkpoint directly into compact frontier items.
    This avoids materializing millions of JSON strings during deep resumes.
    """
    data = load_json_without_frontier_words(path)
    frontier_meta = data.get("frontier")
    if isinstance(frontier_meta, dict):
        raw_level = frontier_meta.get("level")
        expected_count = frontier_meta.get("count")
    else:
        raw_level = data.get("frontier_level", data.get("level"))
        expected_count = None
    if raw_level is None:
        raise ValueError(f"{path} is not a resumable frontier checkpoint")
    level = int(raw_level)

    words_marker = b'"words":['
    frontier = empty_frontier_container(level, lazy_map)
    with path.open("rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            marker_pos = mm.find(words_marker)
            if marker_pos < 0:
                _, words = checkpoint_frontier_words(data, path)
                return level, [frontier_item_for_word(parse_word(text), lazy_map=lazy_map) for text in words]
            pos = marker_pos + len(words_marker)
            words_end = mm.find(b"]", pos)
            if words_end < 0:
                raise ValueError(f"cannot find end of frontier words array in {path}")

            while pos < words_end:
                byte = mm[pos]
                if byte in (ord(","), ord(" "), ord("\n"), ord("\r"), ord("\t")):
                    pos += 1
                    continue
                if byte != ord('"'):
                    raise ValueError(f"unexpected byte while reading frontier words in {path}: {byte!r}")
                end_quote = mm.find(b'"', pos + 1, words_end)
                if end_quote < 0:
                    raise ValueError(f"unterminated frontier word in {path}")
                raw = bytes(mm[pos + 1 : end_quote])
                if len(raw) != level:
                    text = raw.decode("utf-8", errors="replace")
                    raise ValueError(f"frontier word {text} in {path} has length {len(raw)}, expected {level}")
                if lazy_map:
                    frontier.append(pack_word_bytes(raw))
                else:
                    frontier.append(frontier_item_for_word(parse_word_bytes(raw), lazy_map=False))
                pos = end_quote + 1

    if expected_count is not None and len(frontier) != int(expected_count):
        raise ValueError(f"frontier count mismatch in {path}: read {len(frontier)}, expected {expected_count}")
    return level, frontier


def _add_int_mapping(target: DefaultDict[int, int], values: object) -> None:
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        target[int(key)] += int(value)


def _seed_stats_from_summary(stats: "SearchStats", summary: object) -> None:
    if not isinstance(summary, dict):
        return

    scalar_fields = (
        "generated_prefixes",
        "prefix_symmetry_skips",
        "forbidden_suffix_prunes",
        "z3_geometric_checks",
        "z3_geometric_cache_hits",
        "z3_geometric_prunes",
        "z3_geometric_sat",
        "z3_geometric_unknown",
        "plucker_lra_checks",
        "plucker_lra_cache_hits",
        "plucker_lra_prunes",
        "plucker_lra_unknown",
        "plucker_lra_chart_checks",
        "plucker_nonprune_cache_hits",
        "plucker_nra_checks",
        "plucker_nra_cache_hits",
        "plucker_nra_prunes",
        "plucker_nra_unknown",
        "plucker_nra_chart_checks",
        "z3_chart_checks",
        "z3_active_chart_states",
        "z3_active_inequalities",
        "z3_reduction_checks",
        "z3_redundant_inequalities",
        "z3_reduction_unknown",
        "forbidden_cyclic_skips",
        "cyclic_adjacent_skips",
        "imprimitive_skips",
        "equivalence_skips",
        "exact_checks",
        "closed",
    )
    for name in scalar_fields:
        if name in summary:
            setattr(stats, name, int(summary[name]))

    _add_int_mapping(stats.generated_by_period, summary.get("generated_by_period"))
    _add_int_mapping(stats.exact_by_period, summary.get("exact_by_period"))
    _add_int_mapping(stats.closed_by_period, summary.get("closed_by_period"))

    reasons = summary.get("rejection_reasons")
    if isinstance(reasons, dict):
        stats.by_reason.update({str(key): int(value) for key, value in reasons.items()})

    reports = summary.get("level_reports")
    if isinstance(reports, dict):
        stats.level_reports.update({int(key): value for key, value in reports.items() if isinstance(value, dict)})

    elapsed = summary.get("elapsed_seconds")
    if elapsed is not None:
        stats.started_at = time.time() - float(elapsed)


def write_search_checkpoint(
    path: Path,
    level: int,
    stats: "SearchStats",
    forbidden: ForbiddenSubwordCache,
    frontier: Sequence[SearchFrontierItem],
    records_by_id: Dict[str, Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    separators = (",", ":")

    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write("{")
        fh.write('"checkpoint_format":')
        json.dump("tetrahedron_exhaustive_frontier_v1", fh, separators=separators)
        fh.write(',"written_at":')
        json.dump(time.strftime("%Y-%m-%dT%H:%M:%S%z"), fh, separators=separators)
        fh.write(',"level":')
        json.dump(level, fh, separators=separators)

        fh.write(',"frontier":{')
        fh.write('"level":')
        json.dump(level, fh, separators=separators)
        fh.write(',"count":')
        json.dump(len(frontier), fh, separators=separators)
        fh.write(',"words":[')
        for i, item in enumerate(frontier):
            if i:
                fh.write(",")
            json.dump(word_name(search_frontier_word(item, level)), fh, separators=separators)
        fh.write('],"note":')
        json.dump(
            "Z3 prefix chart states are not serialized; resuming remains certified but may redo some local solver work.",
            fh,
            separators=separators,
        )
        fh.write("}")

        fh.write(',"summary":')
        json.dump(stats.to_json(forbidden), fh, separators=separators)
        fh.write(',"forbidden_subwords":')
        json.dump(forbidden.to_json(), fh, separators=separators)
        fh.write(',"orbits":')
        json.dump(
            sorted(records_by_id.values(), key=lambda r: (int(r["period"]), str(r["word"]))),
            fh,
            separators=separators,
        )
        fh.write("}\n")

    tmp_path.replace(path)


@dataclass
class SearchStats:
    max_period: int
    equivalence: str
    primitive_only: bool
    use_forbidden_subwords: bool
    use_xyxy_lemma: bool
    use_z3_geometric_prune: bool
    use_plucker_lra_prune: bool
    use_plucker_nra_prune: bool
    use_ordered_z3_prune: bool
    use_z3_reduce_inequalities: bool
    z3_reduction_scope: str
    z3_timeout_ms: int
    z3_min_length: int
    z3_max_length: int
    z3_warn_after_seconds: float
    workers: int
    deep_plucker_start_level: int
    deep_plucker_workers: int
    use_prefix_symmetry: bool
    even_periods_only: bool
    skip_odd_geometric_prune: bool
    checkpoint_even_levels_only: bool
    checkpoint_stride: int
    expand_stride: int
    started_at: float
    generated_prefixes: int = 0
    prefix_symmetry_skips: int = 0
    forbidden_suffix_prunes: int = 0
    z3_geometric_checks: int = 0
    z3_geometric_cache_hits: int = 0
    z3_geometric_prunes: int = 0
    z3_geometric_sat: int = 0
    z3_geometric_unknown: int = 0
    plucker_lra_checks: int = 0
    plucker_lra_cache_hits: int = 0
    plucker_lra_prunes: int = 0
    plucker_lra_unknown: int = 0
    plucker_lra_chart_checks: int = 0
    plucker_nonprune_cache_hits: int = 0
    plucker_nra_checks: int = 0
    plucker_nra_cache_hits: int = 0
    plucker_nra_prunes: int = 0
    plucker_nra_unknown: int = 0
    plucker_nra_chart_checks: int = 0
    z3_chart_checks: int = 0
    z3_active_chart_states: int = 0
    z3_active_inequalities: int = 0
    z3_reduction_checks: int = 0
    z3_redundant_inequalities: int = 0
    z3_reduction_unknown: int = 0
    forbidden_cyclic_skips: int = 0
    cyclic_adjacent_skips: int = 0
    imprimitive_skips: int = 0
    equivalence_skips: int = 0
    exact_checks: int = 0
    closed: int = 0
    by_reason: Counter = None  # type: ignore[assignment]
    generated_by_period: DefaultDict[int, int] = None  # type: ignore[assignment]
    exact_by_period: DefaultDict[int, int] = None  # type: ignore[assignment]
    closed_by_period: DefaultDict[int, int] = None  # type: ignore[assignment]
    level_reports: Dict[int, Dict[str, int]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.by_reason = Counter()
        self.generated_by_period = defaultdict(int)
        self.exact_by_period = defaultdict(int)
        self.closed_by_period = defaultdict(int)
        self.level_reports = {}

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def to_json(self, forbidden: ForbiddenSubwordCache) -> Dict[str, object]:
        return {
            "max_period": self.max_period,
            "equivalence": self.equivalence,
            "primitive_only": self.primitive_only,
            "use_forbidden_subwords": self.use_forbidden_subwords,
            "use_xyxy_lemma": self.use_xyxy_lemma,
            "use_z3_geometric_prune": self.use_z3_geometric_prune,
            "use_plucker_lra_prune": self.use_plucker_lra_prune,
            "use_plucker_nra_prune": self.use_plucker_nra_prune,
            "use_ordered_z3_prune": self.use_ordered_z3_prune,
            "use_z3_reduce_inequalities": self.use_z3_reduce_inequalities,
            "z3_reduction_scope": self.z3_reduction_scope,
            "z3_timeout_ms": self.z3_timeout_ms,
            "z3_min_length": self.z3_min_length,
            "z3_max_length": self.z3_max_length,
            "z3_warn_after_seconds": self.z3_warn_after_seconds,
            "workers": self.workers,
            "deep_plucker_start_level": self.deep_plucker_start_level,
            "deep_plucker_workers": self.deep_plucker_workers,
            "use_prefix_symmetry": self.use_prefix_symmetry,
            "even_periods_only": self.even_periods_only,
            "skip_odd_geometric_prune": self.skip_odd_geometric_prune,
            "checkpoint_even_levels_only": self.checkpoint_even_levels_only,
            "checkpoint_stride": self.checkpoint_stride,
            "expand_stride": self.expand_stride,
            "elapsed_seconds": self.elapsed(),
            "generated_prefixes": self.generated_prefixes,
            "generated_by_period": dict(sorted(self.generated_by_period.items())),
            "level_reports": {str(k): v for k, v in sorted(self.level_reports.items())},
            "prefix_symmetry_skips": self.prefix_symmetry_skips,
            "forbidden_suffix_prunes": self.forbidden_suffix_prunes,
            "z3_geometric_checks": self.z3_geometric_checks,
            "z3_geometric_cache_hits": self.z3_geometric_cache_hits,
            "z3_geometric_prunes": self.z3_geometric_prunes,
            "z3_geometric_sat": self.z3_geometric_sat,
            "z3_geometric_unknown": self.z3_geometric_unknown,
            "plucker_lra_checks": self.plucker_lra_checks,
            "plucker_lra_cache_hits": self.plucker_lra_cache_hits,
            "plucker_lra_prunes": self.plucker_lra_prunes,
            "plucker_lra_unknown": self.plucker_lra_unknown,
            "plucker_lra_chart_checks": self.plucker_lra_chart_checks,
            "plucker_nonprune_cache_hits": self.plucker_nonprune_cache_hits,
            "plucker_nra_checks": self.plucker_nra_checks,
            "plucker_nra_cache_hits": self.plucker_nra_cache_hits,
            "plucker_nra_prunes": self.plucker_nra_prunes,
            "plucker_nra_unknown": self.plucker_nra_unknown,
            "plucker_nra_chart_checks": self.plucker_nra_chart_checks,
            "z3_chart_checks": self.z3_chart_checks,
            "z3_active_chart_states": self.z3_active_chart_states,
            "z3_active_inequalities": self.z3_active_inequalities,
            "z3_reduction_checks": self.z3_reduction_checks,
            "z3_redundant_inequalities": self.z3_redundant_inequalities,
            "z3_reduction_unknown": self.z3_reduction_unknown,
            "forbidden_cyclic_skips": self.forbidden_cyclic_skips,
            "cyclic_adjacent_skips": self.cyclic_adjacent_skips,
            "imprimitive_skips": self.imprimitive_skips,
            "equivalence_skips": self.equivalence_skips,
            "exact_checks": self.exact_checks,
            "exact_by_period": dict(sorted(self.exact_by_period.items())),
            "closed": self.closed,
            "closed_by_period": dict(sorted(self.closed_by_period.items())),
            "rejection_reasons": dict(sorted(self.by_reason.items())),
            "forbidden_subwords": forbidden.to_json(),
        }


def orbit_to_record(orbit: ExactOrbit, equivalence: str) -> Dict[str, object]:
    bary = [barycentric_abcd(p) for p in orbit.points]
    den = 1
    for row in bary:
        for x in row:
            den = math.lcm(den, x.denominator)
    integer_points = [[int(x * den) for x in row] for row in bary]
    length_exact = orbit.total_length_sympy
    return {
        "id": f"p{orbit.period:02d}_{word_name(orbit.word)}",
        "equivalence": equivalence,
        "period": orbit.period,
        "word": word_name(orbit.word),
        "axis_direction": list(orbit.axis_direction),
        "initial_direction": list(orbit.initial_direction),
        "points_xyz_exact": [[frac_to_json(coord) for coord in p] for p in orbit.points],
        "barycentric_denominator": den,
        "barycentric_points": integer_points,
        "length_exact": sp.sstr(length_exact),
        "length_numeric": float(sp.N(length_exact, 30)),
    }


def maybe_print_progress(stats: SearchStats, report_every: int, period: int) -> None:
    if report_every <= 0 or stats.generated_prefixes % report_every:
        return
    print(
        f"[{time.strftime('%H:%M:%S')}] prefixes={stats.generated_prefixes} "
        f"period={period} exact={stats.exact_checks} closed={stats.closed} "
        f"forbidden_prunes={stats.forbidden_suffix_prunes} "
        f"elapsed={stats.elapsed():.1f}s",
        flush=True,
    )


def exhaustive_search(
    max_period: int,
    equivalence: str,
    primitive_only: bool,
    use_forbidden_subwords: bool,
    use_xyxy_lemma: bool,
    use_z3_geometric_prune: bool,
    use_plucker_lra_prune: bool,
    use_plucker_nra_prune: bool,
    use_ordered_z3_prune: bool,
    use_z3_reduce_inequalities: bool,
    z3_reduction_scope: str,
    z3_timeout_ms: int,
    z3_min_length: int,
    z3_max_length: int,
    z3_warn_after_seconds: float,
    workers: int,
    deep_plucker_start_level: int,
    deep_plucker_workers: int,
    use_prefix_symmetry: bool,
    even_periods_only: bool,
    skip_odd_geometric_prune: bool,
    checkpoint_even_levels_only: bool,
    checkpoint_stride: int,
    expand_stride: int,
    report_every: int,
    forbidden_in: Sequence[Path],
    forbidden_out: Optional[Path],
    resume_checkpoint: Optional[Path],
    checkpoint_out: Optional[Path],
    lazy_frontier_maps: bool,
    candidate_chunk_size: int,
) -> Dict[str, object]:
    if max_period < 2:
        raise ValueError("max_period must be at least 2")
    if equivalence not in {"none", "cyclic", "full"}:
        raise ValueError("equivalence must be one of: none, cyclic, full")
    if candidate_chunk_size < 1:
        raise ValueError("candidate_chunk_size must be at least 1")
    if checkpoint_stride < 1:
        raise ValueError("checkpoint_stride must be at least 1")
    if expand_stride not in {1, 2}:
        raise ValueError("expand_stride must be 1 or 2")
    if deep_plucker_start_level < 1:
        raise ValueError("deep_plucker_start_level must be at least 1")
    if deep_plucker_workers < 0:
        raise ValueError("deep_plucker_workers must be nonnegative")

    forbidden = ForbiddenSubwordCache(enabled=use_forbidden_subwords)
    loaded_forbidden = 0
    loaded_forbidden_through = -1
    resume_data: Optional[Dict[str, object]] = None
    all_forbidden_in = list(forbidden_in)
    if resume_checkpoint is not None:
        resume_data = load_json_without_frontier_words(resume_checkpoint)
        if not all_forbidden_in and resume_checkpoint not in all_forbidden_in:
            all_forbidden_in.append(resume_checkpoint)
    if use_forbidden_subwords:
        for path in all_forbidden_in:
            loaded_forbidden += forbidden.load_patterns_from_file(path)
            try:
                source_data = (
                    resume_data
                    if resume_checkpoint is not None and path == resume_checkpoint and resume_data is not None
                    else load_json_without_frontier_words(path)
                )
                source_level = checkpoint_or_forbidden_level(source_data)
                if source_level is not None:
                    loaded_forbidden_through = max(loaded_forbidden_through, source_level)
            except Exception:
                pass
    stats = SearchStats(
        max_period=max_period,
        equivalence=equivalence,
        primitive_only=primitive_only,
        use_forbidden_subwords=use_forbidden_subwords,
        use_xyxy_lemma=use_xyxy_lemma,
        use_z3_geometric_prune=use_z3_geometric_prune,
        use_plucker_lra_prune=use_plucker_lra_prune,
        use_plucker_nra_prune=use_plucker_nra_prune,
        use_ordered_z3_prune=use_ordered_z3_prune,
        use_z3_reduce_inequalities=use_z3_reduce_inequalities,
        z3_reduction_scope=z3_reduction_scope,
        z3_timeout_ms=z3_timeout_ms,
        z3_min_length=z3_min_length,
        z3_max_length=z3_max_length,
        z3_warn_after_seconds=z3_warn_after_seconds,
        workers=workers,
        deep_plucker_start_level=deep_plucker_start_level,
        deep_plucker_workers=deep_plucker_workers,
        use_prefix_symmetry=use_prefix_symmetry,
        even_periods_only=even_periods_only,
        skip_odd_geometric_prune=skip_odd_geometric_prune,
        checkpoint_even_levels_only=checkpoint_even_levels_only,
        checkpoint_stride=checkpoint_stride,
        expand_stride=expand_stride,
        started_at=time.time(),
    )
    seen_by_period: Dict[int, set[int]] = {m: set() for m in range(2, max_period + 1)}
    records_by_id: Dict[str, Dict[str, object]] = {}
    loaded_previous_orbits = 0
    z3_validity_cache: Dict[Tuple[int, ...], GeometricValidityResult] = {}
    plucker_validity_cache: Dict[Tuple[int, ...], GeometricValidityResult] = {}
    plucker_nra_validity_cache: Dict[Tuple[int, ...], GeometricValidityResult] = {}
    plucker_nonprune_cache_by_length: DefaultDict[int, set[int]] = defaultdict(set)
    plucker_worker_runtime_cap = workers

    def effective_plucker_workers(target_period: int, pending_count: int) -> int:
        cap = max(1, workers)
        if deep_plucker_workers and target_period >= deep_plucker_start_level:
            cap = min(cap, deep_plucker_workers)
        cap = min(cap, plucker_worker_runtime_cap, max(1, pending_count))
        return max(1, cap)

    def plucker_pool_kwargs(target_period: int) -> Dict[str, object]:
        if not deep_plucker_workers or target_period < deep_plucker_start_level:
            return {}
        try:
            return {"mp_context": mp.get_context("spawn")}
        except ValueError:
            return {}

    def plucker_in_flight_limit(target_period: int, pool_workers: int) -> int:
        factor = 2 if target_period >= deep_plucker_start_level else 4
        return max(1, pool_workers * factor)

    def run_plucker_pending(
        target_period: int,
        pending: Dict[Tuple[int, ...], Tuple[int, ...]],
        record_result: Callable[
            [Tuple[int, ...], GeometricValidityResult, Optional[GeometricValidityResult]],
            None,
        ],
    ) -> None:
        nonlocal plucker_worker_runtime_cap

        plucker_started = time.perf_counter()
        last_plucker_progress = plucker_started
        completed_plucker = 0
        completed_plucker_keys: set[Tuple[int, ...]] = set()
        remaining_items = list(pending.items())

        while remaining_items:
            pool_workers = effective_plucker_workers(target_period, len(remaining_items))
            if pool_workers <= 1:
                remaining_count = len(remaining_items)
                print(
                    f"[level {target_period:2d}] running {remaining_count} Plucker checks serially",
                    flush=True,
                )
                for index, (key, word) in enumerate(remaining_items, start=1):
                    _, lra_result, nra_result = plucker_prune_worker(
                        (
                            word,
                            z3_timeout_ms,
                            use_plucker_lra_prune,
                            use_plucker_nra_prune,
                            z3_warn_after_seconds,
                        )
                    )
                    completed_plucker += 1
                    completed_plucker_keys.add(key)
                    record_result(key, lra_result, nra_result)
                    now = time.perf_counter()
                    if now - last_plucker_progress >= 30:
                        print(
                            f"[level {target_period:2d}] plucker serial="
                            f"{index}/{remaining_count} total={completed_plucker}/{len(pending)} "
                            f"elapsed={now - plucker_started:.1f}s",
                            flush=True,
                        )
                        last_plucker_progress = now
                return

            max_in_flight = plucker_in_flight_limit(target_period, pool_workers)
            pending_iter = iter(remaining_items)
            try:
                with ProcessPoolExecutor(
                    max_workers=pool_workers,
                    **plucker_pool_kwargs(target_period),
                ) as executor:
                    futures: Dict[object, Tuple[int, ...]] = {}

                    def submit_more() -> None:
                        while len(futures) < max_in_flight:
                            try:
                                key, word = next(pending_iter)
                            except StopIteration:
                                break
                            futures[
                                executor.submit(
                                    plucker_prune_worker,
                                    (
                                        word,
                                        z3_timeout_ms,
                                        use_plucker_lra_prune,
                                        use_plucker_nra_prune,
                                        z3_warn_after_seconds,
                                    ),
                                )
                            ] = key

                    submit_more()
                    while futures:
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            key = futures.pop(future)
                            _, lra_result, nra_result = future.result()
                            completed_plucker += 1
                            completed_plucker_keys.add(key)
                            record_result(key, lra_result, nra_result)
                        submit_more()
                        now = time.perf_counter()
                        if now - last_plucker_progress >= 30:
                            print(
                                f"[level {target_period:2d}] plucker completed="
                                f"{completed_plucker}/{len(pending)} "
                                f"elapsed={now - plucker_started:.1f}s "
                                f"in_flight={len(futures)} workers={pool_workers}",
                                flush=True,
                            )
                            last_plucker_progress = now
                return
            except BrokenProcessPool as exc:
                remaining_items = [(key, word) for key, word in pending.items() if key not in completed_plucker_keys]
                next_cap = max(1, pool_workers // 2)
                if next_cap < plucker_worker_runtime_cap:
                    plucker_worker_runtime_cap = next_cap
                print(
                    f"[level {target_period:2d}] Plucker worker pool broke after "
                    f"{completed_plucker}/{len(pending)} checks ({exc}); "
                    f"retrying {len(remaining_items)} checks with at most "
                    f"{effective_plucker_workers(target_period, len(remaining_items))} workers",
                    flush=True,
                )

    def check_complete_word(word: Tuple[int, ...], M: MatI, b: VecI, q: int) -> bool:
        period = len(word)
        if even_periods_only and period % 2:
            return False
        if word[0] == word[-1]:
            stats.cyclic_adjacent_skips += 1
            return False
        if forbidden.hits_cyclic_word(word):
            stats.forbidden_cyclic_skips += 1
            return False
        if primitive_only and not primitive_word(word):
            stats.imprimitive_skips += 1
            return False
        if equivalence == "full" and canonical_relabel_linear(word) != word:
            stats.equivalence_skips += 1
            return False

        canon = canonical_by_mode(word, equivalence)
        canon_key = pack_word(canon)
        if canon_key in seen_by_period[period]:
            stats.equivalence_skips += 1
            return False
        seen_by_period[period].add(canon_key)

        stats.exact_checks += 1
        stats.exact_by_period[period] += 1
        result = verify_word_exact(word, M, b, q)
        stats.by_reason[result.reason] += 1
        if result.orbit is None:
            return False

        orbit = result.orbit
        if orbit.word != canon:
            canonical_result = verify_word_exact(canon)
            if canonical_result.orbit is not None:
                orbit = canonical_result.orbit
        record = orbit_to_record(orbit, equivalence)
        if record["id"] not in records_by_id:
            records_by_id[str(record["id"])] = record
            stats.closed += 1
            stats.closed_by_period[period] += 1
            print(
                f"[hit] period {period:2d} word={record['word']} length={record['length_exact']}",
                flush=True,
            )
        return True

    def write_level_outputs(level: int, frontier_items: List[SearchFrontierItem]) -> None:
        if checkpoint_even_levels_only and even_periods_only and level % 2:
            if forbidden_out is not None or checkpoint_out is not None:
                print(f"[level {level:2d}] skipped odd-level checkpoint", flush=True)
            return
        if checkpoint_stride > 1 and (level - checkpoint_anchor_level) % checkpoint_stride:
            if forbidden_out is not None or checkpoint_out is not None:
                next_saved = level + (checkpoint_stride - ((level - checkpoint_anchor_level) % checkpoint_stride))
                print(
                    f"[level {level:2d}] skipped checkpoint "
                    f"(stride={checkpoint_stride}, next saved level {next_saved})",
                    flush=True,
                )
            return
        if forbidden_out is not None:
            forbidden.write_patterns_to_file(forbidden_out, level, stats)
        if checkpoint_out is not None:
            write_search_checkpoint(checkpoint_out, level, stats, forbidden, frontier_items, records_by_id)
        return False

    frontier = empty_frontier_container(0, lazy_frontier_maps)
    frontier.append(frontier_item_for_word(tuple(), lazy_map=lazy_frontier_maps))
    start_period = 0
    if resume_checkpoint is not None:
        if resume_data is None:
            resume_data = load_json_without_frontier_words(resume_checkpoint)
        _seed_stats_from_summary(stats, resume_data.get("summary"))
        start_period, frontier = load_checkpoint_frontier_items(resume_checkpoint, lazy_frontier_maps)

        raw_orbits = resume_data.get("orbits", [])
        if isinstance(raw_orbits, list):
            for record in raw_orbits:
                if not isinstance(record, dict) or "id" not in record:
                    continue
                record_id = str(record["id"])
                if record_id not in records_by_id:
                    records_by_id[record_id] = dict(record)
                    loaded_previous_orbits += 1

        print(
            f"[resume] loaded level {start_period} frontier with {len(frontier)} words "
            f"and {loaded_previous_orbits} prior orbit records from {resume_checkpoint}",
            flush=True,
        )
        if lazy_frontier_maps:
            print(
                "[resume] lazy frontier maps enabled; packed words are kept in memory "
                "and affine maps will be recomputed as needed",
                flush=True,
            )
        if isinstance(resume_data.get("frontier"), dict):
            resume_data["frontier"]["words"] = []
        resume_data = None

    checkpoint_anchor_level = start_period
    completed_level = start_period
    skipped_levels: set[int] = set()

    for next_period in range(start_period + 1, max_period + 1):
        if next_period in skipped_levels:
            continue
        next_frontier = empty_frontier_container(next_period, lazy_frontier_maps)
        level_checked = 0
        level_invalid = 0
        learned_forbidden_this_level = False
        geometric_prune_this_level = use_z3_geometric_prune and not (
            skip_odd_geometric_prune and even_periods_only and next_period % 2
        )
        if use_z3_geometric_prune and not geometric_prune_this_level:
            print(
                f"[level {next_period:2d}] skipping geometric pruning at odd level",
                flush=True,
            )
        packed_skip_geometry_mode = lazy_frontier_maps and not geometric_prune_this_level
        parallel_plucker_only = packed_skip_geometry_mode or (
            workers > 1
            and geometric_prune_this_level
            and not use_ordered_z3_prune
            and (use_plucker_lra_prune or use_plucker_nra_prune)
        )

        if (
            expand_stride == 2
            and parallel_plucker_only
            and lazy_frontier_maps
            and even_periods_only
            and not use_ordered_z3_prune
            and next_period + 1 <= max_period
            and (next_period - checkpoint_anchor_level) % 2 == 1
        ):
            mid_period = next_period
            out_period = next_period + 1
            pair_started = time.perf_counter()
            last_pair_progress = pair_started
            final_frontier = packed_frontier_container(out_period)
            mid_chunk: List[int] = []
            out_chunk: List[int] = []
            mid_valid_total = 0
            frontier_count = len(frontier)
            built_prefixes = 0
            level_state: Dict[int, Dict[str, object]] = {
                mid_period: {
                    "checked": 0,
                    "invalid": 0,
                    "built": 0,
                    "processed": 0,
                    "learned": False,
                    "replayed_notice": False,
                },
                out_period: {
                    "checked": 0,
                    "invalid": 0,
                    "built": 0,
                    "processed": 0,
                    "learned": False,
                    "replayed_notice": False,
                },
            }

            print(
                f"[levels {mid_period:2d}->{out_period:2d}] streaming two-level expansion "
                f"from {frontier_count} level-{mid_period - 1} prefixes",
                flush=True,
            )

            def geometric_enabled_for(level: int) -> bool:
                return use_z3_geometric_prune and not (
                    skip_odd_geometric_prune and even_periods_only and level % 2
                )

            def bump_state(level: int, key: str, amount: int = 1) -> None:
                level_state[level][key] = int(level_state[level][key]) + amount

            def mark_learned(level: int) -> None:
                level_state[level]["learned"] = True

            def prune_packed_chunk(candidates: List[int], target_period: int) -> List[int]:
                if not candidates:
                    return []

                state = level_state[target_period]
                geometric_prune_target = geometric_enabled_for(target_period)
                pending: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
                nonprune_cache = plucker_nonprune_cache_by_length[target_period]

                if geometric_prune_target:
                    if target_period <= loaded_forbidden_through:
                        if not bool(state["replayed_notice"]):
                            print(
                                f"[level {target_period:2d}] replaying from loaded forbidden cache "
                                f"(complete through level {loaded_forbidden_through}); "
                                "skipping Plucker recomputation",
                                flush=True,
                            )
                            state["replayed_notice"] = True
                    else:
                        for child_code in candidates:
                            child = unpack_word(child_code, target_period)
                            z3_length_ok = target_period >= z3_min_length and (
                                z3_max_length <= 0 or target_period <= z3_max_length
                            )
                            if not z3_length_ok:
                                continue
                            cache_key = canonical_forbidden_subword(child)
                            if pack_word(cache_key) in nonprune_cache:
                                stats.plucker_nonprune_cache_hits += 1
                                continue
                            lra_cached = plucker_validity_cache.get(cache_key)
                            nra_cached = plucker_nra_validity_cache.get(cache_key)
                            needs_lra = use_plucker_lra_prune and lra_cached is None
                            needs_nra = (
                                use_plucker_nra_prune
                                and (lra_cached is None or lra_cached.status != "unsat")
                                and nra_cached is None
                            )
                            if needs_lra or needs_nra:
                                pending.setdefault(cache_key, child)

                if pending:
                    print(
                        f"[level {target_period:2d}] plucker checks={len(pending)} "
                        f"chunk_candidates={len(candidates)} "
                        f"built_candidates={int(state['built'])} "
                        f"nonprune_cache={len(nonprune_cache)} "
                        f"nonprune_hits={stats.plucker_nonprune_cache_hits} "
                        f"workers={effective_plucker_workers(target_period, len(pending))}/{workers}",
                        flush=True,
                    )
                    def record_plucker_result(
                        key: Tuple[int, ...],
                        lra_result: GeometricValidityResult,
                        nra_result: Optional[GeometricValidityResult],
                    ) -> None:
                        if use_plucker_lra_prune:
                            plucker_validity_cache[key] = lra_result
                            stats.plucker_lra_checks += 1
                            stats.plucker_lra_chart_checks += lra_result.linear_chart_checks
                        if nra_result is not None:
                            plucker_nra_validity_cache[key] = nra_result
                            stats.plucker_nra_checks += 1
                            stats.plucker_nra_chart_checks += nra_result.linear_chart_checks
                        if (
                            (not use_plucker_lra_prune or lra_result.status != "unsat")
                            and (nra_result is None or nra_result.status != "unsat")
                        ):
                            nonprune_cache.add(pack_word(key))

                    run_plucker_pending(target_period, pending, record_plucker_result)

                survivors: List[int] = []
                for child_code in candidates:
                    z3_length_ok = target_period >= z3_min_length and (
                        z3_max_length <= 0 or target_period <= z3_max_length
                    )
                    if geometric_prune_target and z3_length_ok:
                        if target_period <= loaded_forbidden_through:
                            if forbidden.hits_linear_suffix_packed(child_code, target_period):
                                stats.forbidden_suffix_prunes += 1
                                bump_state(target_period, "invalid")
                                continue
                            survivors.append(child_code)
                            continue

                        child = unpack_word(child_code, target_period)
                        cache_key = canonical_forbidden_subword(child)
                        if pack_word(cache_key) in nonprune_cache:
                            survivors.append(child_code)
                            continue
                        if use_plucker_lra_prune:
                            plucker_validity = plucker_validity_cache.get(cache_key)
                            if plucker_validity is not None:
                                if cache_key not in pending:
                                    stats.plucker_lra_cache_hits += 1
                                if plucker_validity.status == "unsat":
                                    stats.plucker_lra_prunes += 1
                                    stats.z3_geometric_prunes += 1
                                    if forbidden.register(child):
                                        mark_learned(target_period)
                                    bump_state(target_period, "invalid")
                                    continue
                                stats.plucker_lra_unknown += 1
                        if use_plucker_nra_prune:
                            plucker_nra_validity = plucker_nra_validity_cache.get(cache_key)
                            if plucker_nra_validity is not None:
                                if cache_key not in pending:
                                    stats.plucker_nra_cache_hits += 1
                                if plucker_nra_validity.status == "unsat":
                                    stats.plucker_nra_prunes += 1
                                    stats.z3_geometric_prunes += 1
                                    if forbidden.register(child):
                                        mark_learned(target_period)
                                    bump_state(target_period, "invalid")
                                    continue
                                stats.plucker_nra_unknown += 1

                    survivors.append(child_code)

                bump_state(target_period, "processed", len(candidates))
                plucker_validity_cache.clear()
                plucker_nra_validity_cache.clear()
                return survivors

            def flush_out_chunk() -> None:
                if not out_chunk:
                    return
                survivors = prune_packed_chunk(out_chunk, out_period)
                final_frontier.extend(survivors)
                out_chunk.clear()

            def process_mid_chunk() -> None:
                nonlocal mid_valid_total
                if not mid_chunk:
                    return
                mid_survivors = prune_packed_chunk(mid_chunk, mid_period)
                mid_chunk.clear()

                if bool(level_state[mid_period]["learned"]):
                    retained_mid: List[int] = []
                    for mid_code in mid_survivors:
                        if forbidden.hits_linear_subword_packed(mid_code, mid_period):
                            stats.forbidden_suffix_prunes += 1
                            bump_state(mid_period, "invalid")
                        else:
                            retained_mid.append(mid_code)
                    mid_survivors = retained_mid

                mid_valid_total += len(mid_survivors)
                stats.generated_prefixes += len(mid_survivors)
                stats.generated_by_period[mid_period] += len(mid_survivors)

                for mid_code in mid_survivors:
                    last_face = mid_code & 3
                    local_seen: set[Tuple[int, ...]] = set()
                    for face in range(4):
                        if face == last_face:
                            continue
                        child_code = append_packed_face(mid_code, face)
                        if use_prefix_symmetry and equivalence == "full":
                            prefix = unpack_word(mid_code, mid_period)
                            child = unpack_word(child_code, out_period)
                            local_key = canonical_child_under_stabilizer(prefix, child)
                            if local_key in local_seen:
                                stats.prefix_symmetry_skips += 1
                                continue
                            local_seen.add(local_key)
                        bump_state(out_period, "checked")
                        if use_xyxy_lemma:
                            child = unpack_word(child_code, out_period)
                            if forbidden.register_xyxy_suffix_if_present(child):
                                mark_learned(out_period)
                        if forbidden.hits_linear_suffix_packed(child_code, out_period):
                            stats.forbidden_suffix_prunes += 1
                            bump_state(out_period, "invalid")
                            continue
                        out_chunk.append(child_code)
                        bump_state(out_period, "built")
                        if len(out_chunk) >= candidate_chunk_size:
                            flush_out_chunk()

            for prefix_item in frontier:
                built_prefixes += 1
                prefix_code = int(prefix_item)
                last_face = prefix_code & 3 if mid_period > 1 else None
                local_seen: set[Tuple[int, ...]] = set()
                for face in range(4):
                    if last_face is not None and face == last_face:
                        continue
                    child_code = append_packed_face(prefix_code, face)
                    if use_prefix_symmetry and equivalence == "full":
                        prefix = unpack_word(prefix_code, mid_period - 1)
                        child = unpack_word(child_code, mid_period)
                        local_key = canonical_child_under_stabilizer(prefix, child)
                        if local_key in local_seen:
                            stats.prefix_symmetry_skips += 1
                            continue
                        local_seen.add(local_key)

                    bump_state(mid_period, "checked")
                    if use_xyxy_lemma:
                        child = unpack_word(child_code, mid_period)
                        if forbidden.register_xyxy_suffix_if_present(child):
                            mark_learned(mid_period)
                    if forbidden.hits_linear_suffix_packed(child_code, mid_period):
                        stats.forbidden_suffix_prunes += 1
                        bump_state(mid_period, "invalid")
                        continue
                    mid_chunk.append(child_code)
                    bump_state(mid_period, "built")
                    if len(mid_chunk) >= candidate_chunk_size:
                        process_mid_chunk()

                now = time.perf_counter()
                if now - last_pair_progress >= 30:
                    print(
                        f"[levels {mid_period:2d}->{out_period:2d}] prefixes="
                        f"{built_prefixes}/{frontier_count} "
                        f"mid_checked={int(level_state[mid_period]['checked'])} "
                        f"mid_valid={mid_valid_total} "
                        f"out_checked={int(level_state[out_period]['checked'])} "
                        f"out_kept={len(final_frontier)} "
                        f"out_chunk={len(out_chunk)} "
                        f"canon_cache={canonical_forbidden_subword.cache_info().currsize} "
                        f"elapsed={now - pair_started:.1f}s",
                        flush=True,
                    )
                    last_pair_progress = now

            process_mid_chunk()
            flush_out_chunk()

            if bool(level_state[mid_period]["learned"]) or bool(level_state[out_period]["learned"]):
                retained_frontier = packed_frontier_container(out_period)
                for item in final_frontier:
                    code = int(item)
                    if forbidden.hits_linear_subword_packed(code, out_period):
                        stats.forbidden_suffix_prunes += 1
                        bump_state(out_period, "invalid")
                    else:
                        retained_frontier.append(code)
                final_frontier = retained_frontier

            frontier = final_frontier
            level_hits = 0
            if out_period >= 2:
                exact_started = time.perf_counter()
                last_exact_progress = exact_started
                exact_seen = 0
                level_valid = len(frontier)
                for item in frontier:
                    exact_seen += 1
                    stats.generated_prefixes += 1
                    stats.generated_by_period[out_period] += 1
                    if (
                        equivalence == "full"
                        and isinstance(item, int)
                        and not packed_can_be_full_canonical(item, out_period)
                    ):
                        stats.equivalence_skips += 1
                        continue
                    prefix, M, b, q, _ = search_frontier_components(item, out_period)
                    exact_M, exact_b, exact_q = materialize_return_map(prefix, M, b, q)
                    if check_complete_word(prefix, exact_M, exact_b, exact_q):
                        level_hits += 1
                    maybe_print_progress(stats, report_every, out_period)
                    now = time.perf_counter()
                    if now - last_exact_progress >= 30:
                        print(
                            f"[level {out_period:2d}] exact candidates="
                            f"{exact_seen}/{level_valid} hits={level_hits} "
                            f"exact_checks={stats.exact_by_period[out_period]} "
                            f"elapsed={now - exact_started:.1f}s",
                            flush=True,
                        )
                        last_exact_progress = now

            stats.level_reports[mid_period] = {
                "checked": int(level_state[mid_period]["checked"]),
                "valid": int(mid_valid_total),
                "invalid": int(level_state[mid_period]["invalid"]),
                "hits": 0,
            }
            stats.level_reports[out_period] = {
                "checked": int(level_state[out_period]["checked"]),
                "valid": int(len(frontier)),
                "invalid": int(level_state[out_period]["invalid"]),
                "hits": int(level_hits),
            }
            completed_level = out_period
            write_level_outputs(out_period, frontier)
            print(
                f"[level {mid_period:2d}] checked={int(level_state[mid_period]['checked'])} "
                f"valid={mid_valid_total} invalid={int(level_state[mid_period]['invalid'])} hits=0",
                flush=True,
            )
            print(
                f"[level {out_period:2d}] checked={int(level_state[out_period]['checked'])} "
                f"valid={len(frontier)} invalid={int(level_state[out_period]['invalid'])} hits={level_hits}",
                flush=True,
            )

            skipped_levels.add(out_period)
            if not frontier:
                break
            continue

        if parallel_plucker_only:
            candidate_chunk: List[SearchFrontierItem] = []
            build_started = time.perf_counter()
            last_build_progress = build_started
            built_prefixes = 0
            built_candidates_total = 0
            processed_candidates_total = 0
            frontier_count = len(frontier)
            replayed_forbidden_level_notice = False

            def flush_candidate_chunk() -> None:
                nonlocal level_invalid, learned_forbidden_this_level, processed_candidates_total
                nonlocal replayed_forbidden_level_notice
                if not candidate_chunk:
                    return

                pending: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
                nonprune_cache = plucker_nonprune_cache_by_length[next_period]
                if geometric_prune_this_level:
                    if next_period <= loaded_forbidden_through:
                        if not replayed_forbidden_level_notice:
                            print(
                                f"[level {next_period:2d}] replaying from loaded forbidden cache "
                                f"(complete through level {loaded_forbidden_through}); "
                                "skipping Plucker recomputation",
                                flush=True,
                            )
                            replayed_forbidden_level_notice = True
                    else:
                        for candidate in candidate_chunk:
                            child = search_frontier_word(candidate, next_period)
                            z3_length_ok = len(child) >= z3_min_length and (z3_max_length <= 0 or len(child) <= z3_max_length)
                            if not z3_length_ok:
                                continue
                            cache_key = canonical_forbidden_subword(child)
                            if pack_word(cache_key) in nonprune_cache:
                                stats.plucker_nonprune_cache_hits += 1
                                continue
                            lra_cached = plucker_validity_cache.get(cache_key)
                            nra_cached = plucker_nra_validity_cache.get(cache_key)
                            needs_lra = use_plucker_lra_prune and lra_cached is None
                            needs_nra = (
                                use_plucker_nra_prune
                                and (lra_cached is None or lra_cached.status != "unsat")
                                and nra_cached is None
                            )
                            if needs_lra or needs_nra:
                                pending.setdefault(cache_key, child)

                if pending:
                    print(
                        f"[level {next_period:2d}] plucker checks={len(pending)} "
                        f"chunk_candidates={len(candidate_chunk)} "
                        f"built_candidates={built_candidates_total} "
                        f"nonprune_cache={len(nonprune_cache)} "
                        f"nonprune_hits={stats.plucker_nonprune_cache_hits} "
                        f"workers={effective_plucker_workers(next_period, len(pending))}/{workers}",
                        flush=True,
                    )
                    def record_plucker_result(
                        key: Tuple[int, ...],
                        lra_result: GeometricValidityResult,
                        nra_result: Optional[GeometricValidityResult],
                    ) -> None:
                        if use_plucker_lra_prune:
                            plucker_validity_cache[key] = lra_result
                            stats.plucker_lra_checks += 1
                            stats.plucker_lra_chart_checks += lra_result.linear_chart_checks
                        if nra_result is not None:
                            plucker_nra_validity_cache[key] = nra_result
                            stats.plucker_nra_checks += 1
                            stats.plucker_nra_chart_checks += nra_result.linear_chart_checks
                        if (
                            (not use_plucker_lra_prune or lra_result.status != "unsat")
                            and (nra_result is None or nra_result.status != "unsat")
                        ):
                            nonprune_cache.add(pack_word(key))

                    run_plucker_pending(next_period, pending, record_plucker_result)

                for candidate in candidate_chunk:
                    if isinstance(candidate, int):
                        child_code = candidate
                        z3_length_ok = next_period >= z3_min_length and (
                            z3_max_length <= 0 or next_period <= z3_max_length
                        )
                    else:
                        child_code = None
                        child = candidate[0]
                        z3_length_ok = len(child) >= z3_min_length and (z3_max_length <= 0 or len(child) <= z3_max_length)
                    if geometric_prune_this_level and z3_length_ok:
                        if child_code is not None and next_period <= loaded_forbidden_through:
                            if forbidden.hits_linear_suffix_packed(child_code, next_period):
                                stats.forbidden_suffix_prunes += 1
                                level_invalid += 1
                                continue
                            next_frontier.append(candidate)
                            continue

                        if child_code is not None:
                            child = unpack_word(child_code, next_period)
                        cache_key = canonical_forbidden_subword(child)
                        if pack_word(cache_key) in nonprune_cache:
                            next_frontier.append(candidate)
                            continue
                        if use_plucker_lra_prune:
                            plucker_validity = plucker_validity_cache.get(cache_key)
                            if plucker_validity is not None:
                                if cache_key not in pending:
                                    stats.plucker_lra_cache_hits += 1
                                if plucker_validity.status == "unsat":
                                    stats.plucker_lra_prunes += 1
                                    stats.z3_geometric_prunes += 1
                                    learned_forbidden_this_level |= forbidden.register(child)
                                    level_invalid += 1
                                    continue
                                stats.plucker_lra_unknown += 1
                        if use_plucker_nra_prune:
                            plucker_nra_validity = plucker_nra_validity_cache.get(cache_key)
                            if plucker_nra_validity is not None:
                                if cache_key not in pending:
                                    stats.plucker_nra_cache_hits += 1
                                if plucker_nra_validity.status == "unsat":
                                    stats.plucker_nra_prunes += 1
                                    stats.z3_geometric_prunes += 1
                                    learned_forbidden_this_level |= forbidden.register(child)
                                    level_invalid += 1
                                    continue
                                stats.plucker_nra_unknown += 1

                    next_frontier.append(candidate)

                processed_candidates_total += len(candidate_chunk)
                candidate_chunk.clear()
                # Keep these caches local to the chunk.  Certified UNSAT words
                # have already been moved into the forbidden-subword cache; the
                # remaining UNKNOWN/SAT Plucker records are only a speed hint and
                # become a deep-search memory leak if retained for every word.
                plucker_validity_cache.clear()
                plucker_nra_validity_cache.clear()

            for prefix_item in frontier:
                if lazy_frontier_maps:
                    prefix_code = int(prefix_item)
                    last_face = prefix_code & 3 if next_period > 1 else None
                    prefix: Tuple[int, ...] = ()
                    M = b = q = z3_prefix_state = None
                    parent_M = parent_b = parent_q = None
                else:
                    prefix, M, b, q, z3_prefix_state = search_frontier_components(prefix_item, next_period - 1)
                    parent_M, parent_b, parent_q = materialize_return_map(prefix, M, b, q)
                built_prefixes += 1
                local_seen: set[Tuple[int, ...]] = set()
                for face in range(4):
                    if lazy_frontier_maps:
                        if last_face is not None and face == last_face:
                            continue
                        child_code = append_packed_face(prefix_code, face)
                        child: Tuple[int, ...] = ()
                    else:
                        if prefix and face == prefix[-1]:
                            continue
                        child = prefix + (face,)

                    if use_prefix_symmetry and equivalence == "full":
                        if lazy_frontier_maps:
                            prefix = unpack_word(prefix_code, next_period - 1)
                            child = unpack_word(child_code, next_period)
                        local_key = canonical_child_under_stabilizer(prefix, child)
                        if local_key in local_seen:
                            stats.prefix_symmetry_skips += 1
                            continue
                        local_seen.add(local_key)

                    level_checked += 1
                    if use_xyxy_lemma:
                        if lazy_frontier_maps and not child:
                            child = unpack_word(child_code, next_period)
                        learned_forbidden_this_level |= forbidden.register_xyxy_suffix_if_present(child)
                    if lazy_frontier_maps:
                        forbidden_hit = forbidden.hits_linear_suffix_packed(child_code, next_period)
                    else:
                        forbidden_hit = forbidden.hits_linear_suffix(child)
                    if forbidden_hit:
                        stats.forbidden_suffix_prunes += 1
                        level_invalid += 1
                        continue

                    if lazy_frontier_maps:
                        candidate_chunk.append(child_code)
                    else:
                        next_M, next_b, next_q = append_face_to_return_map(parent_M, parent_b, parent_q, face)
                        candidate_chunk.append((child, next_M, next_b, next_q, z3_prefix_state))
                    built_candidates_total += 1
                    if len(candidate_chunk) >= candidate_chunk_size:
                        flush_candidate_chunk()
                now = time.perf_counter()
                if now - last_build_progress >= 30:
                    print(
                        f"[level {next_period:2d}] building candidates "
                        f"prefixes={built_prefixes}/{frontier_count} "
                        f"checked={level_checked} invalid={level_invalid} "
                        f"built_candidates={built_candidates_total} "
                        f"processed_candidates={processed_candidates_total} "
                        f"chunk_candidates={len(candidate_chunk)} "
                        f"kept={len(next_frontier)} "
                        f"elapsed={now - build_started:.1f}s",
                        flush=True,
                    )
                    last_build_progress = now

            flush_candidate_chunk()

            if learned_forbidden_this_level:
                retained_frontier = empty_frontier_container(next_period, lazy_frontier_maps)
                for item in next_frontier:
                    if forbidden.hits_linear_subword(search_frontier_word(item, next_period)):
                        stats.forbidden_suffix_prunes += 1
                        level_invalid += 1
                    else:
                        retained_frontier.append(item)
                next_frontier = retained_frontier

            frontier = next_frontier
            level_valid = len(frontier)

            level_hits = 0
            if next_period >= 2:
                exact_started = time.perf_counter()
                last_exact_progress = exact_started
                exact_seen = 0
                for item in frontier:
                    exact_seen += 1
                    stats.generated_prefixes += 1
                    stats.generated_by_period[next_period] += 1
                    if even_periods_only and next_period % 2:
                        maybe_print_progress(stats, report_every, next_period)
                        continue
                    if (
                        equivalence == "full"
                        and isinstance(item, int)
                        and not packed_can_be_full_canonical(item, next_period)
                    ):
                        stats.equivalence_skips += 1
                        continue
                    prefix, M, b, q, _ = search_frontier_components(item, next_period)
                    exact_M, exact_b, exact_q = materialize_return_map(prefix, M, b, q)
                    if check_complete_word(prefix, exact_M, exact_b, exact_q):
                        level_hits += 1
                    maybe_print_progress(stats, report_every, next_period)
                    now = time.perf_counter()
                    if now - last_exact_progress >= 30:
                        print(
                            f"[level {next_period:2d}] exact candidates="
                            f"{exact_seen}/{level_valid} hits={level_hits} "
                            f"exact_checks={stats.exact_by_period[next_period]} "
                            f"elapsed={now - exact_started:.1f}s",
                            flush=True,
                        )
                        last_exact_progress = now

            stats.level_reports[next_period] = {
                "checked": int(level_checked),
                "valid": int(level_valid),
                "invalid": int(level_invalid),
                "hits": int(level_hits),
            }
            completed_level = next_period
            write_level_outputs(next_period, frontier)
            print(
                f"[level {next_period:2d}] checked={level_checked} "
                f"valid={level_valid} invalid={level_invalid} hits={level_hits}",
                flush=True,
            )

            if not frontier:
                break
            continue

        build_started = time.perf_counter()
        last_build_progress = build_started
        built_prefixes = 0
        frontier_count = len(frontier)

        for prefix_item in frontier:
            built_prefixes += 1
            prefix, M, b, q, z3_prefix_state = search_frontier_components(prefix_item, next_period - 1)
            if lazy_frontier_maps:
                parent_M = parent_b = parent_q = None
            else:
                parent_M, parent_b, parent_q = materialize_return_map(prefix, M, b, q)
            local_seen: set[Tuple[int, ...]] = set()
            for face in range(4):
                if prefix and face == prefix[-1]:
                    continue
                child = prefix + (face,)

                if use_prefix_symmetry and equivalence == "full":
                    local_key = canonical_child_under_stabilizer(prefix, child)
                    if local_key in local_seen:
                        stats.prefix_symmetry_skips += 1
                        continue
                    local_seen.add(local_key)

                level_checked += 1
                if use_xyxy_lemma:
                    learned_forbidden_this_level |= forbidden.register_xyxy_suffix_if_present(child)
                if forbidden.hits_linear_suffix(child):
                    stats.forbidden_suffix_prunes += 1
                    level_invalid += 1
                    continue

                z3_length_ok = len(child) >= z3_min_length and (z3_max_length <= 0 or len(child) <= z3_max_length)
                child_z3_state: Optional[Z3PrefixState] = None
                if geometric_prune_this_level and use_plucker_lra_prune and z3_length_ok:
                    cache_key = canonical_forbidden_subword(child)
                    plucker_validity = plucker_validity_cache.get(cache_key)
                    if plucker_validity is None:
                        stats.plucker_lra_checks += 1
                        plucker_validity = certify_stack_transversal_plucker_lra(
                            child,
                            timeout_ms=z3_timeout_ms,
                            warn_after_seconds=z3_warn_after_seconds,
                        )
                        plucker_validity_cache[cache_key] = plucker_validity
                        stats.plucker_lra_chart_checks += plucker_validity.linear_chart_checks
                    else:
                        stats.plucker_lra_cache_hits += 1
                    if plucker_validity.status == "unsat":
                        stats.plucker_lra_prunes += 1
                        stats.z3_geometric_prunes += 1
                        learned_forbidden_this_level |= forbidden.register(child)
                        level_invalid += 1
                        continue
                    stats.plucker_lra_unknown += 1

                if geometric_prune_this_level and use_plucker_nra_prune and z3_length_ok:
                    cache_key = canonical_forbidden_subword(child)
                    plucker_nra_validity = plucker_nra_validity_cache.get(cache_key)
                    if plucker_nra_validity is None:
                        stats.plucker_nra_checks += 1
                        plucker_nra_validity = certify_stack_transversal_plucker_nra(
                            child,
                            timeout_ms=z3_timeout_ms,
                            warn_after_seconds=z3_warn_after_seconds,
                        )
                        plucker_nra_validity_cache[cache_key] = plucker_nra_validity
                        stats.plucker_nra_chart_checks += plucker_nra_validity.linear_chart_checks
                    else:
                        stats.plucker_nra_cache_hits += 1
                    if plucker_nra_validity.status == "unsat":
                        stats.plucker_nra_prunes += 1
                        stats.z3_geometric_prunes += 1
                        learned_forbidden_this_level |= forbidden.register(child)
                        level_invalid += 1
                        continue
                    stats.plucker_nra_unknown += 1

                if geometric_prune_this_level and use_ordered_z3_prune and z3_length_ok:
                    cache_key = canonical_forbidden_subword(child)
                    cached = z3_validity_cache.get(cache_key)
                    computed_validity = False
                    if cached is not None:
                        validity = cached
                        stats.z3_geometric_cache_hits += 1
                    else:
                        computed_validity = True
                        stats.z3_geometric_checks += 1
                        if use_z3_reduce_inequalities:
                            validity = certify_stack_transversal_z3_reduced(
                                child,
                                parent_state=z3_prefix_state,
                                timeout_ms=z3_timeout_ms,
                                warn_after_seconds=z3_warn_after_seconds,
                                reduce_inequalities=z3_reduction_scope != "none",
                                reduction_scope=z3_reduction_scope,
                            )
                        else:
                            validity = certify_stack_transversal_z3(
                                child,
                                timeout_ms=z3_timeout_ms,
                                warn_after_seconds=z3_warn_after_seconds,
                            )
                        z3_validity_cache[cache_key] = validity
                    if computed_validity:
                        stats.z3_chart_checks += validity.chart_checks
                        stats.z3_active_chart_states += validity.active_chart_states
                        stats.z3_active_inequalities += validity.active_inequalities
                        stats.z3_reduction_checks += validity.reduction_checks
                        stats.z3_redundant_inequalities += validity.redundant_inequalities
                        stats.z3_reduction_unknown += validity.reduction_unknown
                    if validity.status == "unsat":
                        stats.z3_geometric_prunes += 1
                        learned_forbidden_this_level |= forbidden.register(child)
                        level_invalid += 1
                        continue
                    if validity.status == "sat":
                        stats.z3_geometric_sat += 1
                    else:
                        stats.z3_geometric_unknown += 1
                    if validity.word == child:
                        child_z3_state = validity.z3_state
                elif geometric_prune_this_level and use_ordered_z3_prune:
                    child_z3_state = z3_prefix_state

                if lazy_frontier_maps:
                    next_frontier.append(append_packed_face(int(prefix_item), face))
                else:
                    next_M, next_b, next_q = append_face_to_return_map(parent_M, parent_b, parent_q, face)
                    next_frontier.append((child, next_M, next_b, next_q, child_z3_state))
            now = time.perf_counter()
            if now - last_build_progress >= 30:
                print(
                    f"[level {next_period:2d}] building candidates "
                    f"prefixes={built_prefixes}/{frontier_count} "
                    f"checked={level_checked} invalid={level_invalid} "
                    f"kept={len(next_frontier)} elapsed={now - build_started:.1f}s",
                    flush=True,
                )
                last_build_progress = now

        if learned_forbidden_this_level:
            retained_frontier = empty_frontier_container(next_period, lazy_frontier_maps)
            for item in next_frontier:
                if forbidden.hits_linear_subword(search_frontier_word(item, next_period)):
                    stats.forbidden_suffix_prunes += 1
                    level_invalid += 1
                else:
                    retained_frontier.append(item)
            next_frontier = retained_frontier

        frontier = next_frontier
        level_valid = len(frontier)

        level_hits = 0
        if next_period >= 2:
            exact_started = time.perf_counter()
            last_exact_progress = exact_started
            exact_seen = 0
            for item in frontier:
                exact_seen += 1
                stats.generated_prefixes += 1
                stats.generated_by_period[next_period] += 1
                if even_periods_only and next_period % 2:
                    maybe_print_progress(stats, report_every, next_period)
                    continue
                if (
                    equivalence == "full"
                    and isinstance(item, int)
                    and not packed_can_be_full_canonical(item, next_period)
                ):
                    stats.equivalence_skips += 1
                    continue
                prefix, M, b, q, _ = search_frontier_components(item, next_period)
                exact_M, exact_b, exact_q = materialize_return_map(prefix, M, b, q)
                if check_complete_word(prefix, exact_M, exact_b, exact_q):
                    level_hits += 1
                maybe_print_progress(stats, report_every, next_period)
                now = time.perf_counter()
                if now - last_exact_progress >= 30:
                    print(
                        f"[level {next_period:2d}] exact candidates="
                        f"{exact_seen}/{level_valid} hits={level_hits} "
                        f"exact_checks={stats.exact_by_period[next_period]} "
                        f"elapsed={now - exact_started:.1f}s",
                        flush=True,
                    )
                    last_exact_progress = now

        stats.level_reports[next_period] = {
            "checked": int(level_checked),
            "valid": int(level_valid),
            "invalid": int(level_invalid),
            "hits": int(level_hits),
        }
        completed_level = next_period
        write_level_outputs(next_period, frontier)
        print(
            f"[level {next_period:2d}] checked={level_checked} "
            f"valid={level_valid} invalid={level_invalid} hits={level_hits}",
            flush=True,
        )

        if not frontier:
            break

    if checkpoint_out is not None and completed_level == start_period:
        write_search_checkpoint(checkpoint_out, completed_level, stats, forbidden, frontier, records_by_id)

    records = sorted(records_by_id.values(), key=lambda r: (int(r["period"]), str(r["word"])))
    return {
        "summary": stats.to_json(forbidden),
        "loaded_forbidden_subwords": loaded_forbidden,
        "loaded_forbidden_through": loaded_forbidden_through,
        "loaded_previous_orbits": loaded_previous_orbits,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "resume_frontier_level": start_period,
        "completed_level": completed_level,
        "orbits": records,
    }


def verify_one_word(word_text: str) -> Dict[str, object]:
    word = parse_word(word_text)
    result = verify_word_exact(word)
    payload: Dict[str, object] = {
        "word": word_name(word),
        "closed": result.orbit is not None,
        "reason": result.reason,
    }
    if result.orbit is not None:
        payload["orbit"] = orbit_to_record(result.orbit, equivalence="none")
    return payload


def check_one_geometric_word(word_text: str, timeout_ms: int, warn_after_seconds: float) -> Dict[str, object]:
    word = parse_word(word_text)
    result = certify_stack_transversal_z3(
        word,
        timeout_ms=timeout_ms,
        warn_after_seconds=warn_after_seconds,
    )
    return {
        "word": word_name(word),
        "status": result.status,
        "feasible": result.feasible,
        "reason": result.reason,
        "elapsed_seconds": result.elapsed_seconds,
        "timeout_ms": timeout_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact exhaustive closed-path search for regular tetrahedron billiards.")
    parser.add_argument("--max-period", type=int, default=12, help="Search all periods up to this many bounces.")
    parser.add_argument(
        "--equivalence",
        choices=["none", "cyclic", "full"],
        default="full",
        help="Return one representative per chosen equivalence class.",
    )
    parser.add_argument(
        "--include-imprimitive",
        action="store_true",
        help="Include repeated words. By default only primitive words are checked.",
    )
    parser.add_argument(
        "--no-forbidden-subwords",
        action="store_true",
        help="Disable certified forbidden-subword pruning.",
    )
    parser.add_argument(
        "--use-xyxy-lemma",
        action="store_true",
        help="Enable the two-face xyxy forbidden-subword lemma. Leave off unless this lemma is accepted/proved.",
    )
    parser.add_argument(
        "--z3-geometric-prune",
        action="store_true",
        help="Use Z3 QF_NRA to certify geometrically invalid unfolded stacks and prune only on UNSAT.",
    )
    parser.add_argument(
        "--no-plucker-lra-prune",
        action="store_true",
        help="Disable the exact linear Plucker relaxation screen before nonlinear Z3 checks.",
    )
    parser.add_argument(
        "--no-plucker-nra-prune",
        action="store_true",
        help="Disable the fixed-size Plucker quadratic screen before the ordered-stack Z3 check.",
    )
    parser.add_argument(
        "--plucker-only-prune",
        action="store_true",
        help="Skip the ordered-stack Z3 check after Plucker screens; still prunes only on certified UNSAT.",
    )
    parser.add_argument(
        "--z3-reduce-inequalities",
        action="store_true",
        help="Propagate Z3 prefix chart states through the tree; use --z3-reduction-scope to enable redundancy checks.",
    )
    parser.add_argument(
        "--z3-reduction-scope",
        choices=["none", "new", "all"],
        default="none",
        help="Which active inequalities to test for redundancy after a SAT child chart. new/all are certified but can be very slow.",
    )
    parser.add_argument(
        "--z3-timeout-ms",
        type=int,
        default=0,
        help="Per-word Z3 timeout in milliseconds. 0 means no timeout.",
    )
    parser.add_argument(
        "--z3-warn-after-sec",
        type=float,
        default=100.0,
        help="Print a progress warning if one Z3 check runs longer than this many seconds. 0 disables warnings.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for Plucker-only pruning. 1 keeps the search single-process.",
    )
    parser.add_argument(
        "--deep-plucker-start-level",
        type=int,
        default=31,
        help="At this word length and above, use the deep-level Plucker worker cap.",
    )
    parser.add_argument(
        "--deep-plucker-workers",
        type=int,
        default=4,
        help=(
            "Maximum Plucker worker processes at deep levels. 0 disables the "
            "deep-level cap and spawn switch. Deep pools use spawn so they do "
            "not inherit the large frontier."
        ),
    )
    parser.add_argument(
        "--z3-min-length",
        type=int,
        default=2,
        help="Only run Z3 geometric pruning on words at least this long.",
    )
    parser.add_argument(
        "--z3-max-length",
        type=int,
        default=0,
        help="Only run Z3 geometric pruning up to this length. 0 means no upper limit.",
    )
    parser.add_argument(
        "--no-prefix-symmetry",
        action="store_true",
        help="Do not quotient sibling branches by the stabilizer of the current prefix.",
    )
    parser.add_argument(
        "--check-odd-periods",
        action="store_true",
        help="Also run exact closure checks at odd lengths.",
    )
    parser.add_argument(
        "--skip-odd-geometric-prune",
        action="store_true",
        help=(
            "When odd periods are not checked for closure, skip Plucker/Z3 pruning "
            "on odd levels and defer certified geometric pruning to even children."
        ),
    )
    parser.add_argument("--word", help="Verify one explicit closed face word, e.g. DABC, instead of running a search.")
    parser.add_argument(
        "--geometric-word",
        help="Check whether one explicit word has a line through the unfolded open-face stack, e.g. ABAB.",
    )
    parser.add_argument("--json-out", type=Path, default=Path("tetrahedron_exhaustive_closed_paths.json"))
    parser.add_argument(
        "--forbidden-in",
        type=Path,
        action="append",
        default=[],
        help="Load certified forbidden subwords from a previous search JSON or forbidden checkpoint.",
    )
    parser.add_argument(
        "--forbidden-out",
        type=Path,
        help="Write forbidden-subword checkpoint JSON after every completed level.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Resume from a frontier checkpoint written by --checkpoint-out.",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        help="Write a resumable frontier checkpoint JSON after every completed level.",
    )
    parser.add_argument(
        "--checkpoint-even-levels-only",
        action="store_true",
        help=(
            "When odd periods are not checked, skip writing odd-level forbidden/frontier "
            "checkpoints. Odd frontiers are rebuilt from the previous even checkpoint."
        ),
    )
    parser.add_argument(
        "--checkpoint-stride",
        type=int,
        default=1,
        help=(
            "Write forbidden/frontier checkpoints only every N completed levels, anchored at "
            "the resume level. For example, resuming from level 26 with --checkpoint-stride 2 "
            "checks level 27 in memory, then writes level 28."
        ),
    )
    parser.add_argument(
        "--expand-stride",
        type=int,
        choices=[1, 2],
        default=1,
        help=(
            "When set to 2 in lazy Plucker-only mode, stream two levels at a time: "
            "prune children at level k, immediately expand/prune grandchildren at "
            "level k+1, and store only the k+1 frontier."
        ),
    )
    parser.add_argument(
        "--lazy-frontier-maps",
        action="store_true",
        help=(
            "Keep only frontier words in memory and recompute affine return maps "
            "when expanding/verifying. This is slower but much lighter for deep resumes."
        ),
    )
    parser.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=100000,
        help=(
            "Maximum pending candidate words to hold before running Plucker pruning "
            "in parallel. Smaller values reduce peak memory."
        ),
    )
    parser.add_argument("--report-every", type=int, default=50000, help="Progress print interval by generated prefixes.")
    args = parser.parse_args()

    if args.word:
        print(json.dumps(verify_one_word(args.word), indent=2), flush=True)
        return
    if args.geometric_word:
        print(
            json.dumps(
                check_one_geometric_word(
                    args.geometric_word,
                    args.z3_timeout_ms,
                    args.z3_warn_after_sec,
                ),
                indent=2,
            ),
            flush=True,
        )
        return

    if args.z3_reduce_inequalities:
        args.z3_geometric_prune = True
    use_ordered_z3_prune = args.z3_geometric_prune and not args.plucker_only_prune

    data = exhaustive_search(
        max_period=args.max_period,
        equivalence=args.equivalence,
        primitive_only=not args.include_imprimitive,
        use_forbidden_subwords=not args.no_forbidden_subwords,
        use_xyxy_lemma=args.use_xyxy_lemma,
        use_z3_geometric_prune=args.z3_geometric_prune,
        use_plucker_lra_prune=args.z3_geometric_prune and not args.no_plucker_lra_prune,
        use_plucker_nra_prune=args.z3_geometric_prune and not args.no_plucker_nra_prune,
        use_ordered_z3_prune=use_ordered_z3_prune,
        use_z3_reduce_inequalities=args.z3_reduce_inequalities,
        z3_reduction_scope=args.z3_reduction_scope,
        z3_timeout_ms=args.z3_timeout_ms,
        z3_min_length=args.z3_min_length,
        z3_max_length=args.z3_max_length,
        z3_warn_after_seconds=args.z3_warn_after_sec,
        workers=max(1, args.workers),
        deep_plucker_start_level=args.deep_plucker_start_level,
        deep_plucker_workers=args.deep_plucker_workers,
        use_prefix_symmetry=not args.no_prefix_symmetry,
        even_periods_only=not args.check_odd_periods,
        skip_odd_geometric_prune=args.skip_odd_geometric_prune,
        checkpoint_even_levels_only=args.checkpoint_even_levels_only,
        checkpoint_stride=args.checkpoint_stride,
        expand_stride=args.expand_stride,
        report_every=args.report_every,
        forbidden_in=args.forbidden_in,
        forbidden_out=args.forbidden_out,
        resume_checkpoint=args.resume_checkpoint,
        checkpoint_out=args.checkpoint_out,
        lazy_frontier_maps=args.lazy_frontier_maps,
        candidate_chunk_size=args.candidate_chunk_size,
    )
    args.json_out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    summary = data["summary"]
    print("\nExact exhaustive search finished.", flush=True)
    print(f"max period      : {summary['max_period']}", flush=True)
    print(f"equivalence     : {summary['equivalence']}", flush=True)
    print(f"primitive only  : {summary['primitive_only']}", flush=True)
    print(f"use xyxy lemma  : {summary['use_xyxy_lemma']}", flush=True)
    print(f"use z3 geometry : {summary['use_z3_geometric_prune']}", flush=True)
    print(f"even periods only: {summary['even_periods_only']}", flush=True)
    print(f"total orbits    : {len(data['orbits'])}", flush=True)
    print(f"exact checks    : {summary['exact_checks']}", flush=True)
    print(f"elapsed seconds : {summary['elapsed_seconds']:.3f}", flush=True)
    print("per period      :", flush=True)
    for period, count in summary["closed_by_period"].items():
        print(f"  {period}: {count}", flush=True)
    print(f"JSON written to : {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
