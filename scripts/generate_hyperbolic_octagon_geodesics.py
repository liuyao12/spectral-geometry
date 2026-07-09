#!/usr/bin/env python3
"""Generate closed-geodesic data for a compact genus-2 hyperbolic surface.

The surface is the regular hyperbolic octagon with opposite sides identified.
The octagon has interior angle pi/4, so the eight vertices glue to a smooth
point and the quotient has genus two.
"""

from __future__ import annotations

import argparse
import cmath
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path


LETTERS = ("a", "b", "c", "d", "A", "B", "C", "D")
INVERSE = {
    "a": "A",
    "b": "B",
    "c": "C",
    "d": "D",
    "A": "a",
    "B": "b",
    "C": "c",
    "D": "d",
}
ORDER = {letter: index for index, letter in enumerate(LETTERS)}


@dataclass(frozen=True)
class DiskMobius:
    """An SU(1,1) disk isometry z |-> (a z + b)/(conj(b) z + conj(a))."""

    a: complex
    b: complex

    def compose(self, other: "DiskMobius") -> "DiskMobius":
        """Return self after other."""

        return DiskMobius(
            self.a * other.a + self.b * other.b.conjugate(),
            self.a * other.b + self.b * other.a.conjugate(),
        )

    def inverse(self) -> "DiskMobius":
        return DiskMobius(self.a.conjugate(), -self.b)

    def trace(self) -> float:
        return 2.0 * self.a.real


IDENTITY = DiskMobius(1 + 0j, 0 + 0j)


def rotation(angle: float) -> DiskMobius:
    half = angle / 2.0
    return DiskMobius(complex(math.cos(half), math.sin(half)), 0 + 0j)


def real_translation(length: float) -> DiskMobius:
    half = length / 2.0
    return DiskMobius(math.cosh(half) + 0j, math.sinh(half) + 0j)


def oriented_translation(angle: float, length: float) -> DiskMobius:
    return rotation(angle).compose(real_translation(length)).compose(rotation(-angle))


def word_matrix(word: str, generators: dict[str, DiskMobius]) -> DiskMobius:
    matrix = IDENTITY
    for letter in word:
        matrix = generators[letter].compose(matrix)
    return matrix


def fixed_points(matrix: DiskMobius) -> tuple[complex, complex]:
    # conj(b) z^2 + (conj(a) - a) z - b = 0
    qa = matrix.b.conjugate()
    qb = matrix.a.conjugate() - matrix.a
    qc = -matrix.b
    if abs(qa) < 1e-14:
        return 1 + 0j, -1 + 0j
    root = cmath.sqrt(qb * qb - 4.0 * qa * qc)
    z1 = (-qb + root) / (2.0 * qa)
    z2 = (-qb - root) / (2.0 * qa)
    return z1 / abs(z1), z2 / abs(z2)


def canonical_word(word: str) -> str:
    inverse = "".join(INVERSE[letter] for letter in reversed(word))
    rotations = [word[i:] + word[:i] for i in range(len(word))]
    rotations += [inverse[i:] + inverse[:i] for i in range(len(inverse))]
    return min(rotations, key=lambda item: tuple(ORDER[ch] for ch in item))


def is_reduced(word: str) -> bool:
    return all(INVERSE[a] != b for a, b in zip(word, word[1:]))


def is_cyclically_reduced(word: str) -> bool:
    return is_reduced(word) and INVERSE[word[0]] != word[-1]


def is_power(word: str) -> bool:
    for step in range(1, len(word)):
        if len(word) % step == 0 and word == word[:step] * (len(word) // step):
            return True
    return False


def cpair(z: complex, digits: int = 12) -> list[float]:
    return [round(z.real, digits), round(z.imag, digits)]


def cobj(z: complex, digits: int = 12) -> dict[str, float]:
    return {"re": round(z.real, digits), "im": round(z.imag, digits)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-word-length", type=int, default=6)
    parser.add_argument("--limit", type=int, default=72)
    parser.add_argument("--out", type=Path, default=Path("docs/data/hyperbolic/bolza_octagon_geodesics.json"))
    args = parser.parse_args()

    p = 8
    q = 8
    half_turn = math.pi / 4.0
    interior_angle = 2.0 * math.pi / q
    inradius_h = math.acosh(math.cos(math.pi / p) / math.sin(math.pi / q))
    circumradius_h = math.acosh((math.cos(math.pi / p) / math.sin(math.pi / p)) ** 2)
    side_length_h = 2.0 * math.acosh(math.cos(math.pi / q) / math.sin(math.pi / p))
    vertex_radius = math.tanh(circumradius_h / 2.0)
    side_foot_radius = math.tanh(inradius_h / 2.0)
    side_pair_translation = 2.0 * inradius_h

    base_generators: dict[str, DiskMobius] = {}
    for index, letter in enumerate("abcd"):
        base_generators[letter] = oriented_translation(index * half_turn, side_pair_translation)

    generators = dict(base_generators)
    for letter in "abcd":
        generators[INVERSE[letter]] = generators[letter].inverse()

    vertices = []
    for index in range(p):
        angle = (index + 0.5) * half_turn
        vertices.append(
            {
                "index": index,
                "angle": round(angle, 12),
                "point": cpair(vertex_radius * complex(math.cos(angle), math.sin(angle))),
            }
        )

    sides = []
    for index in range(p):
        pair = (index + 4) % 8
        base = "abcd"[index % 4]
        label = base if index < 4 else f"{base}^-1"
        sends_to = pair if index < 4 else pair
        sides.append(
            {
                "index": index,
                "label": label,
                "paired_with": pair,
                "pair_color_index": index % 4,
                "normal_angle": round(index * half_turn, 12),
                "vertices": [(index - 1) % p, index],
                "pairing_generator": base,
                "pairing_direction": f"{base} maps side {index + 4 if index < 4 else index} to side {index if index < 4 else index - 4}",
                "paired_side": sends_to,
            }
        )

    seen: set[str] = set()
    records = []
    for length in range(1, args.max_word_length + 1):
        for letters in itertools.product(LETTERS, repeat=length):
            word = "".join(letters)
            if not is_cyclically_reduced(word) or is_power(word):
                continue
            canonical = canonical_word(word)
            if canonical != word or canonical in seen:
                continue
            seen.add(canonical)
            matrix = word_matrix(word, generators)
            trace = matrix.trace()
            trace_abs = abs(trace)
            if trace_abs <= 2.000000001:
                continue
            z1, z2 = fixed_points(matrix)
            records.append(
                {
                    "id": f"geo-{len(records) + 1:03d}",
                    "word": word,
                    "inverse_word": "".join(INVERSE[letter] for letter in reversed(word)),
                    "word_length": length,
                    "trace": round(trace, 12),
                    "trace_abs": round(trace_abs, 12),
                    "length": round(2.0 * math.acosh(trace_abs / 2.0), 12),
                    "matrix": {"a": cobj(matrix.a), "b": cobj(matrix.b)},
                    "fixed_points": [cpair(z1), cpair(z2)],
                    "primitive": True,
                }
            )

    records.sort(key=lambda item: (item["length"], item["word_length"], item["word"]))
    records = records[: args.limit]
    for index, item in enumerate(records, start=1):
        item["id"] = f"geo-{index:03d}"

    generator_records = []
    for letter in "abcd":
        matrix = generators[letter]
        generator_records.append(
            {
                "letter": letter,
                "inverse": INVERSE[letter],
                "side_pair": [ORDER[letter], ORDER[letter] + 4],
                "axis_angle": round(ORDER[letter] * half_turn, 12),
                "translation_length": round(side_pair_translation, 12),
                "matrix": {"a": cobj(matrix.a), "b": cobj(matrix.b)},
            }
        )

    data = {
        "schema": "spectral-geometry.compact-hyperbolic.v1",
        "surface": {
            "id": "regular-octagon-genus-2",
            "name": "Regular octagon genus-2 surface",
            "genus": 2,
            "compact": True,
            "description": "Opposite sides of a regular hyperbolic octagon are identified. The interior angle is pi/4, so all eight vertices glue smoothly.",
        },
        "polygon": {
            "model": "poincare_disk",
            "type": "regular_hyperbolic_octagon",
            "p": p,
            "q": q,
            "interior_angle": round(interior_angle, 12),
            "area": round((p - 2) * math.pi - p * interior_angle, 12),
            "inradius_h": round(inradius_h, 12),
            "circumradius_h": round(circumradius_h, 12),
            "side_length_h": round(side_length_h, 12),
            "vertex_radius_disk": round(vertex_radius, 12),
            "side_foot_radius_disk": round(side_foot_radius, 12),
            "vertices": vertices,
            "sides": sides,
        },
        "generators": generator_records,
        "geodesics": records,
        "enumeration": {
            "method": "local enumeration of primitive cyclically reduced words in the opposite-side pairing group",
            "max_word_length": args.max_word_length,
            "retained_shortest": len(records),
            "generated_by": "scripts/generate_hyperbolic_octagon_geodesics.py",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
