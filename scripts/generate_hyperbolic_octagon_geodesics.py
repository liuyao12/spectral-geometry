#!/usr/bin/env python3
"""Generate closed-geodesic data for compact regular hyperbolic surfaces.

The surfaces are regular hyperbolic 4g-gons with opposite sides identified.
For genus g the polygon has p = q = 4g and interior angle pi / (2g), so the
vertex angle sum in the quotient is 2pi and the area is 4pi(g - 1).
"""

from __future__ import annotations

import argparse
import cmath
import itertools
import json
import math
import string
from dataclasses import dataclass
from pathlib import Path


SURFACE_DEFAULTS = {
    2: {"max_word_length": 6, "limit": 80, "filename": "bolza_octagon_geodesics.json"},
    3: {"max_word_length": 5, "limit": 80, "filename": "regular_genus3_dodecagon_geodesics.json"},
    4: {"max_word_length": 4, "limit": 80, "filename": "regular_genus4_16gon_geodesics.json"},
    5: {"max_word_length": 4, "limit": 80, "filename": "regular_genus5_20gon_geodesics.json"},
}


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


def labels_for_genus(genus: int) -> tuple[tuple[str, ...], dict[str, str], dict[str, int]]:
    generator_count = 2 * genus
    if generator_count > len(string.ascii_lowercase):
        raise ValueError("This generator only supports up to 13 handles with single-letter labels.")
    lower = tuple(string.ascii_lowercase[:generator_count])
    upper = tuple(label.upper() for label in lower)
    letters = lower + upper
    inverse = {label: label.upper() for label in lower}
    inverse.update({label.upper(): label for label in lower})
    order = {letter: index for index, letter in enumerate(letters)}
    return letters, inverse, order


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


def canonical_word(word: str, inverse: dict[str, str], order: dict[str, int]) -> str:
    inverse_word = "".join(inverse[letter] for letter in reversed(word))
    rotations = [word[i:] + word[:i] for i in range(len(word))]
    rotations += [inverse_word[i:] + inverse_word[:i] for i in range(len(inverse_word))]
    return min(rotations, key=lambda item: tuple(order[ch] for ch in item))


def word_sort_key(word: str, order: dict[str, int]) -> tuple[int, ...]:
    return tuple(order[ch] for ch in word)


def side_to_letter(side_index: int, lower_letters: tuple[str, ...], inverse: dict[str, str]) -> str:
    generator_count = len(lower_letters)
    if side_index < generator_count:
        return lower_letters[side_index]
    return inverse[lower_letters[side_index - generator_count]]


def letter_target_side(letter: str, lower_letters: tuple[str, ...], inverse: dict[str, str]) -> int:
    generator_count = len(lower_letters)
    if letter in lower_letters:
        return lower_letters.index(letter)
    return lower_letters.index(inverse[letter]) + generator_count


def symmetry_word(
    word: str,
    shift: int,
    reflected: bool,
    side_count: int,
    lower_letters: tuple[str, ...],
    inverse: dict[str, str],
) -> str:
    mapped = []
    for letter in word:
        side = letter_target_side(letter, lower_letters, inverse)
        image_side = (shift - side if reflected else shift + side) % side_count
        mapped.append(side_to_letter(image_side, lower_letters, inverse))
    return "".join(mapped)


def symmetry_key(
    word: str,
    side_count: int,
    lower_letters: tuple[str, ...],
    inverse: dict[str, str],
    order: dict[str, int],
) -> str:
    candidates = []
    for shift in range(side_count):
        for reflected in (False, True):
            image = symmetry_word(word, shift, reflected, side_count, lower_letters, inverse)
            candidates.append(canonical_word(image, inverse, order))
    return min(candidates, key=lambda item: tuple(order[ch] for ch in item))


def is_reduced(word: str, inverse: dict[str, str]) -> bool:
    return all(inverse[a] != b for a, b in zip(word, word[1:]))


def is_cyclically_reduced(word: str, inverse: dict[str, str]) -> bool:
    return is_reduced(word, inverse) and inverse[word[0]] != word[-1]


def is_power(word: str) -> bool:
    for step in range(1, len(word)):
        if len(word) % step == 0 and word == word[:step] * (len(word) // step):
            return True
    return False


def polygon_name(p: int) -> str:
    names = {8: "octagon", 12: "dodecagon", 16: "16-gon", 20: "20-gon"}
    return names.get(p, f"{p}-gon")


def cpair(z: complex, digits: int = 12) -> list[float]:
    return [round(z.real, digits), round(z.imag, digits)]


def cobj(z: complex, digits: int = 12) -> dict[str, float]:
    return {"re": round(z.real, digits), "im": round(z.imag, digits)}


def build_surface(genus: int, max_word_length: int, limit: int) -> dict:
    p = 4 * genus
    q = p
    generator_count = p // 2
    side_step = 2.0 * math.pi / p
    letters, inverse, order = labels_for_genus(genus)
    lower_letters = letters[:generator_count]

    interior_angle = 2.0 * math.pi / q
    inradius_h = math.acosh(math.cos(math.pi / p) / math.sin(math.pi / q))
    circumradius_h = math.acosh(
        math.cos(math.pi / p) * math.cos(math.pi / q) / (math.sin(math.pi / p) * math.sin(math.pi / q))
    )
    side_length_h = 2.0 * math.acosh(math.cos(math.pi / q) / math.sin(math.pi / p))
    vertex_radius = math.tanh(circumradius_h / 2.0)
    side_foot_radius = math.tanh(inradius_h / 2.0)
    side_pair_translation = 2.0 * inradius_h

    base_generators: dict[str, DiskMobius] = {}
    for index, letter in enumerate(lower_letters):
        base_generators[letter] = oriented_translation(index * side_step, side_pair_translation)

    generators = dict(base_generators)
    for letter in lower_letters:
        generators[inverse[letter]] = generators[letter].inverse()

    vertices = []
    for index in range(p):
        angle = (index + 0.5) * side_step
        vertices.append(
            {
                "index": index,
                "angle": round(angle, 12),
                "point": cpair(vertex_radius * complex(math.cos(angle), math.sin(angle))),
            }
        )

    sides = []
    for index in range(p):
        pair = (index + generator_count) % p
        base = lower_letters[index % generator_count]
        label = base if index < generator_count else f"{base}^-1"
        sides.append(
            {
                "index": index,
                "label": label,
                "paired_with": pair,
                "pair_color_index": index % generator_count,
                "normal_angle": round(index * side_step, 12),
                "vertices": [(index - 1) % p, index],
                "pairing_generator": base,
                "pairing_direction": (
                    f"{base} maps side {index + generator_count if index < generator_count else index} "
                    f"to side {index if index < generator_count else index - generator_count}"
                ),
                "paired_side": pair,
            }
        )

    seen: set[str] = set()
    records_by_word = {}
    for length in range(1, max_word_length + 1):
        for letters_tuple in itertools.product(letters, repeat=length):
            word = "".join(letters_tuple)
            if not is_cyclically_reduced(word, inverse) or is_power(word):
                continue
            canonical = canonical_word(word, inverse, order)
            if canonical != word or canonical in seen:
                continue
            seen.add(canonical)
            matrix = word_matrix(word, generators)
            trace = matrix.trace()
            trace_abs = abs(trace)
            if trace_abs <= 2.000000001:
                continue
            z1, z2 = fixed_points(matrix)
            records_by_word[word] = {
                "id": f"geo-{len(records_by_word) + 1:03d}",
                "word": word,
                "inverse_word": "".join(inverse[letter] for letter in reversed(word)),
                "word_length": length,
                "trace": round(trace, 12),
                "trace_abs": round(trace_abs, 12),
                "length": round(2.0 * math.acosh(trace_abs / 2.0), 12),
                "matrix": {"a": cobj(matrix.a), "b": cobj(matrix.b)},
                "fixed_points": [cpair(z1), cpair(z2)],
                "primitive": True,
            }

    symmetry_orbits: dict[str, list[dict]] = {}
    for word, record in records_by_word.items():
        key = symmetry_key(word, p, lower_letters, inverse, order)
        symmetry_orbits.setdefault(key, []).append(record)

    records = []
    for orbit_records in symmetry_orbits.values():
        orbit_records.sort(key=lambda item: (item["length"], item["word_length"], word_sort_key(item["word"], order)))
        representative = dict(orbit_records[0])
        representative["symmetry_multiplicity"] = len(orbit_records)
        representative["equivalent_words"] = sorted((item["word"] for item in orbit_records), key=lambda word: word_sort_key(word, order))
        records.append(representative)

    records.sort(key=lambda item: (item["length"], item["word_length"], word_sort_key(item["word"], order)))
    records = records[:limit]
    for index, item in enumerate(records, start=1):
        item["id"] = f"geo-{index:03d}"

    generator_records = []
    for index, letter in enumerate(lower_letters):
        matrix = generators[letter]
        generator_records.append(
            {
                "letter": letter,
                "inverse": inverse[letter],
                "side_pair": [index, index + generator_count],
                "axis_angle": round(index * side_step, 12),
                "translation_length": round(side_pair_translation, 12),
                "matrix": {"a": cobj(matrix.a), "b": cobj(matrix.b)},
            }
        )

    poly_name = polygon_name(p)
    data = {
        "schema": "spectral-geometry.compact-hyperbolic.v1",
        "surface": {
            "id": f"regular-{p}gon-genus-{genus}",
            "name": f"Regular {poly_name} genus-{genus} surface",
            "genus": genus,
            "compact": True,
            "description": (
                f"Opposite sides of a regular hyperbolic {poly_name} are identified. "
                f"The interior angle is pi/{2 * genus}, so all {p} vertices glue smoothly."
            ),
        },
        "polygon": {
            "model": "poincare_disk",
            "type": f"regular_hyperbolic_{p}gon",
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
            "method": "local enumeration of primitive cyclically reduced words, condensed by regular-polygon dihedral symmetries",
            "max_word_length": max_word_length,
            "retained_shortest": len(records),
            "raw_primitive_conjugacy_classes": len(records_by_word),
            "symmetry_orbits_before_limit": len(symmetry_orbits),
            "generated_by": "scripts/generate_hyperbolic_octagon_geodesics.py",
        },
    }
    return data


def write_surface(path: Path, genus: int, max_word_length: int, limit: int) -> None:
    data = build_surface(genus, max_word_length, limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genus", type=int, default=2)
    parser.add_argument("--max-word-length", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--all", action="store_true", help="write the built-in genus 2 through 5 datasets")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/data/hyperbolic"))
    args = parser.parse_args()

    if args.all:
        for genus, defaults in SURFACE_DEFAULTS.items():
            write_surface(
                args.out_dir / str(defaults["filename"]),
                genus,
                int(defaults["max_word_length"]),
                int(defaults["limit"]),
            )
        return

    if args.genus < 2:
        raise SystemExit("genus must be at least 2")
    defaults = SURFACE_DEFAULTS.get(args.genus, {"max_word_length": 4, "limit": 80, "filename": f"regular_genus{args.genus}.json"})
    max_word_length = args.max_word_length if args.max_word_length is not None else int(defaults["max_word_length"])
    limit = args.limit if args.limit is not None else int(defaults["limit"])
    out = args.out or args.out_dir / str(defaults["filename"])
    write_surface(out, args.genus, max_word_length, limit)


if __name__ == "__main__":
    main()
