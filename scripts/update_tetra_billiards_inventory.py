from __future__ import annotations

import argparse
import json
import math
import time
import mmap
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


VERTICES = {
    "A": (Fraction(1), Fraction(0), Fraction(0)),
    "B": (Fraction(0), Fraction(1), Fraction(0)),
    "C": (Fraction(0), Fraction(0), Fraction(1)),
    "D": (Fraction(1), Fraction(1), Fraction(1)),
}
FACE_LABELS = ("A", "B", "C", "D")


def frac_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def parse_frac(value: object) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    return Fraction(str(value))


def point_from_barycentric(row: Sequence[object], denominator: object) -> Tuple[Fraction, Fraction, Fraction]:
    den = parse_frac(denominator)
    weights = [parse_frac(value) / den for value in row]
    coords: List[Fraction] = []
    for axis in range(3):
        coords.append(sum(weights[i] * VERTICES[FACE_LABELS[i]][axis] for i in range(4)))
    return coords[0], coords[1], coords[2]


def barycentric_strings(row: Sequence[object], denominator: object) -> List[str]:
    den = parse_frac(denominator)
    return [frac_text(parse_frac(value) / den) for value in row]


def face_from_barycentric(row: Sequence[object]) -> str:
    zeros = [FACE_LABELS[i] for i, value in enumerate(row) if str(value) == "0"]
    return zeros[0] if zeros else "?"


def float_point(values: Sequence[object]) -> List[float]:
    return [float(parse_frac(value)) for value in values]


def normalize_current_record(record: Dict[str, object]) -> Optional[Dict[str, object]]:
    if record.get("singular") or record.get("kind") == "singular_normal_cone":
        return None
    word = str(record.get("word") or record.get("word_name_string") or "")
    if not word:
        return None
    period = int(record.get("period", len(word)))
    bary_rows = record.get("barycentric_points")
    denominator = record.get("barycentric_denominator")
    if not isinstance(bary_rows, list) or denominator is None:
        return None

    bary_rows_text = [[str(value) for value in row] for row in bary_rows]
    denominator_text = str(denominator)
    faces = [face_from_barycentric(row) for row in bary_rows_text]

    raw_points = record.get("points_xyz_exact") or record.get("points_exact")
    if isinstance(raw_points, list) and len(raw_points) == len(bary_rows_text):
        points_exact = [[str(value) for value in row] for row in raw_points]
        points_float = [float_point(row) for row in points_exact]
    else:
        exact_points = [point_from_barycentric(row, denominator_text) for row in bary_rows_text]
        points_exact = [[frac_text(coord) for coord in point] for point in exact_points]
        points_float = [[float(coord) for coord in point] for point in exact_points]

    length_exact = str(record.get("length_exact") or record.get("total_length_exact") or "")
    length_numeric = float(record.get("length_numeric") or record.get("total_length_float") or 0.0)

    height = max(int(abs(int(value))) for row in bary_rows_text for value in row)
    den_int = int(denominator_text)
    boundary_margin_num = min(int(value) for row in bary_rows_text for value in row if int(value) > 0)

    return {
        "id": str(record.get("id") or f"p{period:02d}_{word}"),
        "kind": "ordinary",
        "equivalence": str(record.get("equivalence") or "full"),
        "period": period,
        "word": word,
        "faces": faces,
        "length_exact": length_exact,
        "length_numeric": length_numeric,
        "barycentric_denominator": denominator_text,
        "barycentric_points": bary_rows_text,
        "barycentric_exact": [barycentric_strings(row, denominator_text) for row in bary_rows_text],
        "points_xyz_exact": points_exact,
        "points_xyz": points_float,
        "axis_direction": [str(value) for value in record.get("axis_direction", [])],
        "initial_direction": [str(value) for value in record.get("initial_direction", [])],
        "height": str(max(height, den_int)),
        "boundary_margin": frac_text(Fraction(boundary_margin_num, den_int)),
    }


def load_json_without_frontier_words(path: Path) -> Dict[str, object]:
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


def load_source(path: Path) -> Dict[str, object]:
    if path.stat().st_size == 0:
        raise ValueError(f"{path} is empty")
    if "checkpoint" in path.name:
        return load_json_without_frontier_words(path)
    return json.loads(path.read_text(encoding="utf-8"))


def orbit_records_from_source(path: Path) -> List[Dict[str, object]]:
    data = load_source(path)
    records = data.get("orbits", [])
    if not isinstance(records, list):
        return []
    out = []
    for record in records:
        if not isinstance(record, dict):
            continue
        normalized = normalize_current_record(record)
        if normalized is not None:
            normalized["source"] = str(path)
            out.append(normalized)
    return out


def default_sources() -> List[Path]:
    candidates = [
        REPO_ROOT / "data" / "tetra_frontier_checkpoint_period60.json",
        REPO_ROOT / "data" / "tetra_z3_period60.json",
        REPO_ROOT / "data" / "level42_shards" / "merged_level42.json",
    ]
    return [path for path in candidates if path.exists() and path.stat().st_size > 0]


def build_inventory(sources: Sequence[Path]) -> Dict[str, object]:
    by_id: Dict[str, Dict[str, object]] = {}
    source_summaries = []
    for path in sources:
        records = orbit_records_from_source(path)
        source_summaries.append({"path": str(path), "ordinary_orbits": len(records)})
        for record in records:
            by_id[str(record["id"])] = record

    orbits = sorted(by_id.values(), key=lambda item: (int(item["period"]), float(item["length_numeric"]), str(item["word"])))
    by_period = Counter(int(record["period"]) for record in orbits)
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return {
        "schema": "spectral.tetra_billiards_inventory.v1",
        "generated_at": generated_at,
        "title": "Closed billiard paths in the regular tetrahedron",
        "ordinary_only": True,
        "source_notebook": "https://observablehq.com/@liuyao12/billiard-in-tetrahedra",
        "tetrahedron": {
            "vertices": {name: [float(coord) for coord in point] for name, point in VERTICES.items()},
            "faces": {
                "A": ["B", "C", "D"],
                "B": ["A", "C", "D"],
                "C": ["A", "B", "D"],
                "D": ["A", "B", "C"],
            },
            "face_convention": "A, B, C, D denote the faces opposite the corresponding vertices.",
        },
        "summary": {
            "total_orbits": len(orbits),
            "max_period": max((int(record["period"]) for record in orbits), default=0),
            "period_counts": {str(period): by_period[period] for period in sorted(by_period)},
            "sources": source_summaries,
        },
        "orbits": orbits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the static-site ordinary tetrahedron billiards inventory.")
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=[],
        help="Exact search JSON/checkpoint/merged shard JSON to include. May be repeated.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "site" / "data" / "tetra" / "billiards_inventory.json",
        help="Output JSON path for the GitHub Pages viewer.",
    )
    args = parser.parse_args()

    sources = [path.resolve() for path in args.source] if args.source else default_sources()
    if not sources:
        raise SystemExit("No inventory sources found. Pass --source explicitly.")

    inventory = build_inventory(sources)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(
        f"wrote {args.out} with {inventory['summary']['total_orbits']} ordinary orbit classes "
        f"from {len(sources)} source(s)"
    )


if __name__ == "__main__":
    main()
