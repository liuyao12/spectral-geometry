#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DOCS_INVENTORY = REPO_ROOT / "docs" / "data" / "tetra" / "billiards_inventory.json"
EXPLORATORY_SOURCE = REPO_ROOT / "data" / "tetra_exploratory_paths.json"


def default_workers() -> int:
    count = os.cpu_count() or 4
    return max(1, min(8, count - 1 if count > 2 else count))


def require_z3() -> None:
    try:
        import z3  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "The exact search needs z3-solver for certified geometric pruning. "
            "Install it with: python3 -m pip install --user z3-solver"
        ) from exc


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run/resume the exact tetrahedron billiards search and publish the static inventory."
    )
    parser.add_argument("--max-period", type=int, default=40)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--deep-plucker-workers", type=int, default=4)
    parser.add_argument("--candidate-chunk-size", type=int, default=50000)
    parser.add_argument("--report-every", type=int, default=50000)
    parser.add_argument("--checkpoint-stride", type=int, default=2)
    parser.add_argument("--no-publish", action="store_true", help="Run the search but do not rebuild docs/data.")
    parser.add_argument(
        "--exact-out",
        type=Path,
        help="Compact exact-search JSON. Defaults to data/tetra_exhaustive_period{max_period}.json.",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        help="Large resumable checkpoint. Defaults to data/tetra_exhaustive_period{max_period}_checkpoint.json.",
    )
    parser.add_argument(
        "--forbidden-out",
        type=Path,
        help="Certified forbidden-subword cache. Defaults to data/tetra_exhaustive_period{max_period}_forbidden.json.",
    )
    parser.add_argument("--inventory-out", type=Path, default=DOCS_INVENTORY)
    args = parser.parse_args()

    if args.max_period < 2:
        raise SystemExit("--max-period must be at least 2")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    require_z3()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    exact_out = args.exact_out or DATA_DIR / f"tetra_exhaustive_period{args.max_period}.json"
    checkpoint_out = args.checkpoint_out or DATA_DIR / f"tetra_exhaustive_period{args.max_period}_checkpoint.json"
    forbidden_out = args.forbidden_out or DATA_DIR / f"tetra_exhaustive_period{args.max_period}_forbidden.json"

    search_command = [
        sys.executable,
        "tetrahedron_exhaustive_closed_paths.py",
        "--max-period",
        str(args.max_period),
        "--z3-geometric-prune",
        "--plucker-only-prune",
        "--skip-odd-geometric-prune",
        "--lazy-frontier-maps",
        "--expand-stride",
        "2",
        "--workers",
        str(args.workers),
        "--deep-plucker-start-level",
        "31",
        "--deep-plucker-workers",
        str(args.deep_plucker_workers),
        "--candidate-chunk-size",
        str(args.candidate_chunk_size),
        "--checkpoint-out",
        str(checkpoint_out),
        "--forbidden-out",
        str(forbidden_out),
        "--checkpoint-even-levels-only",
        "--checkpoint-stride",
        str(args.checkpoint_stride),
        "--json-out",
        str(exact_out),
        "--report-every",
        str(args.report_every),
    ]
    if checkpoint_out.exists() and checkpoint_out.stat().st_size > 0:
        search_command.extend(["--resume-checkpoint", str(checkpoint_out)])
    run(search_command)

    if args.no_publish:
        return

    sources = [exact_out]
    if EXPLORATORY_SOURCE.exists() and EXPLORATORY_SOURCE.stat().st_size > 0:
        sources.append(EXPLORATORY_SOURCE)

    publish_command = [
        sys.executable,
        "scripts/update_tetra_billiards_inventory.py",
        "--exhaustive-through",
        str(args.max_period),
        "--out",
        str(args.inventory_out),
    ]
    for source in sources:
        publish_command.extend(["--source", str(source)])
    run(publish_command)


if __name__ == "__main__":
    main()
