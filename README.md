Placeholder for a database for **spectral geometry**, that the *geometry* of a Riemannian manifold (with boundary) is "determined" by the **spectrum**

* of eigenvalues of the Laplacian, with either Dirichlet or Neumann (or mixed) boundary condition
* of lengths of closed geodesics or billiard trajectories
* or variations thereof

specifically what is known about individual eigenvalues or lengths (rational or algebraic), or collectively as a set. Inspired by and modeled on the databases of integer sequences and of L-functions and modular forms.

## Spectral Geometry Pages

The static project index is published from `docs/`:

https://liuyao12.github.io/spectral-geometry/

The tetrahedron billiards inventory page is:

https://liuyao12.github.io/spectral-geometry/tetra-billiards.html

The companion exposition on closed billiard trajectories and singular
normal-cone representatives is:

https://liuyao12.github.io/spectral-geometry/closed-billiards.html

To refresh the published ordinary billiard inventory after a search update, run:

```bash
python3 -m pip install --user z3-solver
python3 scripts/run_tetra_exhaustive_inventory.py --max-period 40
```

This writes a compact exact source such as
`data/tetra_exhaustive_period40.json`, keeps a large resumable local checkpoint
next to it, and rebuilds `docs/data/tetra/billiards_inventory.json`.

To rebuild the static inventory directly from existing exact/exploratory
sources:

```bash
python3 scripts/update_tetra_billiards_inventory.py \
  --source data/tetra_exhaustive_period40.json \
  --source data/tetra_exploratory_paths.json \
  --out docs/data/tetra/billiards_inventory.json
```

The generated tetrahedron billiards inventory currently records ordinary paths
checked exhaustively through level 40, plus exactified paths found by the
exploratory shooting search. The published page also loads a small companion
file of singular normal-cone representatives imported from the Observable
notebook visualization.

To run another non-exhaustive shooting pass:

```bash
python3 scripts/explore_tetra_billiards.py \
  --trials 1500 \
  --max-bounces 90 \
  --max-period 80 \
  --out data/tetra_exploratory_paths.json

python3 scripts/update_tetra_billiards_inventory.py \
  --source docs/data/tetra/billiards_inventory.json \
  --source data/tetra_exploratory_paths.json \
  --out docs/data/tetra/billiards_inventory.json
```

The exploratory script uses `numpy`; `scipy` enables the optional local
least-squares relaxation, and `sympy` is used by the exact verifier.
