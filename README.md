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

To refresh the published ordinary billiard inventory after a search update, run:

```bash
python3 scripts/update_tetra_billiards_inventory.py \
  --source /path/to/tetra_frontier_checkpoint_period60.json \
  --source data/tetra_exploratory_paths.json \
  --out docs/data/tetra/billiards_inventory.json
```

The tetrahedron billiards table currently records ordinary paths checked
exhaustively through level 40, plus exactified paths found by the exploratory
shooting search.

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
