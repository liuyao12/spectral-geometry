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
  --out docs/data/tetra/billiards_inventory.json
```
