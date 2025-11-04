import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# === Convert A2 coordinates (a,b,c) to 2D ===
def a2_to_xy(a, b, c):
    assert np.allclose(a + b + c, 0)
    x = a + 0.5*b
    y = (np.sqrt(3)/2)*b
    return np.column_stack([x, y])

# === Your polygon in A2 lattice coordinates ===
verts_abc = np.array([
    [0,-3,3],[1,-3,2],[2,-4,2],[3,-3,0],
    [2,-1,-1],[1,1,-2],[-1,2,-1],[-1,3,-2],
    [-2,4,-2],[-3,3,0],[-2,1,1],[-3,1,2],
    [-3,0,3],[-1,-1,2]
])
verts_xy = a2_to_xy(verts_abc[:,0], verts_abc[:,1], verts_abc[:,2])
N_edges = len(verts_xy)

# === Edge types matching the 14 edges ===
edge_types = ["N","N","D","D","D","D","N","N","D","D","N","N","D","D"]

# === Create lattice-aligned triangular mesh via simple Delaunay refinement ===
# Here we do a uniform subdivision by adding extra points along edges (optional)
# For simplicity, just use triangle centroid mesh
def create_mesh(points, max_area=0.02):
    import meshpy.triangle as triangle
    info = triangle.MeshInfo()
    info.set_points(points.tolist())
    segments = [(i,(i+1)%len(points)) for i in range(len(points))]
    info.set_facets(segments)
    mesh = triangle.build(info, max_volume=max_area)
    pts = np.array(mesh.points)
    tris = np.array(mesh.elements)
    return pts, tris

pts, tris = create_mesh(verts_xy, max_area=0.02)

# === FEM assembly (linear triangles) ===
def laplacian_fem(pts, tris):
    n = len(pts)
    A = sp.lil_matrix((n,n))
    M = sp.lil_matrix((n,n))
    for tri in tris:
        p = pts[tri]
        B = np.array([[1,p[0,0],p[0,1]],
                      [1,p[1,0],p[1,1]],
                      [1,p[2,0],p[2,1]]])
        area = 0.5*abs(np.linalg.det(B))
        C = np.linalg.inv(B)
        grads = C[1:, :]
        localA = area * grads.T @ grads
        localM = (area/12)*np.array([[2,1,1],[1,2,1],[1,1,2]])
        for i in range(3):
            for j in range(3):
                A[tri[i], tri[j]] += localA[i,j]
                M[tri[i], tri[j]] += localM[i,j]
    return A.tocsr(), M.tocsr()

A, M = laplacian_fem(pts, tris)

# === Identify boundary nodes for mixed BCs ===
is_dirichlet = np.zeros(len(pts), bool)
is_neumann = np.zeros(len(pts), bool)

for (i, v_start), v_end, etype in zip(enumerate(verts_xy), np.roll(verts_xy,-1,axis=0), edge_types):
    vec = v_end - v_start
    for idx, p in enumerate(pts):
        t = np.cross(vec, p - v_start)
        dot = np.dot(p - v_start, vec)
        if abs(t) < 1e-9 and 0 <= dot <= np.dot(vec,vec):
            if etype=="D":
                is_dirichlet[idx] = True
            else:
                is_neumann[idx] = True

free = np.where(~is_dirichlet)[0]

# === Solve generalized eigenproblem with Dirichlet on selected edges ===
A_free = A[free,:][:,free]
M_free = M[free,:][:,free]
vals, vecs = spla.eigsh(A_free, M=M_free, k=12, sigma=0.0, which='LM')

# === Visualization: first 6 eigenmodes in subplots ===
nplot = 6
cols = 3
rows = (nplot + cols - 1)//cols
plt.figure(figsize=(4*cols,3*rows))
for i in range(nplot):
    u = np.zeros(len(pts))
    u[free] = vecs[:,i]
    ax = plt.subplot(rows, cols, i+1)
    tcf = ax.tricontourf(pts[:,0], pts[:,1], tris, u, levels=30, cmap='coolwarm')
    ax.triplot(pts[:,0], pts[:,1], tris, lw=0.3, color='k', alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title(f"Mode {i+1}, λ={vals[i]:.3f}")
    ax.axis('off')
plt.tight_layout()
plt.show()
