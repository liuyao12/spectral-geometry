import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import scipy.sparse as sp
import meshpy.triangle as triangle
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- Standard Functions (Omitted) ---
def create_mesh(points, max_area=0.015):
    info = triangle.MeshInfo(); info.set_points(points.tolist())
    segments = [(i,(i+1)%len(points)) for i in range(len(points))]; info.set_facets(segments)
    try:
        area = np.abs(np.sum(points[:,0] * np.roll(points[:,1], -1) - np.roll(points[:,0], -1) * points[:,1])) / 2.0
        dynamic_max_area = area / 1500 if area > 0 else max_area
    except: dynamic_max_area = max_area
    mesh = triangle.build(info, max_volume=dynamic_max_area, min_angle=30)
    return np.array(mesh.points), np.array(mesh.elements)

def laplacian_fem(pts, tris):
    n = len(pts); A = sp.lil_matrix((n,n)); M = sp.lil_matrix((n,n))
    for tri in tris:
        p = pts[tri]; B = np.array([[1,p[0,0],p[0,1]], [1,p[1,0],p[1,1]], [1,p[2,0],p[2,1]]])
        area = 0.5*abs(np.linalg.det(B)); C = np.linalg.inv(B); grads = C[1:, :]
        localA = area * grads.T @ grads; localM = (area/12)*np.array([[2,1,1],[1,2,1],[1,1,2]])
        for i in range(3):
            for j in range(3): A[tri[i], tri[j]] += localA[i,j]; M[tri[i], tri[j]] += localM[i,j]
    return A.tocsr(), M.tocsr()

def a2_to_xy(a, b, c):
    x = a + 0.5*b; y = (np.sqrt(3)/2)*b; return np.column_stack([x, y])

def solve_eigenmodes_nd(A_free, M_free, pts, free, k_total):
    if A_free.shape[0] < k_total: k_total = A_free.shape[0] - 1
    if k_total <= 0: return np.full(24, np.nan), [np.zeros(len(pts))] * 24
        
    vals, vecs = spla.eigsh(A_free, M=M_free, k=k_total, sigma=0.0, which='LM')
    
    U_modes = []
    for i in range(k_total):
        u = np.zeros(len(pts)); u[free] = vecs[:,i]; U_modes.append(u)
    
    while len(vals) < 24: vals = np.append(vals, np.nan); U_modes.append(np.zeros(len(pts)))
        
    return vals[:24], U_modes[:24]

# --- Core Geometry and Setup ---
verts_abc = np.array([
    [0,-3,3],[1,-3,2],[2,-4,2],[3,-3,0], [2,-1,-1],[1,1,-2],[-1,2,-1],[-1,3,-2],
    [-2,4,-2],[-3,3,0],[-2,1,1],[-3,1,2], [-3,0,3],[-1,-1,2]
])
L_A_orig = 1.0; L_B_orig = np.sqrt(3.0)
N_edges = len(verts_abc)
verts_xy_orig = a2_to_xy(verts_abc[:,0], verts_abc[:,1], verts_abc[:,2])
L_orig = np.sqrt(np.sum((np.roll(verts_xy_orig, -1, axis=0) - verts_xy_orig)**2, axis=1))
is_B_type = np.isclose(L_orig, L_B_orig)
unit_directions = (np.roll(verts_xy_orig, -1, axis=0) - verts_xy_orig) / L_orig[:, np.newaxis]
k_modes_total = 24

# --- Function to calculate results for a given BC set ---
def run_bc_set(bc_name, type_A_bc, type_B_bc):
    
    edge_types_bc = np.where(is_B_type, type_B_bc, type_A_bc)
    
    results = {'name': bc_name, 'data': {}}
    
    for alpha in [0.0, 1.0]: # Only need Turtle (0.0) and Hat (1.0)
        L_A = (1 - alpha) * L_A_orig + alpha * L_B_orig
        L_B = (1 - alpha) * L_B_orig + alpha * L_A_orig
        L_current = np.where(is_B_type, L_B, L_A)
        
        verts = np.zeros((N_edges, 2)); current_pos = np.array([0.0, 0.0])
        for i in range(N_edges): verts[i] = current_pos; current_pos += L_current[i] * unit_directions[i]

        pts, tris = create_mesh(verts)
        A, M = laplacian_fem(pts, tris)

        is_dirichlet = np.zeros(len(pts), bool)
        for i in range(N_edges):
            if edge_types_bc[i] == 'D':
                v_start = verts[i]; v_end = verts[(i + 1) % N_edges]
                vec = v_end - v_start; L2 = np.dot(vec, vec)
                for idx, p in enumerate(pts):
                    t = np.cross(vec, p - v_start); dot = np.dot(p - v_start, vec)
                    if abs(t) < 1e-8 and 0 - 1e-8 <= dot <= L2 + 1e-8: is_dirichlet[idx] = True

        free = np.where(~is_dirichlet)[0]
        vals, U_modes = solve_eigenmodes_nd(A[free,:][:,free], M[free,:][:,free], pts, free, k_modes_total)
        
        results['data'][alpha] = {
            'verts': verts, 'pts': pts, 'tris': tris, 
            'vals': vals, 'modes': U_modes, 'bc': edge_types_bc
        }
    return results

# --- EXECUTE CALCULATIONS ---

# 1. Original BCs: L1 -> N, L_sqrt3 -> D
results_original = run_bc_set("Original N/D", 'N', 'D')

# 2. Switched BCs: L1 -> D, L_sqrt3 -> N
results_switched = run_bc_set("Switched D/N", 'D', 'N')

# --- Visualization Setup ---

# Mode indices (0-based):
I_MODE5 = 4; I_MODE16 = 15; I_MODE18 = 17; I_MODE21 = 20

# Define the 6 plots to visualize
plot_config = [
    # ROW 1: Original N/D
    {'res': results_original, 'alpha': 0.0, 'mode_i': I_MODE5, 'mode_N': 5, 'geom': 'Turtle'},
    {'res': results_original, 'alpha': 0.0, 'mode_i': I_MODE21, 'mode_N': 21, 'geom': 'Turtle'},
    {'res': results_original, 'alpha': 1.0, 'mode_i': I_MODE16, 'mode_N': 16, 'geom': 'Hat'},
    
    # ROW 2: Switched D/N
    {'res': results_switched, 'alpha': 0.0, 'mode_i': I_MODE21, 'mode_N': 21, 'geom': 'Turtle'},
    {'res': results_switched, 'alpha': 1.0, 'mode_i': I_MODE18, 'mode_N': 18, 'geom': 'Hat'},
    {'res': results_switched, 'alpha': 1.0, 'mode_i': I_MODE5, 'mode_N': 5, 'geom': 'Hat (Ref)'},
]

all_plots_data = []
vmax = 0.0

for p in plot_config:
    data = p['res']['data'][p['alpha']]
    mode = data['modes'][p['mode_i']]
    all_plots_data.append(data)
    vmax = max(vmax, np.max(np.abs(mode)))

if vmax == 0.0: vmax = 1.0

# Visualization color map (Fixed definition for all cases)
# Original N/D: D (red), N (k); Switched D/N: D (k), N (red)
COLOR_MAPS = {
    ('D', 'Original N/D'): 'red', ('N', 'Original N/D'): 'k',
    ('D', 'Switched D/N'): 'k',   ('N', 'Switched D/N'): 'red',
}

plt.figure(figsize=(18, 10))

for idx, p in enumerate(plot_config):
    ax = plt.subplot(2, 3, idx + 1)
    
    geom_data = p['res']['data'][p['alpha']]
    mode_field = geom_data['modes'][p['mode_i']]
    mode_lambda = geom_data['vals'][p['mode_i']]
    
    ax.tricontourf(geom_data['pts'][:,0], geom_data['pts'][:,1], geom_data['tris'], mode_field, levels=30, cmap='coolwarm', 
                   vmin=-vmax, vmax=vmax)
    
    # Draw boundaries
    verts = geom_data['verts']
    for k in range(N_edges):
        etype = geom_data['bc'][k]
        color = COLOR_MAPS[(etype, p['res']['name'])]
        
        v_start = verts[k]; v_end = verts[(k + 1) % N_edges]
        
        ax.plot([v_start[0], v_end[0]], [v_start[1], v_end[1]], 
                color=color, 
                linestyle='-' if etype=='D' else '--',
                lw=2.5, alpha=0.9) 

    ax.set_aspect('equal')
    ax.axis('off')
    
    # Titles
    bc_label = f"({p['res']['name']})"
    title_lambda = f"$\\lambda={mode_lambda:.3f}$"
    
    if idx == 0:
        ax.text(-0.2, 0.5, "Original N/D", transform=ax.transAxes, rotation=90, fontsize=12, ha='right', va='center')
    elif idx == 3:
        ax.text(-0.2, 0.5, "Switched D/N", transform=ax.transAxes, rotation=90, fontsize=12, ha='right', va='center')

    ax.set_title(f"Mode {p['mode_N']} ({p['geom']})\n{title_lambda}", fontsize=12)
    
plt.tight_layout(rect=[0.02, 0, 1, 0.96])
plt.suptitle("Comparison of Key Regularity Modes Across Boundary Conditions and Geometry Extremes", fontsize=16)
plt.show()
