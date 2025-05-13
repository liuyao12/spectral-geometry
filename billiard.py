# --- Changes applied to the previous code block ---

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import math
from collections import defaultdict
import random # For shuffling
import os # Added for directory creation and path checking
import string # Added for filename suffix generation

# Global list for new simulation instances (prevent garbage collection)
simulations = []
# Global list to hold references to the stable configuration figures
stable_figures = []

# ----------------------------
# Global Parameters & Constants
# ----------------------------
MERGE_THRESHOLD = 0.005
DT = 0.01
DAMPING = 0.1
FORCE_SCALE = 1.0
# --- Convergence Criteria ---
# Threshold for change in total graph length to be considered "still"
# Based on .4f display format, 1e-4 means no change in displayed digits.
LENGTH_STILL_THRESHOLD = 1e-8 # <<< NEW
# Higher count means the system must remain still for longer
STILL_COUNT_REQUIRED = 150 # (Was 150, kept the same duration requirement)
# --- Other Constants ---
NON_TRIVIAL_LENGTH_THRESHOLD = 1e-4
MAX_RESTARTS = 5             # Max restarts for simulation *failures* (empty/trivial)
MAX_INIT_ATTEMPTS = 3        # Max attempts for the *initialization* phase itself
EPSILON = np.finfo(float).eps * 100 # Use numpy's float info
SAVE_BASE_DIR = "billiards_results"
# STILL_THRESHOLD = 1e-9 # <<< REMOVED (replaced by LENGTH_STILL_THRESHOLD)



# ----------------------------
# Polygon Generation Utilities
# ----------------------------
def create_regular_polygon(n_sides, scale=1.0):
    """
    Create a regular n‑gon with side length 1:
      • vertex 0 at (0, 0)
      • vertex 1 at (1, 0)
      • vertex 2 in the first quadrant
    Returns an (n_sides x 2) array of vertex coordinates.
    """
    if n_sides < 3:
        raise ValueError("Polygon needs at least 3 sides.")

    # side length = 1 * scale
    s = 1.0 * scale
    # circumradius for side length s: R = s / (2 * sin(pi / n))
    R = s / (2.0 * np.sin(np.pi / n_sides))

    # center lies above midpoint of v0->v1: midpoint = (s/2, 0)
    cx = s / 2.0
    cy = np.sqrt(max(R**2 - (s/2.0)**2, 0.0))

    # angle to first vertex (v0 at origin): from center to origin
    theta0 = np.arctan2(-cy, -cx)

    # build all vertices
    angles = theta0 + np.arange(n_sides) * (2.0 * np.pi / n_sides)
    xs = cx + R * np.cos(angles)
    ys = cy + R * np.sin(angles)
    coords = np.column_stack((xs, ys))

    return coords

# ----------------------------
# Irregular / Custom Shapes
# First vertex at origin, shapes lie in first quadrant
# ----------------------------
# Square (3rd vertex at (1,1))
square_coords = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])

# Rhombus (60° angles)
rhombus_coords = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.5, np.sqrt(3) / 2],
    [0.5, np.sqrt(3) / 2]
])

# L-shape (missing upper-right corner)
l_shape_coords = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.5],
    [0.5, 0.5],
    [0.5, 1.0],
    [0.0, 1.0]
])

# ----------------------------
# DEFAULT_SHAPES DICTIONARY
# ----------------------------
DEFAULT_SHAPES = {
    # Regular polygons with side length 1:
    "Triangle": create_regular_polygon(n_sides=3),
    "SquareReg": create_regular_polygon(n_sides=4),
    "Pentagon": create_regular_polygon(n_sides=5),
    "Hexagon": create_regular_polygon(n_sides=6),
    "Octagon": create_regular_polygon(n_sides=8),

    # Irregular polygons in first quadrant:
    "Square": square_coords,
    "Rhombus": rhombus_coords,
    "L-Shape": l_shape_coords,
}

# ----------------------------
# Geometry Utility Functions
# (No changes in this section)
# ----------------------------
def find_segment_intersection(p1, q1, p2, q2):
    r = q1 - p1; s = q2 - p2; r_cross_s = np.cross(r, s)
    q_minus_p = p2 - p1; q_minus_p_cross_r = np.cross(q_minus_p, r); epsilon_intersect = 1e-9
    if abs(r_cross_s) < epsilon_intersect: return None
    t = np.cross(q_minus_p, s) / r_cross_s; u = q_minus_p_cross_r / r_cross_s
    if -epsilon_intersect <= t <= 1.0 + epsilon_intersect and -epsilon_intersect <= u <= 1.0 + epsilon_intersect: return p1 + t * r
    return None

def point_to_segment_dist_proj(P, A, B):
    AP = P - A; AB = B - A; AB_squared = np.dot(AB, AB)
    if AB_squared < EPSILON: return np.linalg.norm(P - A), A
    t = np.dot(AP, AB) / AB_squared; t_clamped = np.clip(t, 0, 1)
    proj = A + t_clamped * AB; d = np.linalg.norm(P - proj)
    return d, proj

def project_point_to_polygon(P, poly):
    best_d = np.inf; best_proj = P; best_edge_index = -1; n = len(poly)
    if n < 2: return P, 0, -1
    for i in range(n):
        A, B = poly[i], poly[(i + 1) % n]; d, proj = point_to_segment_dist_proj(P, A, B)
        if d < best_d: best_d = d; best_proj = proj; best_edge_index = i
    return best_proj, best_d, best_edge_index

def is_point_in_polygon(P, poly, tolerance=1e-9):
    n = len(poly);
    if n < 3: return False
    min_xy = np.min(poly, axis=0); max_xy = np.max(poly, axis=0)
    if not (min_xy[0] - tolerance <= P[0] <= max_xy[0] + tolerance and min_xy[1] - tolerance <= P[1] <= max_xy[1] + tolerance): return False
    for i in range(n):
        A, B = poly[i], poly[(i + 1) % n]
        if np.cross(B - A, P - A) < -tolerance: return False
    return True

# ----------------------------
# Simulation Class
# ----------------------------
class Simulation:
    """
    Simulates interacting points within a polygon boundary.
    Convergence uses total graph length change threshold (LENGTH_STILL_THRESHOLD)
    held for a duration (STILL_COUNT_REQUIRED).
    """
    def __init__(self, lambda_red=4, fig_num=1, shape_name="Hexagon", polygon_coords=None, red_options=None):
        self.lambda_red = lambda_red
        self.fig_num = fig_num
        self.shape_name = shape_name
        self.red_options = red_options

        # Polygon Setup
        if polygon_coords is not None: self.polygon = np.array(polygon_coords)
        elif shape_name in DEFAULT_SHAPES: self.polygon = DEFAULT_SHAPES[shape_name].copy()
        else: self.shape_name = "Hexagon"; self.polygon = DEFAULT_SHAPES["Hexagon"].copy(); print(f"Warn: Shape '{shape_name}' not found. Using Hexagon.")
        if len(self.polygon) < 3: raise ValueError("Polygon needs >= 3 vertices.")
        self.n_edges_poly = len(self.polygon); self.min_xy = np.min(self.polygon, axis=0); self.max_xy = np.max(self.polygon, axis=0)

        # State & Control Variables
        self.vertices = []; self.n_blue = 0; self.n_red = 0
        self.prev_total_length = 0.0 # <<< NEW: Track previous total length
        self.still_frame_count = 0; self.frame_count = 0 # Frame count still useful for debug/performance
        self.accumulated_length_change = 0.0
        self.is_running = False; self.sim_fail_restart_attempts = 0; self.max_sim_fail_restarts = MAX_RESTARTS

        # Plotting Variables
        self.fig = None; self.ax = None; self.anim = None; self.all_connections_line = None
        self.blue_scatter = None; self.red_scatter = None; self.poly_patch = None
        self.ax_restart_random = None; self.button_restart_random = None; self.stable_config_count = 0
        self._start_simulation()

    def _setup_figure(self):
        # (No changes needed in _setup_figure)
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(num=f"Simulation Runner {self.fig_num} ({self.shape_name})")
            plt.subplots_adjust(bottom=0.2)
            self.all_connections_line, = self.ax.plot([], [], 'k-', lw=1, zorder=1, label='connections')
            self.blue_scatter = plt.Line2D([], [], color='blue', marker='o', linestyle='None', ms=5, zorder=3, label='blue_pts'); self.ax.add_line(self.blue_scatter)
            self.red_scatter = plt.Line2D([], [], color='red', marker='o', linestyle='None', ms=5, zorder=3, label='red_pts'); self.ax.add_line(self.red_scatter)
            self.poly_patch = plt.Polygon(self.polygon, fill=False, color='grey', lw=2, zorder=0, label='boundary'); self.ax.add_patch(self.poly_patch)
            self.ax.set_aspect('equal')
            x_pad = (self.max_xy[0] - self.min_xy[0]) * 0.1 + 0.1; y_pad = (self.max_xy[1] - self.min_xy[1]) * 0.1 + 0.1
            self.ax.set_xlim(self.min_xy[0] - x_pad, self.max_xy[0] + x_pad); self.ax.set_ylim(self.min_xy[1] - y_pad, self.max_xy[1] + y_pad)
            self.ax.set_title("Figure Setup...")
            button_width = 0.25; button_height = 0.06; button_ypos = 0.05; button_xpos = 0.5 - button_width / 2
            try:
                self.ax_restart_random = plt.axes([button_xpos, button_ypos, button_width, button_height])
                self.button_restart_random = Button(self.ax_restart_random, "Restart Random")
                self.button_restart_random.on_clicked(self.restart_from_scratch_button)
            except ValueError as e: print(f"Warning: Error creating restart button: {e}"); self.ax_restart_random=None; self.button_restart_random=None
        else: # Reuse existing figure
            self.fig.canvas.get_tk_widget().winfo_toplevel().title(f"Simulation Runner {self.fig_num} ({self.shape_name})")
            self.all_connections_line.set_data([], []); self.blue_scatter.set_data([], []); self.red_scatter.set_data([], [])
            current_poly_in_axes = next((p for p in self.ax.patches if p.get_label() == 'boundary'), None)
            if current_poly_in_axes: current_poly_in_axes.set_xy(self.polygon)
            elif self.poly_patch: self.ax.add_patch(self.poly_patch)
            x_pad = (self.max_xy[0] - self.min_xy[0]) * 0.1 + 0.1; y_pad = (self.max_xy[1] - self.min_xy[1]) * 0.1 + 0.1
            self.ax.set_xlim(self.min_xy[0] - x_pad, self.max_xy[0] + x_pad); self.ax.set_ylim(self.min_xy[1] - y_pad, self.max_xy[1] + y_pad)
            self.ax.set_title(f"Figure Cleared... ({self.shape_name})")
            if self.ax_restart_random and not self.ax_restart_random.axison: self.ax_restart_random.set_visible(True)

    def _clear_plot(self):
        # (No changes needed in _clear_plot)
        if self.ax:
            self.all_connections_line.set_data([], []); self.blue_scatter.set_data([], []); self.red_scatter.set_data([], [])
            self.ax.set_title(f"Initializing... ({self.shape_name})")
        return self.all_connections_line, self.blue_scatter, self.red_scatter

    def _add_edge(self, v1_id, v2_id, vertices_map):
        # (No changes needed in _add_edge)
        if v1_id == v2_id: return False
        if v1_id in vertices_map and v2_id in vertices_map:
            v1 = vertices_map[v1_id]; v2 = vertices_map[v2_id]
            v1.setdefault('neighbors', set()); v2.setdefault('neighbors', set())
            if v2_id not in v1['neighbors']: v1['neighbors'].add(v2_id); v2['neighbors'].add(v1_id); return True
        return False

    def _remove_edge(self, v1_id, v2_id, vertices_map):
        # (No changes needed in _remove_edge)
        removed = False
        if v1_id in vertices_map and v2_id in vertices_map:
            v1 = vertices_map[v1_id]; v2 = vertices_map[v2_id]
            if 'neighbors' in v1 and v2_id in v1['neighbors']: v1['neighbors'].discard(v2_id); removed = True
            if 'neighbors' in v2 and v1_id in v2['neighbors']: v2['neighbors'].discard(v1_id); removed = True
        return removed

    def init_state(self):
        # (No changes needed in init_state)
        self.vertices = []; blue_vertices_temp = []; red_vertices_temp = []
        blue_vertices_temp = []
        total_blue = 0
        for edge_idx in range(self.n_edges_poly):
            A, B = self.polygon[edge_idx], self.polygon[(edge_idx + 1) % self.n_edges_poly]

            # pick between 1 and 4 points this edge
            num_blue = np.random.randint(1, 8)      # 1,2,3 or 4
            total_blue += num_blue

            # random positions along [A→B]
            ratios = np.random.uniform(0.05, 0.95, size=num_blue)
            ratios.sort()
            for ratio in ratios:
                pos = A + ratio * (B - A)
                blue_vertices_temp.append({
                    'pos': pos.copy(),
                    'color': 'blue',
                    'edge': edge_idx,
                    'neighbors': set()
                })

        self.n_blue = total_blue
        # pick the number of reds
        if self.red_options:
            # choose exactly one of the options
            n_red = random.choice(self.red_options)
        else:
            # old behavior: Poisson around lambda_red
            n_red = np.random.poisson(lam=self.lambda_red)
        # enforce minimum of 2 and evenness
        self.n_red = max(2, n_red)
        min_xy, max_xy = self.min_xy, self.max_xy; attempts, max_attempts = 0, self.n_red * 200
        min_dim = min(max_xy[0]-min_xy[0], max_xy[1]-min_xy[1]); inset = max(min_dim * 0.1, 0.05)
        while len(red_vertices_temp) < self.n_red and attempts < max_attempts:
            if max_xy[0] - min_xy[0] <= 2 * inset or max_xy[1] - min_xy[1] <= 2 * inset: break
            candidate = np.array([np.random.uniform(min_xy[0] + inset, max_xy[0] - inset), np.random.uniform(min_xy[1] + inset, max_xy[1] - inset)])
            if is_point_in_polygon(candidate, self.polygon, tolerance=inset/2): red_vertices_temp.append({'pos': candidate.copy(), 'color': 'red', 'neighbors': set()})
            attempts += 1
        self.n_red = len(red_vertices_temp)
        if self.n_red < 2: print(f"  Init fail: Placed only {self.n_red} red."); self.vertices = []; return False
        self.vertices = blue_vertices_temp + red_vertices_temp
        vertices_map = {id(v): v for v in self.vertices}; all_vertex_ids = list(vertices_map.keys())
        if len(all_vertex_ids) > 1:
            shuffled_ids = all_vertex_ids[:]; random.shuffle(shuffled_ids)
            for i in range(len(shuffled_ids)): self._add_edge(shuffled_ids[i], shuffled_ids[(i + 1) % len(shuffled_ids)], vertices_map)
        if not self.assign_red_matching(): print("  Init fail: assign_red_matching failed."); self.vertices = []; return False
        for v in self.vertices:
            if v['color'] == 'blue' and np.all(np.isfinite(v['pos'])):
                proj_pos_poly, _, edge_index_poly = project_point_to_polygon(v['pos'], self.polygon)
                v['pos'] = proj_pos_poly; v['edge'] = edge_index_poly
            v.setdefault('neighbors', set())
        return True

    def assign_red_matching(self):
        # (No changes needed in assign_red_matching)
        red_verts = [v for v in self.vertices if v['color'] == 'red']; n_red = len(red_verts)
        if n_red < 2: return True
        red_ids = [id(v) for v in red_verts]; vertices_map = {id(v): v for v in self.vertices}
        MAX_ITERATIONS = n_red * n_red * 2; iterations = 0; target_degree = 3
        while iterations < MAX_ITERATIONS:
            iterations += 1
            low_degree_reds = [v_id for v_id in red_ids if v_id in vertices_map and len(vertices_map[v_id].get('neighbors', set())) < target_degree]
            if not low_degree_reds: break
            added_edge_this_iteration = False; possible_pairs = []
            if len(low_degree_reds) >= 2:
                 for i in range(len(low_degree_reds)):
                     for j in range(i + 1, len(low_degree_reds)): possible_pairs.append(tuple(sorted((low_degree_reds[i], low_degree_reds[j]))))
            other_reds = [rid for rid in red_ids if rid not in low_degree_reds]
            for v_low in low_degree_reds:
                for v_other in other_reds: possible_pairs.append(tuple(sorted((v_low, v_other))))
            possible_pairs = list(set(possible_pairs)); random.shuffle(possible_pairs)
            for v1_id, v2_id in possible_pairs:
                if v1_id in vertices_map and v2_id in vertices_map and v2_id not in vertices_map[v1_id].get('neighbors', set()):
                    if self._add_edge(v1_id, v2_id, vertices_map): added_edge_this_iteration = True; break
            if not added_edge_this_iteration:
                found_any_new_edge = False
                all_possible_red_pairs = [tuple(sorted((red_ids[i], red_ids[j]))) for i in range(n_red) for j in range(i + 1, n_red)]
                random.shuffle(all_possible_red_pairs)
                for r1_id, r2_id in all_possible_red_pairs:
                     if r1_id in vertices_map and r2_id in vertices_map and r2_id not in vertices_map[r1_id].get('neighbors', set()):
                          if self._add_edge(r1_id, r2_id, vertices_map): added_edge_this_iteration = True; found_any_new_edge = True; break
                if not found_any_new_edge: break
        final_low_degree_reds = [v_id for v_id in red_ids if v_id in vertices_map and len(vertices_map[v_id].get('neighbors', set())) < target_degree]
        if iterations >= MAX_ITERATIONS and final_low_degree_reds: print(f"    Warn: Max iter in red match. {len(final_low_degree_reds)} remain low deg."); return False
        if final_low_degree_reds: print(f"    Warn: Cannot reach target deg for {len(final_low_degree_reds)} red."); return False
        return True

    def update_state(self):
        # (No changes needed in update_state)
        if not self.vertices: return
        n = len(self.vertices); forces = np.zeros((n, 2)); vertices_map = {id(v): v for v in self.vertices}
        for i, v in enumerate(self.vertices):
            pos = v['pos'];
            if not np.all(np.isfinite(pos)): continue
            force_sum_unit_vectors = np.zeros(2)
            for neighbor_id in v.get('neighbors', set()):
                if neighbor_id in vertices_map:
                    neighbor_pos = vertices_map[neighbor_id]['pos']
                    if not np.all(np.isfinite(neighbor_pos)): continue
                    vec = neighbor_pos - pos; dist = np.linalg.norm(vec)
                    if dist > EPSILON: force_sum_unit_vectors += vec / dist
            forces[i] = force_sum_unit_vectors * FORCE_SCALE
        target_positions = [None] * n; proximity_threshold_sq = (2.5 * MERGE_THRESHOLD)**2
        for i, v in enumerate(self.vertices):
            if not np.all(np.isfinite(v['pos'])): target_positions[i] = v['pos']; continue
            step_reduction_factor = 1.0
            if n > 1:
                min_dist_sq = float('inf')
                for neighbor_id in v.get('neighbors', set()):
                    if neighbor_id in vertices_map:
                         neighbor_pos = vertices_map[neighbor_id]['pos']
                         if np.all(np.isfinite(neighbor_pos)): min_dist_sq = min(min_dist_sq, np.sum((v['pos'] - neighbor_pos)**2))
                if min_dist_sq < proximity_threshold_sq: step_reduction_factor = 0.5
            dPos = forces[i] * DT * (1.0 - DAMPING) * step_reduction_factor;
            # ── clamp to at most half the shortest neighbor‐distance ──
            if v.get('neighbors'):
                # compute the minimum distance to any neighbor
                dists = []
                for nbr in v['neighbors']:
                    if nbr in vertices_map:
                        nbr_pos = vertices_map[nbr]['pos']
                        if np.all(np.isfinite(nbr_pos)):
                            dists.append(np.linalg.norm(v['pos'] - nbr_pos))
                if dists:
                    max_step = 0.5 * min(dists)
                    step_norm = np.linalg.norm(dPos)
                    if step_norm > max_step:
                        dPos = dPos / step_norm * max_step
            target_positions[i] = v['pos'] + dPos
        for i, v in enumerate(self.vertices):
             target_pos = target_positions[i]
             if target_pos is None or not np.all(np.isfinite(target_pos)): final_pos = v['pos']
             elif v['color'] == 'blue': proj_pos, _, edge_index = project_point_to_polygon(target_pos, self.polygon); final_pos = proj_pos; v['edge'] = edge_index
             else:
                 if not is_point_in_polygon(target_pos, self.polygon, tolerance=EPSILON): proj_pos, _, _ = project_point_to_polygon(target_pos, self.polygon); final_pos = proj_pos
                 else: final_pos = target_pos
             v['pos'] = final_pos

    def _get_neighbors_graph(self, vertex_id, vertices_map):
        # (No changes needed in _get_neighbors_graph)
        return set(vertices_map.get(vertex_id, {}).get('neighbors', set()))

    def check_and_merge(self):
        # (No changes needed in check_and_merge logic)
        if len(self.vertices) < 2: return False
        overall_merged_flag = False; max_passes = len(self.vertices) // 2 + 2
        for _ in range(max_passes):
            made_merge_this_pass = False; n = len(self.vertices)
            if n < 2: break
            vertices_map = {id(v): v for v in self.vertices}; merge_candidates = []; processed_pairs = set()
            for v1_id, v1 in vertices_map.items():
                if not np.all(np.isfinite(v1['pos'])): continue
                for v2_id in v1.get('neighbors', set()):
                    pair_key = tuple(sorted((v1_id, v2_id)))
                    if v2_id not in vertices_map or pair_key in processed_pairs: continue
                    v2 = vertices_map[v2_id];
                    if not np.all(np.isfinite(v2['pos'])): continue
                    processed_pairs.add(pair_key); dist = np.linalg.norm(v1['pos'] - v2['pos'])
                    if dist < MERGE_THRESHOLD: merge_candidates.append({'v1': v1, 'v2': v2, 'dist': dist})
            if not merge_candidates: break
            merge_candidates.sort(key=lambda x: x['dist'])
            vertices_involved_ids = set(); vertices_to_remove_ids = set()
            for merge_info in merge_candidates:
                v1, v2 = merge_info['v1'], merge_info['v2']; v1_id, v2_id = id(v1), id(v2)
                # if v1['color']=='red' and v2['color']=='red': continue
                if v1_id in vertices_involved_ids or v2_id in vertices_involved_ids: continue
                if v1_id not in vertices_map or v2_id not in vertices_map: continue
                if v1['color'] == 'blue' and v2['color'] == 'red': v_kept, v_removed = v1, v2
                elif v2['color'] == 'blue' and v1['color'] == 'red': v_kept, v_removed = v2, v1
                else: v_kept, v_removed = v1, v2
                kept_id, removed_id = id(v_kept), id(v_removed)
                neighbors_kept = self._get_neighbors_graph(kept_id, vertices_map); neighbors_removed = self._get_neighbors_graph(removed_id, vertices_map)
                all_neighbor_ids = (neighbors_kept | neighbors_removed) - {kept_id, removed_id}
                v_kept['pos'] = (v_kept['pos'] + v_removed['pos']) / 2.0
                if v_kept['color'] == 'blue': v_kept['pos'], _, v_kept['edge'] = project_point_to_polygon(v_kept['pos'], self.polygon)
                else:
                    if not is_point_in_polygon(v_kept['pos'], self.polygon): v_kept['pos'], _, _ = project_point_to_polygon(v_kept['pos'], self.polygon)
                v_kept['neighbors'] = set()
                for neighbor_id in all_neighbor_ids:
                    if neighbor_id in vertices_map:
                        neighbor_v = vertices_map[neighbor_id]
                        if 'neighbors' in neighbor_v: neighbor_v['neighbors'].discard(removed_id)
                        self._add_edge(kept_id, neighbor_id, vertices_map)
                vertices_to_remove_ids.add(removed_id); vertices_involved_ids.add(kept_id); vertices_involved_ids.add(removed_id)
                made_merge_this_pass = True; overall_merged_flag = True
            if vertices_to_remove_ids: self.vertices = [v for v in self.vertices if id(v) not in vertices_to_remove_ids]
            if not made_merge_this_pass: break
        self.n_blue = sum(1 for v in self.vertices if v['color'] == 'blue'); self.n_red = len(self.vertices) - self.n_blue
        return overall_merged_flag

    def get_total_length(self):
        # (No changes needed in get_total_length)
        if not self.vertices: return 0.0
        total_length = 0.0; processed_pairs = set(); vertices_map = {id(v): v for v in self.vertices}
        for v1_id, v1 in vertices_map.items():
            pos1 = v1['pos']
            if not np.all(np.isfinite(pos1)): continue
            for v2_id in v1.get('neighbors', set()):
                pair_key = tuple(sorted((v1_id, v2_id)))
                if v2_id in vertices_map and pair_key not in processed_pairs:
                    v2 = vertices_map[v2_id]; pos2 = v2['pos']
                    if np.all(np.isfinite(pos2)): total_length += np.linalg.norm(pos1 - pos2); processed_pairs.add(pair_key)
        return total_length

    # --- Plotting and Stats ---
    def update_plot(self):
        """Updates plot elements, displaying still_frame_count instead of frame_count."""
        if not self.ax or not self.all_connections_line or not self.blue_scatter or not self.red_scatter: return []
        vertices = self.vertices; artists = (self.all_connections_line, self.blue_scatter, self.red_scatter)
        if not vertices:
            for artist in artists: artist.set_data([], [])
            status = "Empty" if not self.is_running else "Initializing"
            # <<< Display still frame count even when empty/init >>>
            self.ax.set_title(f"Sim Runner {self.fig_num} ({self.shape_name}): {status} Still Fr:{self.still_frame_count}")
            if self.fig: self.fig.canvas.draw_idle()
            return artists
        segments_x, segments_y = [], []; processed_pairs = set(); vertices_map = {id(v): v for v in vertices}
        for v1_id, v1 in vertices_map.items():
            pos1 = v1['pos']
            if not np.all(np.isfinite(pos1)): continue
            for v2_id in v1.get('neighbors', set()):
                pair_key = tuple(sorted((v1_id, v2_id)))
                if v2_id in vertices_map and pair_key not in processed_pairs:
                    v2 = vertices_map[v2_id]; pos2 = v2['pos']
                    if np.all(np.isfinite(pos2)): segments_x.extend([pos1[0], pos2[0], np.nan]); segments_y.extend([pos1[1], pos2[1], np.nan]); processed_pairs.add(pair_key)
        self.all_connections_line.set_data(segments_x, segments_y)
        blue_pts_list = [v['pos'] for v in vertices if v['color'] == 'blue' and np.all(np.isfinite(v['pos']))]
        red_pts_list = [v['pos'] for v in vertices if v['color'] == 'red' and np.all(np.isfinite(v['pos']))]
        if blue_pts_list: blue_arr = np.array(blue_pts_list); self.blue_scatter.set_data(blue_arr[:, 0], blue_arr[:, 1])
        else: self.blue_scatter.set_data([], [])
        if red_pts_list: red_arr = np.array(red_pts_list); self.red_scatter.set_data(red_arr[:, 0], red_arr[:, 1])
        else: self.red_scatter.set_data([], [])
        total_len = self.get_total_length(); num_edges_plot = len(processed_pairs); status_str = "Running"
        if not self.is_running:
             current_title = self.ax.get_title()
             if "[" in current_title and "]" in current_title:
                reason = current_title[current_title.find('[')+1 : current_title.find(']')]
                frozen_reasons = ["Converged", "Error", "Empty", "Init Failed", "Collapsed", "Saved", "Trivial", "Max Restarts"]
                if any(fr in reason for fr in frozen_reasons): status_str = f"Frozen ({reason})"
                else: status_str = "Frozen"
             else: status_str = "Frozen"
        # <<< MODIFIED TITLE: Display still_frame_count >>>
        title = (f"Sim Runner {self.fig_num} ({self.shape_name}): B:{self.n_blue}, R:{self.n_red}, Edges:{num_edges_plot}, "
                 f"Len:{total_len:.4f} Still Fr:{self.still_frame_count} [{status_str}]")
        self.ax.set_title(title, fontsize=10)
        if self.fig: self.fig.canvas.draw_idle()
        return artists

    # --- Convergence and Animation Loop ---
    def check_convergence_and_validity(self, merged_this_frame):
        """
        Checks for simulation end conditions using total length change.
        Uses global LENGTH_STILL_THRESHOLD and STILL_COUNT_REQUIRED.
        Returns False if simulation should stop/restart.
        """
        # Calculate current length ONCE
        current_length = self.get_total_length()

        # --- Keep track of the previous length for comparison THIS frame ---
        # (We'll update self.prev_total_length for the *next* frame at the end)
        length_before_this_frame = self.prev_total_length

        # Check 1: Empty or Collapsed State (Failure)
        is_empty = not self.vertices
        is_collapsed = len(self.vertices) <= 1 or \
                       (len(self.vertices) > 1 and current_length < NON_TRIVIAL_LENGTH_THRESHOLD)

        if is_empty or is_collapsed:
            fail_reason = "Empty" if is_empty else "Collapsed"
            print(f"Window {self.fig_num} ({self.shape_name}): FAILED! State is {fail_reason}. Restart attempt {self.sim_fail_restart_attempts + 1}/{self.max_sim_fail_restarts}")
            self.sim_fail_restart_attempts += 1
            # Update prev_length before stopping (good practice, reflects final state)
            self.prev_total_length = current_length
            if self.sim_fail_restart_attempts < self.max_sim_fail_restarts:
                self._schedule_restart()
                self.freeze_simulation(f"{fail_reason} - Restarting")
            else:
                self.freeze_simulation(f"{fail_reason} - Max Restarts Reached")
            return False # Stop current run

        # Check 2: Stillness based on Length Change
        # Reset still count immediately if merges occurred this frame
        if merged_this_frame:
            self.still_frame_count = 0
            self.accumulated_length_change = 0.0
        else:
            # accumulate the absolute changes
            length_change = abs(current_length - length_before_this_frame)
            self.accumulated_length_change += length_change

            if self.accumulated_length_change < LENGTH_STILL_THRESHOLD:
                self.still_frame_count += 1
            else:
                # exceeded threshold → reset both
                self.still_frame_count = 0
                self.accumulated_length_change = 0.0

        # Check 3: Convergence Condition (Have we been still long enough?)
        if self.still_frame_count >= STILL_COUNT_REQUIRED:
            is_non_trivial = len(self.vertices) > 1 and current_length > NON_TRIVIAL_LENGTH_THRESHOLD
            # Update prev_length before stopping (good practice)
            self.prev_total_length = current_length
            if is_non_trivial:
                print(f"Window {self.fig_num} ({self.shape_name}): Converged non-trivial (Len:{current_length:.4f}). Saving & Restarting.")
                self.display_and_save_stable_configuration()
                self._schedule_restart()
                self.freeze_simulation("Converged - Saved")
            else: # Converged to trivial state (Failure)
                print(f"Window {self.fig_num} ({self.shape_name}): Converged TRIVIAL. Restart attempt {self.sim_fail_restart_attempts + 1}/{self.max_sim_fail_restarts}")
                self.sim_fail_restart_attempts += 1
                if self.sim_fail_restart_attempts < self.max_sim_fail_restarts:
                    self._schedule_restart()
                    self.freeze_simulation("Converged - Trivial - Restarting")
                else:
                    self.freeze_simulation("Converged - Trivial - Max Restarts")
            return False # Stop current run

        # --- If we get here, simulation continues ---
        # Update prev_total_length *NOW* so the NEXT frame compares against THIS frame's end length.
        self.prev_total_length = current_length
        return True # Continue simulation

    def _schedule_restart(self):
        # (No changes needed in _schedule_restart)
        try:
            if self.fig and self.fig.canvas and self.fig.canvas.manager and hasattr(self.fig.canvas.manager, 'window'):
                self.fig.canvas.manager.window.after(100, self._start_simulation)
            else: print("Error: Cannot schedule restart."); self.freeze_simulation("Restart Scheduling Error")
        except Exception as e: print(f"Error scheduling restart: {e}"); self.freeze_simulation("Restart Scheduling Exception")

    def freeze_simulation(self, reason="Frozen"):
        # (No changes needed in freeze_simulation)
        if not self.is_running: return
        self.is_running = False
        try:
            if self.anim and hasattr(self.anim, 'event_source') and self.anim.event_source is not None: self.anim.event_source.stop()
            if self.ax:
                current_title = self.ax.get_title()
                if "[" in current_title and "]" in current_title: base_title = current_title[:current_title.find('[')].strip(); self.ax.set_title(f"{base_title} [{reason}]", fontsize=10)
                else: self.ax.set_title(f"Sim Runner {self.fig_num} ({self.shape_name}): ... [{reason}]", fontsize=10)
                self.update_plot() # Final plot update shows frozen state
            if self.fig: self.fig.canvas.draw_idle(); # plt.pause(0.05)
        except Exception as e: print(f"Error during freeze_simulation: {e}")

    def animate(self, frame):
        """Main animation loop step."""
        artists = (self.all_connections_line, self.blue_scatter, self.red_scatter)
        if not self.is_running: return artists
        self.frame_count += 1; merged_this_frame = False; should_continue = True
        try:
            self.update_state()
            if not self.vertices:
                 # Pass merged_this_frame=False as state is already empty
                 should_continue = self.check_convergence_and_validity(False)
                 return self.update_plot()

            merged_this_frame = self.check_and_merge()
            if not self.vertices and merged_this_frame:
                 # Pass merged_this_frame=False as state is already empty
                 should_continue = self.check_convergence_and_validity(False)
                 return self.update_plot()

            # Check convergence based on length change
            # Pass merged_this_frame status to reset counter if needed
            should_continue = self.check_convergence_and_validity(merged_this_frame)

            artists = self.update_plot()
            # If stopped, reset prev_length to avoid immediate convergence on restart?
            # It's already updated inside check_convergence_and_validity if continuing.
            # And reset by _start_simulation if restarting. So no action needed here.

        except Exception as e:
             print(f"!!!! ERROR in Sim {self.fig_num} ({self.shape_name}) animate frame {self.frame_count}: {type(e).__name__} - {e} !!!!")
             import traceback; traceback.print_exc(); self.freeze_simulation("Error in Animate")
             self.prev_total_length = 0.0 # Reset prev length on error
             artists = self.update_plot()
        return artists

    def restart_from_scratch_button(self, event):
        # (No changes needed in restart_from_scratch_button)
        print(f"--- Button Restart requested for Sim Runner {self.fig_num} ({self.shape_name}) ---")
        if self.anim and hasattr(self.anim, 'event_source') and self.anim.event_source is not None:
             try: self.anim.event_source.stop()
             except Exception as e: print(f"Warning: Error stopping previous animation timer: {e}")
        self.anim = None; self.is_running = False; self._start_simulation()

    def _start_simulation(self):
        """Internal method to initialize and start/restart the animation."""
        print(f"Starting/Restarting Sim Runner {self.fig_num} ({self.shape_name})...")
        if self.anim and hasattr(self.anim, 'event_source') and self.anim.event_source is not None:
             try: self.anim.event_source.stop()
             except Exception as e: print(f"Warn: Error stopping prev anim timer: {e}")
        self.anim = None; self.is_running = False; self._setup_figure()
        init_success = False
        for init_attempt in range(MAX_INIT_ATTEMPTS):
             print(f"  Init attempt {init_attempt + 1}/{MAX_INIT_ATTEMPTS}...")
             init_success = self.init_state()
             if init_success: print("  Init successful."); break
             else: print(f"  Init attempt {init_attempt + 1} failed.")
        if not init_success:
            print(f"ERROR: Init failed after {MAX_INIT_ATTEMPTS} attempts."); self.freeze_simulation("Max Init Attempts Reached")
            if self.ax: self.ax.set_title(f"Sim Runner {self.fig_num} ({self.shape_name}) [INIT FAILED]", fontsize=10)
            if self.fig: self.fig.canvas.draw_idle(); return
        # --- Initialization Succeeded ---
        self.sim_fail_restart_attempts = 0
        # <<< Initialize prev_total_length AFTER successful init_state >>>
        self.prev_total_length = self.get_total_length()
        self.accumulated_length_change = 0.0 
        self.still_frame_count = 0; self.frame_count = 0; self.is_running = True; self._clear_plot()
        try:
            self.anim = FuncAnimation(self.fig, self.animate, init_func=self._clear_plot, interval=1, blit=False, save_count=150)
            self.fig.canvas.draw_idle(); print(f"Sim Runner {self.fig_num} ({self.shape_name}) animation started.")
        except Exception as e: print(f"!!!! ERROR Creating FuncAnimation: {e} !!!!"); import traceback; traceback.print_exc(); self.freeze_simulation("Animation Creation Error"); self.is_running = False
    
    def display_and_save_stable_configuration(self):
        # (No changes needed in display_and_save_stable_configuration)
        self.stable_config_count += 1; fig_stable_num_str = f"{self.fig_num}.{self.stable_config_count}"
        new_fig, new_ax = plt.subplots(num=f"Stable Config {fig_stable_num_str} ({self.shape_name})"); plt.subplots_adjust(bottom=0.1)
        poly_patch_stable = plt.Polygon(self.polygon, fill=False, color='grey', lw=2, zorder=0); new_ax.add_patch(poly_patch_stable)
        segments_x, segments_y = [], []; processed_pairs = set(); vertices_map = {id(v): v for v in self.vertices}
        for v1_id, v1 in vertices_map.items():
            pos1 = v1['pos'];
            if not np.all(np.isfinite(pos1)): continue
            for v2_id in v1.get('neighbors', set()):
                pair_key = tuple(sorted((v1_id, v2_id)))
                if v2_id in vertices_map and pair_key not in processed_pairs:
                    v2 = vertices_map[v2_id]; pos2 = v2['pos']
                    if np.all(np.isfinite(pos2)): segments_x.extend([pos1[0], pos2[0], np.nan]); segments_y.extend([pos1[1], pos2[1], np.nan]); processed_pairs.add(pair_key)
        new_ax.plot(segments_x, segments_y, 'k-', lw=1, zorder=1)
        blue_pts_list = [v['pos'] for v in self.vertices if v['color'] == 'blue' and np.all(np.isfinite(v['pos']))]
        red_pts_list = [v['pos'] for v in self.vertices if v['color'] == 'red' and np.all(np.isfinite(v['pos']))]
        if blue_pts_list: blue_arr = np.array(blue_pts_list); new_ax.scatter(blue_arr[:, 0], blue_arr[:, 1], color='blue', s=25, zorder=3, label=f'Blue ({self.n_blue})')
        if red_pts_list: red_arr = np.array(red_pts_list); new_ax.scatter(red_arr[:, 0], red_arr[:, 1], color='red', s=25, zorder=3, label=f'Red ({self.n_red})')
        new_ax.set_aspect('equal')
        x_pad = (self.max_xy[0] - self.min_xy[0]) * 0.1 + 0.1; y_pad = (self.max_xy[1] - self.min_xy[1]) * 0.1 + 0.1
        new_ax.set_xlim(self.min_xy[0] - x_pad, self.max_xy[0] + x_pad); new_ax.set_ylim(self.min_xy[1] - y_pad, self.max_xy[1] + y_pad)
        new_ax.legend()
        total_len = self.get_total_length(); num_edges_plot = len(processed_pairs)
        title = (f"Stable Config {fig_stable_num_str} ({self.shape_name}): B:{self.n_blue}, R:{self.n_red}, Edges:{num_edges_plot}, Len:{total_len:.4f}")
        new_ax.set_title(title, fontsize=10)
        filepath = None

        try:
            target_dir = os.path.join(SAVE_BASE_DIR, self.shape_name)
            os.makedirs(target_dir, exist_ok=True)
            base_filename = f"{total_len:.4f}_{self.n_blue}+{self.n_red}"

            filepath = os.path.join(target_dir, f"{base_filename}.png")
            if os.path.exists(filepath):
                for i in range(10):  # Only try first 10 suffixes
                    suffix = string.ascii_lowercase[i]
                    filepath = os.path.join(target_dir, f"{base_filename}{suffix}.png")
                    if not os.path.exists(filepath):
                        break
                else:
                    print(f"Warn: Could not find unused filename for {base_filename} after 10 tries. Skipping save.")
                    plt.close(new_fig)
                    return  # Skip display and saving

            print(f"  Saving stable config to: {filepath}")
            new_fig.savefig(filepath, dpi=150, bbox_inches='tight')

        except Exception as e:
            print(f"Error saving file to {filepath if filepath else target_dir}: {e}")

        stable_figures.append(new_fig)
        new_fig.show()
        # plt.close(new_fig)

# ----------------------------
# Main execution block
# ----------------------------
if __name__ == "__main__":
    try:
        target_avg_red = 8
        possible_red = [2, 4, 6, 8, 10]
        # selected_shape_name = "Rhombus" # Choose shape here
        selected_shape_name = "Pentagon" # Choose shape here
        # Available: list(DEFAULT_SHAPES.keys())

        if selected_shape_name not in DEFAULT_SHAPES:
            print(f"Error: Shape '{selected_shape_name}' not found. Available: {list(DEFAULT_SHAPES.keys())}"); exit(1)

        shape_name = selected_shape_name
        if shape_name == "L-Shape": print("\n--- WARNING: Running with L-Shape (non-convex) ---\n")

        print(f"----- Running simulation with shape: {shape_name} -----")
        print(f"----- Configuration: -----")
        print(f"      - Blue points (on boundary): {len(DEFAULT_SHAPES[shape_name]) * 2}")
        print(f"      - Red points (inside): Even number >= 2 (target avg: {target_avg_red})")
        print(f"      - Merge Threshold: {MERGE_THRESHOLD}, Damping: {DAMPING}, DT: {DT}")
        # <<< Updated Convergence Info >>>
        print(f"      - Convergence: Total Length change < {LENGTH_STILL_THRESHOLD:.1E} for {STILL_COUNT_REQUIRED} frames")
        print("----- Behavior: -----")
        print("      - Restarts after finding/saving a stable non-trivial configuration.")
        save_dir_display = os.path.join(SAVE_BASE_DIR, shape_name)
        print(f"      - Saves stable configurations to '{save_dir_display}/'")
        print(f"      - Restarts on simulation failure (empty/collapsed/trivial), up to {MAX_RESTARTS} times.")
        print(f"      - Retries initialization on setup failure, up to {MAX_INIT_ATTEMPTS} times.")
        print(f"      - 'Restart Random' button starts a new configuration.")

        sim_instance = Simulation(lambda_red=target_avg_red, fig_num=1, shape_name=shape_name, red_options=possible_red)
        simulations.append(sim_instance)

        print(f"\nStarting Matplotlib event loop... Close simulation window(s) to exit.")
        plt.show()
        print("Matplotlib event loop finished.")

    except Exception as e:
        print(f"\n--- An error occurred during setup or run ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        import traceback
        print("\n--- Traceback ---"); traceback.print_exc(); print("-----------------")