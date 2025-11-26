import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import math
import random
import json
from datetime import datetime
import copy
import sys
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as transforms
from matplotlib.widgets import Button
import threading
import queue

class A2Geometry:
    def __init__(self):
        self.v1_cart, self.v2_cart = np.array([1.0, 0.0]), np.array([-0.5, np.sqrt(3) / 2])
        R = np.array([[ 1, 0,  1], [ 0, 0, -1], [-1, 0,  0]], dtype=int)
        S = np.array([[ 1, 0,  1], [-1, 0,  0], [ 0, 0, -1]], dtype=int)
        rotations = [np.identity(3, dtype=int)]; [rotations.append(R @ rotations[-1]) for _ in range(5)]
        self.isometry_matrices = rotations + [S @ rot for rot in rotations]
    def a2_to_cartesian(self, v):
        v = np.array(v); a, b = v[..., 0], v[..., 1]; x, y = a + 0.5 * b, (np.sqrt(3) / 2) * b
        return np.array([x.item(), y.item()]) if v.ndim == 1 else np.column_stack([x.flatten(), y.flatten()])
    def a2_trunc(self, v): v = np.array(v); return v[:2] if v.ndim==1 else v[:,:2,np.newaxis]
    def generate_visual_templates(self, prototile, allowed_iso_indices=None):
        if allowed_iso_indices is None: allowed_iso_indices = range(12)
        templates = []
        for i in range(12):
            if i not in allowed_iso_indices: templates.append(None); continue
            M = self.isometry_matrices[i]; verts = (M @ prototile.base_vertices.T).T
            occupancies = {tuple(verts[j]): prototile.angles[j] for j in range(len(verts))}
            if i >= 6: verts = verts[[0] + list(range(len(verts) - 1, 0, -1))]
            min_c, max_c = verts.min(axis=0), verts.max(axis=0)
            a2_verts_cart = self.a2_to_cartesian(verts)
            path = Path(a2_verts_cart)
            for px in range(min_c[0] - 1, max_c[0] + 2):
                for py in range(min_c[1] - 1, max_c[1] + 2):
                    pt_a2 = np.array([px, py, -px-py])
                    if tuple(pt_a2) in occupancies: continue
                    pt_cart = self.a2_to_cartesian(pt_a2)
                    if path.contains_point(pt_cart):
                        occupancies[tuple(pt_a2)] = 12
            templates.append({'verts': verts, 'occupancies': occupancies})
        return templates

class Prototile:
    def __init__(self, name, base_vertices, angles, stripes, illustrative_shape_direct=None, illustrative_shape_reflected=None, allowed_iso_indices=None, rotational_symmetry=1, is_achiral=False):
        self.name, self.base_vertices, self.angles, self.stripes = name, np.array(base_vertices, dtype=int), angles, stripes
        self.illustrative_shape_direct = np.array(illustrative_shape_direct, dtype=int) if illustrative_shape_direct is not None else None
        self.illustrative_shape_reflected = np.array(illustrative_shape_reflected, dtype=int) if illustrative_shape_reflected is not None else None
        self.rotational_symmetry = rotational_symmetry
        self.is_achiral = is_achiral
        base_indices = allowed_iso_indices if allowed_iso_indices is not None else list(range(12))
        if self.rotational_symmetry > 1:
            if 6 % self.rotational_symmetry != 0:
                raise ValueError(f"Rotational symmetry must be a divisor of 6, but got {self.rotational_symmetry}")
            canonical_direct = range(0, 6, self.rotational_symmetry)
            canonical_indices = set(canonical_direct)
            if not self.is_achiral:
                canonical_reflected = [6 + i for i in canonical_direct]
                canonical_indices.update(canonical_reflected)
            self.allowed_iso_indices = [i for i in base_indices if i in canonical_indices]
        else:
            self.allowed_iso_indices = base_indices
        self.visual_templates = GEOMETRY.generate_visual_templates(self, range(12))
        self.vertex_labels = ['g' if angle in [3,6,9] else 'c' for angle in self.angles] if self.angles else None

def TurtleTile(**kwargs):
    verts = [[3,-2,-1], [2,0,-2], [0,1,-1],[0,2,-2], [-1,3,-2], [-2,2,0], [-1,0,1], [-2,0,2], [-2,-1,3], [0,-2,2], [1,-4,3], [2,-4,2], [3,-5,2], [4,-4,0]]
    angles = [6,4,9,4,3,4,9,4,3,8,3,8,3,4]
    stripe_defs = [
        {'p1': verts[0], 'p2': verts[10], 'value': 1},
        {'p1': verts[0], 'p2': verts[6], 'value': -1},
        {'p1': verts[2], 'p2': verts[8], 'value': -1},
        {'p1': verts[12], 'p2': verts[4], 'value': -1}
    ]
    triangle = [verts[1], verts[5], verts[9]]
    return Prototile("Turtle", np.array(verts, dtype=int), angles, stripe_defs, 
                      illustrative_shape_direct=triangle, 
                      illustrative_shape_reflected=triangle, 
                      **kwargs)

def HatTile(**kwargs):
    verts = [[3,-2,-1], [4,-2,-2], [4,-1,-3], [2,0,-2], [1,2,-3], [0,2,-2], [-1,3,-2], [-2,2,0], [-1,0,1], [-2,0,2], [-2,-1,3], [0,-2,2], [1,-1,0], [2,-2,0]]
    angles = [6,4,3,8,3,8,3,4,9,4,3,4,9,4]
    stripe_defs = [
        {'p1': verts[0], 'p2': verts[8], 'value': -1},
        {'p1': verts[6], 'p2': verts[12], 'value': -1},
        {'p1': verts[4], 'p2': verts[10], 'value': [1, -1]},
        {'p1': verts[0], 'p2': verts[2], 'value': [1, -1]},
    ]
    triangle = [verts[3], verts[7], verts[11]]
    return Prototile("Hat", np.array(verts, dtype=int), angles, stripe_defs, 
                     illustrative_shape_direct=triangle, 
                     illustrative_shape_reflected=triangle, 
                     **kwargs)

def HexTile(**kwargs):
    verts = [[1,1,-2],[0,2,-2],[-1,2,-1],[-2,2,0],[-2,1,1],[-2,0,2],[-1,-1,2],[0,-2,2],[1,-2,1],[2,-2,0],[2,-1,-1],[2,0,-2]]
    angles = [6,4] * 6
    stripe_defs = [
        {'p1': verts[0], 'p2': verts[2], 'value': [1, -1]},
        {'p1': verts[2], 'p2': verts[4], 'value': [1, -1]},
        {'p1': verts[4], 'p2': verts[6], 'value': [1, -1]},
        {'p1': verts[6], 'p2': verts[8], 'value': [1, -1]},
        {'p1': verts[8], 'p2': verts[10], 'value': [1, -1]},
        {'p1': verts[10], 'p2': verts[0], 'value': [1, -1]},
    ]
    kwargs.setdefault('rotational_symmetry', 6)
    kwargs.setdefault('is_achiral', True)
    return Prototile("Hex", np.array(verts, dtype=int), angles, stripe_defs, **kwargs)

def PropellerTile(**kwargs):
    verts = [[1,0,-1],[2,0,-2],[2,1,-3],[0,2,-2],[-1,1,0],[-2,2,0],[-3,2,1],[-2,0,2],[0,-1,1],[0,-2,2],[1,-3,2],[2,-2,0]]
    angles = [9,4,3,4] * 3
    stripe_defs = [
        {'p1': verts[0], 'p2': verts[6], 'value': -1},
        {'p1': verts[4], 'p2': verts[10], 'value': -1},
        {'p1': verts[8], 'p2': verts[2], 'value': -1},
    ]
    triangle = [verts[3], verts[7], verts[11]]
    kwargs.setdefault('rotational_symmetry', 3)
    return Prototile("Propeller", np.array(verts, dtype=int), angles, stripe_defs, 
                     illustrative_shape_direct=triangle, 
                     illustrative_shape_reflected=triangle, 
                     **kwargs)


class HeeschTiler:
    def __init__(self, config, visualizer=None):
        self.config, self.live_visualizer, self.max_tiles = config, visualizer, config.get('max_tiles', 100)
        self.tiling_mode = config.get('tiling_mode', 'plane_filling')
        self.description = config.get('description', '');
        if self.live_visualizer: self.live_visualizer.tiler = self
        self.rotation_order = config.get('rotation_order', 1)
        self.use_vertex_matching = config.get('use_vertex_matching', True)
        self.use_stripe_matching = config.get('use_stripe_matching', True)
        self.stripe_matching_tiles = {"Turtle", "Propeller"}
        if self.rotation_order > 1:
            if self.rotation_order not in [2, 3, 6]: raise ValueError(f"rotation_order must be 1, 2, 3, or 6, but got {self.rotation_order}")
            if self.rotation_order == 2: self.cluster_rot_steps = [0, 3]
            elif self.rotation_order == 3: self.cluster_rot_steps = [0, 2, 4]
            else: self.cluster_rot_steps = list(range(6))
            self.cluster_rot_matrices = [np.linalg.matrix_power(GEOMETRY.isometry_matrices[1], j) for j in self.cluster_rot_steps]
            self.rotation_composition_table = []
            for j in self.cluster_rot_steps:
                row = []
                for i in range(12):
                    if i < 6:
                        new_iso_idx = (i + j) % 6
                    else:
                        base_rot_k = i - 6
                        new_rot_k = (-j + base_rot_k + 6) % 6
                        new_iso_idx = 6 + new_rot_k
                    row.append(new_iso_idx)
                self.rotation_composition_table.append(row)
        config_prototiles = config.get('tiling_prototiles', {})
        direct_tiles, reflected_tiles = config_prototiles.get('direct', []), config_prototiles.get('reflected', [])
        priority_list, seen_tiles = [], set()
        for name in direct_tiles + reflected_tiles:
            if name not in seen_tiles: priority_list.append(name); seen_tiles.add(name)
        tile_to_indices = defaultdict(set)
        for name in direct_tiles: tile_to_indices[name].update(range(6))
        for name in reflected_tiles: tile_to_indices[name].update(range(6, 12))
        final_prototile_rules = {name: sorted(list(indices)) for name, indices in tile_to_indices.items()}
        initial_placement_names = set(p['tile_name'] for p in config['initial_placements'])
        all_names = seen_tiles | initial_placement_names
        self.prototiles = {}
        for name in all_names:
            tile_factory = globals().get(f"{name}Tile")
            if not tile_factory: continue
            self.prototiles[name] = tile_factory(allowed_iso_indices=final_prototile_rules.get(name))
        self.tiling_prototiles = {name: self.prototiles[name] for name in priority_list}
        self.initial_placements = []; self.start_time = None; self.largest_tiling_placements = None; self.max_tiles_found_so_far = 0
        self.initial_frontier_edges = set()

    def _get_line_canonical_form(self, p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        A, B = -dy, dx
        C = A * p1[0] + B * p1[1]
        common_divisor = math.gcd(math.gcd(A, B), C)
        A //= common_divisor
        B //= common_divisor
        C //= common_divisor
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)

    def _generate_cluster(self, seed_placement):
        if self.rotation_order == 1: return [seed_placement]
        cluster = []
        for i in range(self.rotation_order):
            c_rot_matrix = self.cluster_rot_matrices[i]
            new_translation = (c_rot_matrix @ seed_placement['translation'].T).T
            new_iso_idx = self.rotation_composition_table[i][seed_placement['iso_idx']]
            cluster.append({'tile_name': seed_placement['tile_name'], 'iso_idx': new_iso_idx, 'translation': new_translation})
        return cluster

    def _get_edges_of_placement(self, p):
        verts = self.prototiles[p['tile_name']].visual_templates[p['iso_idx']]['verts'] + p['translation']
        return [(tuple(verts[i-1]), tuple(verts[i])) for i in range(len(verts))]

    def _is_geometrically_valid(self, move, state):
        move_occupancies = defaultdict(int)
        for p in move:
            template = self.prototiles[p['tile_name']].visual_templates[p['iso_idx']]
            for pt_local, val in template['occupancies'].items():
                pt_global = tuple(np.array(pt_local) + p['translation'])
                if state['occupancies'].get(pt_global, 0) + val > 12: return False
                move_occupancies[pt_global] += val
        for pt, val in move_occupancies.items():
            if val > 12: return False
        return True

    def _passes_heuristic(self, move, state):
        if self.use_vertex_matching:
            move_labels = defaultdict(list)
            for p in move:
                prototile = self.prototiles[p['tile_name']]
                if not prototile.vertex_labels: continue
                template, labels = prototile.visual_templates[p['iso_idx']], prototile.vertex_labels
                if p['iso_idx'] >= 6: labels = [labels[0]] + labels[1:][::-1]
                for v_idx, vert_local in enumerate(template['verts']):
                    pt_global = tuple(np.array(vert_local) + p['translation'])
                    if pt_global in state['vertex_labels'] and state['vertex_labels'][pt_global] != labels[v_idx]: return False
                    move_labels[pt_global].append(labels[v_idx])
            for pt, labels in move_labels.items():
                if len(set(labels)) > 1: return False
        
        if self.use_stripe_matching:
            move_global_lines = {}
            for p in move:
                if p['tile_name'] in self.stripe_matching_tiles:
                    prototile, is_reflected = self.prototiles[p['tile_name']], p['iso_idx'] >= 6
                    M, t = GEOMETRY.isometry_matrices[p['iso_idx']], p['translation']
                    for stripe in prototile.stripes:
                        if isinstance(stripe['value'], (list, tuple)): continue
                        p1 = tuple((M @ np.array(stripe['p1']).T).T + t)
                        p2 = tuple((M @ np.array(stripe['p2']).T).T + t)
                        value = -stripe['value'] if is_reflected else stripe['value']
                        line_key = self._get_line_canonical_form(p1, p2)
                        if line_key in state['global_lines'] and state['global_lines'][line_key] != value: return False
                        if line_key in move_global_lines and move_global_lines[line_key] != value: return False
                        move_global_lines[line_key] = value
        return True

    def _apply_move(self, move, state, is_initial=False):
        for p in move:
            prototile, t = self.prototiles[p['tile_name']], p['translation']
            template = prototile.visual_templates[p['iso_idx']]
            for pt_local, val in template['occupancies'].items():
                state['occupancies'][tuple(np.array(pt_local) + t)] += val
            for p1, p2 in self._get_edges_of_placement(p):
                state['edge_registry'][(p1, p2)] += 1; state['edge_registry'][(p2, p1)] -= 1
                if state['edge_registry'].get((p1,p2)) == 0: del state['edge_registry'][(p1,p2)]
                if state['edge_registry'].get((p2,p1)) == 0: del state['edge_registry'][(p2,p1)]
            if self.use_vertex_matching and prototile.vertex_labels:
                labels = prototile.vertex_labels
                if p['iso_idx'] >= 6: labels = [labels[0]] + labels[1:][::-1]
                for v_idx, vert_local in enumerate(template['verts']):
                    pt_global = tuple(np.array(vert_local) + t)
                    if pt_global not in state['vertex_labels']: state['vertex_labels'][pt_global] = labels[v_idx]
            if self.use_stripe_matching and p['tile_name'] in self.stripe_matching_tiles:
                is_reflected, M = p['iso_idx'] >= 6, GEOMETRY.isometry_matrices[p['iso_idx']]
                for stripe in prototile.stripes:
                    if isinstance(stripe['value'], (list, tuple)): continue
                    p1 = tuple((M @ np.array(stripe['p1']).T).T + t)
                    p2 = tuple((M @ np.array(stripe['p2']).T).T + t)
                    value = -stripe['value'] if is_reflected else stripe['value']
                    line_key = self._get_line_canonical_form(p1, p2)
                    if line_key not in state['global_lines']: state['global_lines'][line_key] = value
        if is_initial: state['initial_placements'].append(move[0])
        else: state['placed_tiles'].append(move[0])

    def _get_candidate_moves(self, edge, state):
        geometrically_valid_moves = []
        p1, p2 = np.array(edge['p1']), np.array(edge['p2']); target = p1 - p2
        for name, p_tile in self.tiling_prototiles.items():
            for i in p_tile.allowed_iso_indices:
                template = p_tile.visual_templates[i]
                if template is None: continue
                verts = template['verts']
                for j in range(len(verts)):
                    if np.array_equal(verts[j] - verts[j-1], target):
                        seed = {'tile_name': name, 'iso_idx': i, 'translation': p1 - verts[j]}
                        move = self._generate_cluster(seed)
                        if self._is_geometrically_valid(move, state):
                            geometrically_valid_moves.append(move)
        heuristically_valid_moves = [m for m in geometrically_valid_moves if self._passes_heuristic(m, state)]
        return heuristically_valid_moves, len(geometrically_valid_moves)

    def _update_max_tiling(self, state):
        count = len(state['initial_placements']) + (len(state['placed_tiles']) * self.rotation_order)
        if count > self.max_tiles_found_so_far:
            self.max_tiles_found_so_far = count
            placements = state['initial_placements'][:]
            for seed in state['placed_tiles']:
                placements.extend(self._generate_cluster(seed))
            self.largest_tiling_placements = placements
            if self.live_visualizer:
                self.live_visualizer.request_draw(self, self.largest_tiling_placements, title=f"New Max: {count} tiles")

    def _search(self, state, level=0, history=()):
        indent = "".join(['  ' if h else '│ ' for h in history])
        if self.live_visualizer and self.live_visualizer.quit_flag: return False
        
        while True:
            frontier = [{'p1': p1, 'p2': p2} for (p1, p2), c in state['edge_registry'].items() if c == 1]
            if not frontier:
                self._update_max_tiling(state)
                return True

            choice_points = []
            forced_moves_found = False
            for edge in frontier:
                if state['edge_registry'].get((edge['p1'], edge['p2']), 0) != 1: continue
                candidate_moves, total_geo = self._get_candidate_moves(edge, state)
                if not candidate_moves:
                    self._update_max_tiling(state)
                    return False
                if len(candidate_moves) == 1:
                    self._apply_move(candidate_moves[0], state)
                    forced_moves_found = True
                else:
                    choice_points.append({'edge': edge, 'moves': candidate_moves, 'total_geo': total_geo})
            
            if self.live_visualizer and forced_moves_found:
                placements = state['initial_placements'][:]
                for seed in state['placed_tiles']: placements.extend(self._generate_cluster(seed))
                self.live_visualizer.request_draw(self, placements, "Applied forced moves")
            
            if not forced_moves_found: break
        
        if not choice_points:
            self._update_max_tiling(state)
            return True

        count = len(state['initial_placements']) + (len(state['placed_tiles']) * self.rotation_order)
        if self.tiling_mode == 'plane_filling' and count >= self.max_tiles:
            self._update_max_tiling(state)
            return True
            
        # choice_points.sort(key=lambda cp: cp['edge']['p1'][0]**2 + cp['edge']['p1'][1]**2)
        first_choice = choice_points[0]
        
        for i, move in enumerate(first_choice['moves']):
            is_last = i == len(first_choice['moves']) - 1
            log_msg = f"branch {i+1}/{len(first_choice['moves'])}"
            if self.use_vertex_matching or self.use_stripe_matching: log_msg += f"/{first_choice['total_geo']}"
            print(f"{indent}{'└─' if is_last else '├─'}[{count}] {log_msg}")
            
            next_state = {
                'edge_registry': state['edge_registry'].copy(),
                'occupancies': state['occupancies'].copy(),
                'placed_tiles': state['placed_tiles'][:],
                'initial_placements': state['initial_placements'],
                'vertex_labels': state['vertex_labels'].copy(),
                'global_lines': state['global_lines'].copy()
            }
            
            self._apply_move(move, next_state)

            if self.live_visualizer:
                placements = next_state['initial_placements'][:]
                for seed in next_state['placed_tiles']: placements.extend(self._generate_cluster(seed))
                self.live_visualizer.request_draw(self, placements, f"Choice {i+1}/{len(first_choice['moves'])}")
            
            if self._search(next_state, level + 1, history + (is_last,)):
                return True
        
        return False

    def tile(self):
        print(f"\n--- Tiling with '{','.join(self.tiling_prototiles.keys())}' in '{self.tiling_mode}' mode (Rotation order: {self.rotation_order}) ---")
        state = {'edge_registry': defaultdict(int), 'occupancies': defaultdict(int), 'placed_tiles': [], 
                 'initial_placements': [], 'vertex_labels': {}, 'global_lines': {}}
        self.largest_tiling_placements = None; self.max_tiles_found_so_far = 0;
        initial_placements = [p.copy() for p in self.config['initial_placements']]
        for p in initial_placements: p['translation'] = np.array(p['translation'], dtype=int)
        for placement in initial_placements:
            if not self._is_geometrically_valid([placement], state) or not self._passes_heuristic([placement], state):
                print(f"Initial configuration is invalid (collision or rule violation)."); return False
            self._apply_move([placement], state, is_initial=True)
        self.initial_placements = state['initial_placements']
        if self.live_visualizer: 
            self.live_visualizer.clear_all()
            self.live_visualizer.draw_prototile_templates(self.prototiles.values())
        if self.tiling_mode == 'corona_completion':
            self.initial_frontier_edges = {edge for edge, count in state['edge_registry'].items() if count == 1}
            print(f"Goal: Cover {len(self.initial_frontier_edges)} initial frontier edges.")
        self._update_max_tiling(state)
        if self.live_visualizer: self.live_visualizer.request_draw(self, self.largest_tiling_placements, "Initial State")
        if self.live_visualizer and self.live_visualizer.quit_flag: return False
        print("\n--- Starting Search ---"); self.start_time = time.time()
        success = self._search(state)
        print(f"\n--- Search Complete in {time.time() - self.start_time:.2f}s ---")
        print(f"Result: {'SUCCESS' if success else 'FAILURE'} | Largest tiling: {self.max_tiles_found_so_far} tiles")
        if self.live_visualizer and self.largest_tiling_placements:
            self.live_visualizer.request_draw(self, self.largest_tiling_placements, title="Final State")
        return success

class StaticVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 9))
        self.fig.canvas.manager.set_window_title("Heesch Tiler")
        
        ## FIX: Use a simple GridSpec with a more aggressive ratio and a bottom margin for buttons.
        # This gives the main plot the majority of the space.
        gs = GridSpec(1, 2, width_ratios=[5, 1], bottom=0.1, top=0.95, left=0.05, right=0.98)

        self.ax_main = self.fig.add_subplot(gs[0])
        self.ax_templates_container = self.fig.add_subplot(gs[1])
        
        self.tiler = None
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('resize_event', self.on_resize)
        self.quit_flag = False
        self.template_axes = []
        self.show_stripes = True
        self.show_shading = True
        self.latest_placements = None
        self.latest_title = ""
        self.shrink_factor = 0.80
        self.corner_rounding_fraction = 0.3
        self.drawn_placements_keys = set()
        self.drawn_artists = {} 

        # The buttons are placed relative to the figure, in the space we reserved at the bottom.
        ax_stripes = plt.axes([0.01, 0.01, 0.1, 0.04]); self.button_stripes = Button(ax_stripes, 'Toggle Stripes'); self.button_stripes.on_clicked(self.toggle_stripes)
        ax_shading = plt.axes([0.12, 0.01, 0.1, 0.04]); self.button_shading = Button(ax_shading, 'Toggle Shading'); self.button_shading.on_clicked(self.toggle_shading)
        ax_quit = plt.axes([0.23, 0.01, 0.1, 0.04]); self.button_quit = Button(ax_quit, 'Quit'); self.button_quit.on_clicked(self.quit_app)

        self.timer = self.fig.canvas.new_timer(interval=100); self.timer.add_callback(self._throttled_draw); self.timer.start()
        self.geometry_cache = {}; self.translation_cache = {}; plt.ion()

    def on_resize(self, event):
        self._redraw_main_canvas(force=True)

    def clear_all(self):
        self.ax_main.clear(); self.ax_main.set_xticks([]); self.ax_main.set_yticks([])
        for ax in self.template_axes: ax.remove()
        self.template_axes = []; self.ax_templates_container.set_axis_off()
        self.translation_cache = {}; self.latest_placements = None
        self.drawn_placements_keys = set()
        self.drawn_artists = {}

    def _get_placement_key(self, p):
        return (p['tile_name'], p['iso_idx'], tuple(p['translation']))

    def _create_stripe_artists(self, p1, p2, value):
        stripe_color = 'green'; linewidth_map = {1: 1.5, -1: 0.5}
        artists = []
        if isinstance(value, (list, tuple)):
            lw1, lw2 = linewidth_map.get(value[0], 1.5), linewidth_map.get(value[1], 1.5)
            midpoint = (p1 + p2) / 2.0
            artists.append(plt.Line2D([p1[0], midpoint[0]], [p1[1], midpoint[1]], color=stripe_color, linewidth=lw1, zorder=1.2))
            artists.append(plt.Line2D([midpoint[0], p2[0]], [midpoint[1], p2[1]], color=stripe_color, linewidth=lw2, zorder=1.2))
        else: artists.append(plt.Line2D([p1[0], p2[0]], [p1[1], p2[1]], color=stripe_color, linewidth=linewidth_map.get(value, 2.5), zorder=1.2))
        return artists

    def _create_rounded_triangle_path(self, vertices, fraction):
        p0, p1, p2 = vertices
        p01, p10 = p0 + fraction * (p1 - p0), p1 + fraction * (p0 - p1)
        p12, p21 = p1 + fraction * (p2 - p1), p2 + fraction * (p1 - p2)
        p20, p02 = p2 + fraction * (p0 - p2), p0 + fraction * (p2 - p0)
        path_verts = [p01, p10, p1, p12, p21, p2, p20, p02, p0, p01]
        path_codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.LINETO, Path.LINETO, Path.CURVE3, Path.LINETO, Path.LINETO, Path.CURVE3, Path.LINETO]
        return Path(path_verts, path_codes)

    def _precompute_geometries(self, prototiles):
        print("Pre-computing geometries...")
        self.geometry_cache = {}
        color_direct = '#87CEEB'; color_reflected = '#FFA07A'
        for prototile in prototiles:
            for i in range(12):
                is_reflected, M = i >= 6, GEOMETRY.isometry_matrices[i]
                verts_cart = GEOMETRY.a2_to_cartesian((M @ prototile.base_vertices.T).T)
                shape_path = None
                if prototile.illustrative_shape_direct is not None:
                    local_shape = prototile.illustrative_shape_reflected if is_reflected else prototile.illustrative_shape_direct
                    shape_cart = GEOMETRY.a2_to_cartesian((M @ local_shape.T).T)
                    if self.shrink_factor < 1.0:
                        centroid = np.mean(shape_cart, axis=0)
                        shape_cart = centroid + self.shrink_factor * (shape_cart - centroid)
                    shape_path = self._create_rounded_triangle_path(shape_cart, self.corner_rounding_fraction)
                stripe_artists = []
                if prototile.stripes:
                    for stripe_def in prototile.stripes:
                        p1, p2 = GEOMETRY.a2_to_cartesian((M @ np.array(stripe_def['p1']).T).T), GEOMETRY.a2_to_cartesian((M @ np.array(stripe_def['p2']).T).T)
                        value = stripe_def['value']
                        final_val = [-v for v in value] if (is_reflected and isinstance(value, (list, tuple))) else (-value if is_reflected else value)
                        stripe_artists.extend(self._create_stripe_artists(p1, p2, final_val))
                self.geometry_cache[(prototile.name, i)] = {'verts_cart': verts_cart, 'face_color': color_reflected if is_reflected else color_direct,
                    'illustrative_shape_path': shape_path, 'illustrative_shape_parity_key': (is_reflected, i % 2), 'stripe_artists': stripe_artists}
        print("Pre-computation finished.")

    def draw_prototile_templates(self, prototiles):
        if not self.geometry_cache: self._precompute_geometries(prototiles)
        for ax in self.template_axes: ax.remove()
        self.template_axes.clear(); self.ax_templates_container.set_axis_off()
        sorted_prototiles = sorted(prototiles, key=lambda p: p.name)
        n = len(sorted_prototiles)
        if n == 0: return
        template_face_color, template_tri_color = '#87CEEB', '#FFFFFF'
        
        # This nested GridSpec correctly places the templates within their container subplot.
        gs_nested = self.ax_templates_container.get_subplotspec().subgridspec(n + 2, 1, hspace=0, height_ratios=[1] + [10] * n + [1])
        
        for i, prototile in enumerate(sorted_prototiles):
            ax = self.fig.add_subplot(gs_nested[i + 1]); self.template_axes.append(ax)
            ax.set_aspect('equal', adjustable='box'); ax.set_xticks([]); ax.set_yticks([])
            cached_geom = self.geometry_cache[(prototile.name, 0)]; verts_cart = cached_geom['verts_cart']
            min_x, max_x, min_y, max_y = verts_cart[:, 0].min(), verts_cart[:, 0].max(), verts_cart[:, 1].min(), verts_cart[:, 1].max()
            cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2; half_size = max(max_x - min_x, max_y - min_y) * 0.5 + 1.0
            ax.set_xlim(cx - half_size, cx + half_size); ax.set_ylim(cy - half_size, cy + half_size)
            ax.fill(verts_cart[:, 0], verts_cart[:, 1], fc=template_face_color, ec='black', lw=1, zorder=1)
            if cached_geom['illustrative_shape_path'] is not None:
                ax.add_patch(PathPatch(cached_geom['illustrative_shape_path'], fc=template_tri_color, alpha=0.8, zorder=1.1, ec='none'))
            if cached_geom['stripe_artists']:
                for artist in cached_geom['stripe_artists']: ax.add_line(copy.copy(artist))
            for j, v_cart in enumerate(verts_cart):
                v_a2 = prototile.base_vertices[j]
                ax.text(v_cart[0], v_cart[1], f"({v_a2[0]},{v_a2[1]})", fontsize=6, ha='center', va='center', color='black', clip_on=True, zorder=3)

    def _get_tile_geometry(self, p): return self.geometry_cache.get((p['tile_name'], p['iso_idx']))
    def _get_cartesian_translation(self, a2_translation_vec):
        key = tuple(a2_translation_vec)
        if key not in self.translation_cache: self.translation_cache[key] = GEOMETRY.a2_to_cartesian(a2_translation_vec)
        return self.translation_cache[key]

    def toggle_stripes(self, event): self.show_stripes = not self.show_stripes; print(f"Stripes toggled {'ON' if self.show_stripes else 'OFF'}"); self._redraw_main_canvas(force=True)
    def toggle_shading(self, event): self.show_shading = not self.show_shading; print(f"Shading toggled {'ON' if self.show_shading else 'OFF'}"); self._redraw_main_canvas(force=True)
    def quit_app(self, event): print("Quit button pressed. Stopping search and closing."); self.quit_flag = True; plt.close(self.fig)
    def request_draw(self, tiler, placements, title="Tiling"): self.tiler = tiler; self.latest_placements = placements; self.latest_title = title
    def _throttled_draw(self): self._redraw_main_canvas()

    def _redraw_main_canvas(self, force=False):
        if self.quit_flag or self.latest_placements is None: return
        
        if force:
            self.ax_main.clear()
            self.ax_main.set_aspect('equal', adjustable='box'); self.ax_main.set_xticks([]); self.ax_main.set_yticks([])
            self.drawn_artists = {}
            self.drawn_placements_keys = set()
        
        placements = self.latest_placements
        if not placements:
            self.fig.canvas.draw_idle(); return
        
        new_placements_keys = {self._get_placement_key(p) for p in placements}
        
        keys_to_remove = self.drawn_placements_keys - new_placements_keys
        if keys_to_remove:
            for key in keys_to_remove:
                if key in self.drawn_artists:
                    for artist in self.drawn_artists[key]:
                        artist.remove()
                    del self.drawn_artists[key]
        
        keys_to_add = new_placements_keys - self.drawn_placements_keys
        if keys_to_add:
            placements_to_add = [p for p in placements if self._get_placement_key(p) in keys_to_add]
            tri_colors = {(False, 0): '#FFFFFF', (False, 1): '#4682B4', (True, 0):  '#FFF5EE', (True, 1):  '#E9967A'}
            for p in placements_to_add:
                geom = self._get_tile_geometry(p)
                if not geom: continue
                
                placement_key = self._get_placement_key(p)
                t_cart = self._get_cartesian_translation(p['translation'])
                verts_world = geom['verts_cart'] + t_cart
                
                artists = []
                poly = plt.Polygon(verts_world, fc=geom['face_color'], ec='black', lw=1.0, alpha=0.8, zorder=1)
                self.ax_main.add_patch(poly); artists.append(poly)
                
                if self.show_shading and geom['illustrative_shape_path'] is not None:
                    patch = PathPatch(geom['illustrative_shape_path'], fc=tri_colors.get(geom['illustrative_shape_parity_key'], 'gray'), alpha=0.7, zorder=1.1, ec='none')
                    patch.set_transform(transforms.Affine2D().translate(t_cart[0], t_cart[1]) + self.ax_main.transData)
                    self.ax_main.add_patch(patch); artists.append(patch)
                
                if self.show_stripes and geom['stripe_artists']:
                    for artist in geom['stripe_artists']:
                        new_artist = copy.copy(artist)
                        local_x, local_y = artist.get_xdata(), artist.get_ydata()
                        new_artist.set_data(local_x + t_cart[0], local_y + t_cart[1])
                        self.ax_main.add_line(new_artist); artists.append(new_artist)

                self.drawn_artists[placement_key] = artists

        self.drawn_placements_keys = new_placements_keys
        self.ax_main.set_title(self.latest_title)
        
        all_verts = np.vstack([self.geometry_cache[(p['tile_name'], p['iso_idx'])]['verts_cart'] + self._get_cartesian_translation(p['translation']) for p in placements])
        min_x, max_x = all_verts[:, 0].min(), all_verts[:, 0].max()
        min_y, max_y = all_verts[:, 1].min(), all_verts[:, 1].max()

        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        half_size = max(max_x - min_x, max_y - min_y) * 0.6 + 2
        
        if force:
             self.ax_main.set_xlim(cx - half_size, cx + half_size)
             self.ax_main.set_ylim(cy - half_size, cy + half_size)
        else:
            current_xlim = self.ax_main.get_xlim()
            current_ylim = self.ax_main.get_ylim()
            if min_x < current_xlim[0] or max_x > current_xlim[1] or min_y < current_ylim[0] or max_y > current_ylim[1]:
                 self.ax_main.set_xlim(cx - half_size, cx + half_size)
                 self.ax_main.set_ylim(cy - half_size, cy + half_size)
        
        ## FIX: Remove the now-unnecessary tight_layout call.
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes != self.ax_main: return
        scale = 1/1.2 if event.button == 'up' else 1.2
        cur_xlim, cur_ylim = self.ax_main.get_xlim(), self.ax_main.get_ylim()
        x, y = event.xdata, event.ydata
        if x is None or y is None: return
        new_w, new_h = (cur_xlim[1] - cur_xlim[0]) * scale, (cur_ylim[1] - cur_ylim[0]) * scale
        rel_x, rel_y = (cur_xlim[1] - x) / (cur_xlim[1] - cur_xlim[0]), (cur_ylim[1] - y) / (cur_ylim[1] - cur_ylim[0])
        self.ax_main.set_xlim([x - new_w * (1 - rel_x), x + new_w * rel_x])
        self.ax_main.set_ylim([y - new_h * (1 - rel_y), y + new_h * rel_y])
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    GEOMETRY = A2Geometry()
    
    n = 16
    config = {
        'initial_placements': [
            # {'tile_name': 'Turtle', 'iso_idx': 6, 'translation': np.array([0,0,0])},
            {'tile_name':'Turtle', 'iso_idx':9, 'translation':np.array([n, 0, -n])},
            {'tile_name':'Turtle', 'iso_idx':7, 'translation':np.array([-n, n, 0])},
            {'tile_name':'Turtle', 'iso_idx':11, 'translation':np.array([0, -n, n])},
        ],
        'tiling_prototiles': {
            'reflected': ['Turtle'],
            'direct': ['Turtle'],
        },
        'description': 'Test G/C decorations based on angle',
        'max_tiles': 10000,
        'tiling_mode': 'plane_filling',
        'rotation_order': 1,
        'use_vertex_matching': True,
        'use_stripe_matching': True # Set to False to disable the long-range stripe heuristic
    }

    vis = StaticVisualizer()
    tiler = HeeschTiler(config=config, visualizer=vis)
    
    tiler_thread = threading.Thread(target=tiler.tile)
    tiler_thread.daemon = True
    tiler_thread.start()
    
    print("\n--- UI is active. Tiling is running in the background. ---")
    plt.show(block=True)
    
    print("\n--- Full process finished. ---")
