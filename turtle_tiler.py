import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.lines as mlines
from collections import Counter, defaultdict
import time
import math
import random

class A2Geometry:
    def __init__(self):
        self.v1_cart, self.v2_cart = np.array([1.0, 0.0]), np.array([-0.5, np.sqrt(3) / 2])
        R = np.array([[ 1, 0,  1], [ 0, 0, -1], [-1, 0,  0]], dtype=int)
        S = np.array([[ 1, 0,  1], [-1, 0,  0], [ 0, 0, -1]], dtype=int)
        rotations = [np.identity(3, dtype=int)]; [rotations.append(R @ rotations[-1]) for _ in range(5)]
        self.isometry_matrices = rotations + [S @ rot for rot in rotations]

    def a2_to_cartesian(self, v):
        v = np.array(v)
        return (v[0]*self.v1_cart - v[2]*self.v2_cart) if v.ndim==1 else (v[:,0,np.newaxis]*self.v1_cart - v[:,2,np.newaxis]*self.v2_cart)

    def a2_trunc(self, v):
        v = np.array(v)
        return v[:2] if v.ndim==1 else v[:,:2,np.newaxis]

    def _point_in_polygon(self, p, poly):
        x, y = p
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if ((p1y <= y and p2y > y) or (p1y > y and p2y <= y)):
                val = (x - p1x) * (p2y - p1y) - (y - p1y) * (p2x - p1x)
                if p2y > p1y:  # Edge goes upwards relative to p1
                    if val < 0: # Point is to the left of the edge
                        inside = not inside
                else: # p1y > p2y (Edge goes downwards relative to p1)
                    if val > 0: # Point is to the left of the edge (relative to an upward ray)
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def get_line_parameters(self, p1, p2):
        d = p2 - p1; n = np.array([d[2]-d[1], d[0]-d[2], d[1]-d[0]])
        gcd = math.gcd(math.gcd(n[0], n[1]), n[2])
        if gcd==0: return None, None
        un = tuple(n // gcd)
        for x in un:
            if x!=0:
                if x<0: un = tuple(-y for y in un)
                break
        return (un, np.dot(un, p1))

    def generate_visual_templates(self, prototile, allowed_iso_indices=None):
        if allowed_iso_indices is None:
            allowed_iso_indices = range(12) # Default to all 12 isometries
        templates = []
        for i in range(12):
            if i not in allowed_iso_indices:
                templates.append(None) # Placeholder for disallowed isometries
                continue
            M = self.isometry_matrices[i]
            verts = (M @ prototile.base_vertices.T).T
            occupancies = dict()
            for j in range(len(verts)): occupancies[tuple(verts[j])] = prototile.angles[j]
            if i >= 6: verts = verts[[0] + list(range(len(verts) - 1, 0, -1))]
            min_c, max_c = verts.min(axis=0), verts.max(axis=0)
            for px in range(min_c[0], max_c[0] + 1):
                for py in range(min_c[1], max_c[1] + 1):
                    pt = np.array([px, py, -px-py])
                    if tuple(pt) in occupancies: 
                        continue
                    elif self._point_in_polygon(self.a2_trunc(pt), self.a2_trunc(verts)): 
                        occupancies[tuple(pt)] = 12
            print({'verts': verts, 'occupancies': occupancies})
            templates.append({'verts': verts, 'occupancies': occupancies})
        return templates

    def generate_dual_information(self, prototile, allowed_iso_indices=None):
        if allowed_iso_indices is None:
            allowed_iso_indices = range(12)
        dual_tiles = []
        for i in range(12):
            if i not in allowed_iso_indices:
                dual_tiles.append(None) # Placeholder for disallowed isometries
                continue
            is_reflected = i >= 6; tile_stripes = []
            for stripe in prototile.stripes:
                p1, p2, color = self.isometry_matrices[i] @ stripe['p1'], self.isometry_matrices[i] @ stripe['p2'], stripe['color']
                params = self.get_line_parameters(p1, p2)
                if params[0] is not None:
                    actual_color = ('red' if color == 'blue' else 'blue') if is_reflected else color
                    tile_stripes.append({'params': params, 'color': actual_color})
            dual_tiles.append(tile_stripes)
        return dual_tiles

class Prototile:
    def __init__(self, name, base_vertices, angles, stripes, matching_rule_type, allowed_iso_indices=None):
        self.name = name
        self.base_vertices = np.array(base_vertices, dtype=int) 
        self.angles = angles
        self.stripes = stripes
        self.matching_rule_type = matching_rule_type
        self.allowed_iso_indices = allowed_iso_indices if allowed_iso_indices is not None else list(range(12))
        self.visual_templates = GEOMETRY.generate_visual_templates(self, self.allowed_iso_indices)
        self.dual_info = GEOMETRY.generate_dual_information(self, self.allowed_iso_indices)

    def is_valid_heuristic(self, placement, occupied_stripes, enable_stripe_heuristic):
        if enable_stripe_heuristic and self.matching_rule_type == 'long_range_stripes':
            stripes_to_check = self.dual_info[placement['iso_idx']]
            t = placement['translation']
            for s in stripes_to_check:
                n,d0,c = s['params'][0], s['params'][1], s['color']
                d = d0 + np.dot(n,t)
                if (n, d, 'red' if c == 'blue' else 'blue') in occupied_stripes:
                    return False
        return True

def TurtleTile(allowed_iso_indices=None):
    verts = [[0,-3,3],[1,-3,2],[2,-4,2],[3,-3,0],[2,-1,-1],[1,1,-2],[-1,2,-1],[-1,3,-2],[-2,4,-2],[-3,3,0],[-2,1,1],[-3,1,2],[-3,0,3],[-1,-1,2]]
    angles = [3,8,3,4,6,4,9,4,3,4,9,4,3,8]
    shift = np.array([-1, -1, 2], dtype=int)
    stripe_defs = [
        {'p1': [-1,-4,5], 'p2': [1,-2,1], 'color': 'blue'},
        {'p1': [1,-2,1], 'p2': [-1,2,-1], 'color': 'red'},
        {'p1': [2,-1,-1], 'p2': [-2,1,1], 'color': 'red'},
        {'p1': [-2,1,1], 'p2': [-4,-1,5], 'color': 'red'},
    ]
    projections = {
        0: {-1: -1, 0: 0, 1: 1},
        1: {-1: 0, 0: -1},
        2: {-1: 0, 0: -1, 1: 0},
        3: {-1: 1, 0: 0, 1: -1},
        4: {0: -1, 1: 0},
        5: {-1: 0, 0: -1, 1: 0},
    }
    return Prototile("Turtle", verts, angles, stripe_defs, matching_rule_type='long_range_stripes', allowed_iso_indices=allowed_iso_indices)

def PropellerTile(allowed_iso_indices=[0,1,6,7]):
    verts = [[1,-1,0],[2,0,-2],[1,2,-3],[0,2,-2],[0,1,-1],[-2,2,0],[-3,1,2],[-2,0,2],[-1,0,1],[0,-2,2],[2,-3,1],[2,-2,0]]
    angles = [9,4,3,4] * 3
    stripe_defs = [
        {'p1': verts[0], 'p2': verts[6], 'color': 'blue'},
        {'p1': verts[4], 'p2': verts[10], 'color': 'blue'},
        {'p1': verts[8], 'p2': verts[2], 'color': 'blue'},
    ]
    projections = {
        0: {0: 1},
        1: {0: 1},
        2: {0: 1},
        3: {0: 1},
        4: {0: 1},
        5: {0: 1},
    }
    return Prototile("Propeller", verts, angles, stripe_defs, matching_rule_type='long_range_stripes', allowed_iso_indices=allowed_iso_indices)

class HeeschTiler:
    def __init__(self, config, visualizer=None, enable_stripe_heuristic=True):
        self.config = config
        self.live_visualizer = visualizer
        self.max_tiles, self.tiling_mode = config.get('max_tiles', 50), config.get('tiling_mode', 'plane_filling')
        self.enable_stripe_heuristic = enable_stripe_heuristic
        
        # Extract prototiles from config, combining initial_placements and tiling_prototiles
        all_tile_names_in_config = set(p_data['tile_name'] for p_data in config['initial_placements'])
        all_tile_names_in_config.update(config['tiling_prototiles'])
        
        self.prototiles = {}
        for tile_name in all_tile_names_in_config:
            allowed_iso_indices = None
            if 'allowed_iso_indices' in config.get('tile_properties', {}) and tile_name in config['tile_properties']:
                allowed_iso_indices = config['tile_properties'][tile_name].get('allowed_iso_indices')
            
            if tile_name == "Turtle":
                self.prototiles[tile_name] = TurtleTile()
            elif tile_name == "Propeller":
                self.prototiles[tile_name] = PropellerTile()

        self.tiling_prototiles = {name: self.prototiles[name] for name in config['tiling_prototiles']} # Prototiles used for tiling

        self.placed_tiles, self.initial_placements = [], []
        self.occupancies, self.occupied_stripes = {}, {}
        self.start_time = None
        self.edge_registry = Counter()
        self.choice_indent_level = 0

    def _get_edges_of_placement(self, p):
        verts = self.prototiles[p['tile_name']].visual_templates[p['iso_idx']]['verts'] + p['translation']
        return [(tuple(verts[i-1]), tuple(verts[i])) for i in range(len(verts))]

    def _apply_placement_to_state(self, placement, is_initial=False):
        name, i, t = placement['tile_name'], placement['iso_idx'], placement['translation']
        prototile = self.prototiles[name] 
        tile_data = prototile.visual_templates[i]
        occupancies = {}
        for pt in tile_data['occupancies']: 
            val = tile_data['occupancies'][pt]
            occupancies[tuple(pt+t)] = val
            self.occupancies[tuple(pt+t)] = self.occupancies.get(tuple(pt+t), 0) + val
            if self.occupancies[tuple(pt+t)] > 12:
                print(tuple(pt+t), self.occupancies.get(tuple(pt+t), 0), '+', val, '> 12')
        newly_added_stripes = set()
        if self.enable_stripe_heuristic and prototile.matching_rule_type == 'long_range_stripes':
            for s in prototile.dual_info[i]:
                n, d0, c = s['params'][0], s['params'][1], s['color']; d = d0 + np.dot(n, t); sid = (n, d, c)
                self.occupied_stripes[sid] = self.occupied_stripes.get(sid, 0) + 1
                newly_added_stripes.add(sid)
        for edge in self._get_edges_of_placement(placement):
            p1, p2 = edge
            self.edge_registry[(p1, p2)] += 1
            self.edge_registry[(p2, p1)] -= 1
            # if self.edge_registry[(p1, p2)] == 0: del self.edge_registry[(p1, p2)]
            # if self.edge_registry[(p2, p1)] == 0: del self.edge_registry[(p2, p1)]
        if is_initial: self.initial_placements.append(placement)
        else: self.placed_tiles.append(placement)
        return {'occupancies': occupancies, 'stripes': newly_added_stripes, 'is_initial': is_initial, 'placement_ref': placement}

    def _undo_placement(self, undo_info):
        placement_ref = undo_info['placement_ref']
        target_list = self.initial_placements if undo_info['is_initial'] else self.placed_tiles
        idx_to_remove = next((i for i, p in enumerate(target_list) if p['tile_name'] == placement_ref['tile_name'] and p['iso_idx'] == placement_ref['iso_idx'] and np.array_equal(p['translation'], placement_ref['translation'])), -1)
        if idx_to_remove != -1: target_list.pop(idx_to_remove)
        for p in undo_info['occupancies']:
            self.occupancies[p] -= undo_info['occupancies'].get(p, 0)
            if self.occupancies[p] == 0: del self.occupancies[p]
        if 'stripes' in undo_info and self.enable_stripe_heuristic:
            for sid in undo_info['stripes']:
                self.occupied_stripes[sid] -= 1
                if self.occupied_stripes[sid] == 0: del self.occupied_stripes[sid]
        for edge in self._get_edges_of_placement(placement_ref):
            p1, p2 = edge
            self.edge_registry[(p1, p2)] -= 1
            self.edge_registry[(p2, p1)] += 1
            if self.edge_registry[(p1, p2)] == 0: del self.edge_registry[(p1, p2)]
            if self.edge_registry[(p2, p1)] == 0: del self.edge_registry[(p2, p1)]

    def _get_candidate_placements(self, edge):
        cands, p1, p2 = [], np.array(edge['p1']), np.array(edge['p2']); target = p1 - p2
        for name, prototile in self.tiling_prototiles.items():
            for i in prototile.allowed_iso_indices: # Iterate only over allowed iso_indices
                template = prototile.visual_templates[i]
                if template is None: continue # Skip if isometry is disallowed
                verts = template['verts']
                for j in range(len(verts)):
                    if np.array_equal(verts[j] - verts[j-1], target):
                        cands.append({'tile_name': name, 'iso_idx': i, 'translation': p1 - verts[j]})
        return cands

    def _is_placement_valid(self, p):
        name, i, t = p['tile_name'], p['iso_idx'], p['translation']
        prototile = self.prototiles[name]
        tile = prototile.visual_templates[i]
        for pt in tile['occupancies']:
            if self.occupancies.get(tuple(pt+t), 0) + tile['occupancies'][tuple(pt)] > 12:
                return False
        return prototile.is_valid_heuristic(p, self.occupied_stripes, self.enable_stripe_heuristic)

    def _search(self, last_choice_edge, level=0, history_is_last_choice=()):
        indent_parts = []
        for i in range(level):
            if i < len(history_is_last_choice) and history_is_last_choice[i]:
                indent_parts.append(' ')
            else:
                indent_parts.append('│')
        indent_str = "".join(indent_parts)
        undo_stack_for_this_level = []
        try:
            all_forced_moves = []
            while True:
                if len(self.placed_tiles) >= self.max_tiles: return True
                all_frontier_edges = [{'p1': p1, 'p2': p2} for (p1, p2), c in self.edge_registry.items() if c == 1]
                if not all_frontier_edges: return True
                forced_placements_map, choice_edges = {}, []
                for edge in all_frontier_edges:
                    placements = [tile for tile in self._get_candidate_placements(edge) if self._is_placement_valid(tile)]
                    if len(placements) == 0:
                        all_forced_moves.append(0)
                        if self.live_visualizer: self.live_visualizer.draw(self, "Dead End on Frontier", forced_move_history=all_forced_moves)
                        for u in reversed(undo_stack_for_this_level): self._undo_placement(u)
                        return False
                    elif len(placements) == 1:
                        tile = placements[0]
                        key = (tile['tile_name'], tile['iso_idx'], tuple(tile['translation']))
                        if key not in forced_placements_map: forced_placements_map[key] = tile
                    else:
                        choice_edges.append({'edge': edge, 'placements': placements})

                if forced_placements_map:
                    forced_moves = 0
                    for p in forced_placements_map.values():
                        if not self._is_placement_valid(p):
                            all_forced_moves.append(forced_moves)
                            if self.live_visualizer: self.live_visualizer.draw(self, "Dead End on Frontier", forced_move_history=all_forced_moves)
                            for u in reversed(undo_stack_for_this_level): self._undo_placement(u)
                            return False
                        undo_stack_for_this_level.append(self._apply_placement_to_state(p))
                        forced_moves += 1
                    all_forced_moves.append(forced_moves)
                    if self.live_visualizer:
                        self.live_visualizer.draw(self, f"Applying Forced Moves", forced_move_history=all_forced_moves)
                else:
                    break

            if not choice_edges: return True
            current_choice_info = choice_edges[0]
            current_choice_edge = current_choice_info['edge']
            num_choices = len(current_choice_info['placements'])
            N = len(self.placed_tiles) + len(self.initial_placements)

            for i, placement in enumerate(current_choice_info['placements']):
                is_current_choice_last = (i == num_choices - 1)
                if is_current_choice_last:
                    print(f"{indent_str}└[{N}] branch {i+1}/{num_choices}")
                else:
                    print(f"{indent_str}├[{N}] branch {i+1}/{num_choices}")
                undo_info = self._apply_placement_to_state(placement)
                if self.live_visualizer:
                    self.live_visualizer.draw(self, f"Applied choice {i+1}/{num_choices}", forced_move_history=all_forced_moves)
                next_history = history_is_last_choice + (is_current_choice_last,)
                if self._search(last_choice_edge=current_choice_edge, level=level + 1, history_is_last_choice=next_history):
                    return True
                self._undo_placement(undo_info)

            for u in reversed(undo_stack_for_this_level):
                self._undo_placement(u)
            return False
        finally:
            pass

    def tile(self):
        print(f"\n--- Tiling with '{','.join(self.tiling_prototiles.keys())}' on A2 Geometry (Mode: {self.tiling_mode}) ---")
        
        # Process initial placements with validity checks
        for i, p_data in enumerate(self.config['initial_placements']):
            print(f"Attempting initial placement #{i+1}: {p_data['tile_name']} (iso_idx: {p_data['iso_idx']}, translation: {p_data['translation']})")
            if not self._is_placement_valid(p_data):
                print(f"FATAL: Initial placement #{i+1} is invalid (occupancy or stripe conflict)."); return
            self._apply_placement_to_state(p_data, is_initial=True)
            if self.live_visualizer:
                self.live_visualizer.draw(self, f"Initial Placement #{i+1} Applied", forced_move_history=[])
        
        self.global_center_emb = np.mean([p['translation'] for p in self.initial_placements], axis=0) if self.initial_placements else np.array([0,0,0])
        if self.live_visualizer: self.live_visualizer.draw(self, "Initial State Ready", forced_move_history=[])
        
        print("\n--- Starting Search ---"); self.start_time = time.time()
        success = self._search(last_choice_edge=None, level=0)
        print(f"\n--- Search Complete in {time.time() - self.start_time:.2f}s ---")
        print(f"Result: {'SUCCESS' if success else 'FAILURE'} | Total Tiles: {len(self.placed_tiles) + len(self.initial_placements)}")
        if self.live_visualizer:
            self.live_visualizer.draw(self, "Final Tiling - SUCCESS" if success else "Final State - FAILURE", forced_move_history=[])
            plt.ioff(); plt.show()


class StaticVisualizer:
    def __init__(self, show_coronas=True, initial_enable_stripe_heuristic=True):
        self.fig = plt.figure(figsize=(18, 10))
        gs = self.fig.add_gridspec(7, 8)
        self.ax_main = self.fig.add_subplot(gs[:, 1:-1])
        self.ax_main.set_aspect('equal', adjustable='box')
        self.ax_directs = [self.fig.add_subplot(gs[i, 0]) for i in range(6)]
        self.ax_reflects = [self.fig.add_subplot(gs[i, -1]) for i in range(6)]
        self.ax_tally_direct = self.fig.add_subplot(gs[6, 0])
        self.ax_tally_reflected = self.fig.add_subplot(gs[6, -1])
        plt.ion(); self.fig.canvas.manager.set_window_title("Heesch Tiler")
        
        self.tiler = None; self.color_map = {0:'magenta', 1:'lime', 2:'yellow'}
        self.drawn_placed_artists = {}; self.drawn_boundary_artists = {}
        self.temporary_artists = []; self.palette_xlim, self.palette_ylim = None, None
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.dominant_color = None; self.decision_tile_count = 10; self.majority_threshold = 5
        self.last_dominant_color = None
        self.max_tiles_reached = 0 

    def on_scroll(self, event):
        if event.inaxes != self.ax_main: return
        scale_factor = 1.2; cur_xlim, cur_ylim = self.ax_main.get_xlim(), self.ax_main.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None: return
        scale = 1 / scale_factor if event.button == 'up' else scale_factor
        new_x_span, new_y_span = (cur_xlim[1] - cur_xlim[0]) * scale, (cur_ylim[1] - cur_ylim[0]) * scale
        rel_x, rel_y = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0]), (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        self.ax_main.set_xlim([xdata - new_x_span * (1 - rel_x), xdata + new_x_span * rel_x])
        self.ax_main.set_ylim([ydata - new_y_span * (1 - rel_y), ydata + new_y_span * rel_y])
        self.fig.canvas.draw_idle()

    def _calculate_palette_limits(self):
        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
        # Use prototiles for palette to show all types
        for name, prototile in self.tiler.prototiles.items():
            for i in prototile.allowed_iso_indices:
                template = prototile.visual_templates[i]
                if template is None: continue # Skip if isometry is disallowed
                verts_cart = GEOMETRY.a2_to_cartesian(template['verts'])
                min_x, max_x = min(min_x, verts_cart[:, 0].min()), max(max_x, verts_cart[:, 0].max())
                min_y, max_y = min(min_y, verts_cart[:, 1].min()), max(max_y, verts_cart[:, 1].max())
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        half_size = max(max_x-center_x, center_x-min_x, max_y-center_y, center_y-min_y) * 1.15
        self.palette_xlim, self.palette_ylim = (center_x-half_size, center_x+half_size), (center_y-half_size, center_y+half_size)
    
    def _get_tile_geometry(self, p):
        name, t, i = p['tile_name'], p['translation'], p['iso_idx']
        prototile = self.tiler.prototiles[name]
        is_reflected = i >= 6
        verts = prototile.visual_templates[i]['verts'] + t
        stripes_data = []
        if prototile.matching_rule_type == 'long_range_stripes':
            M = GEOMETRY.isometry_matrices[i]
            for stripe in prototile.stripes:
                p1_emb, p2_emb = M @ stripe['p1'] + t, M @ stripe['p2'] + t
                base_color = stripe['color']
                actual_color = ('red' if base_color == 'blue' else 'blue') if is_reflected else base_color
                stripes_data.append({'p1c': GEOMETRY.a2_to_cartesian(p1_emb), 'p2c': GEOMETRY.a2_to_cartesian(p2_emb),'color': actual_color})
        return {'cartesian_verts': GEOMETRY.a2_to_cartesian(verts).T, 'face_color': '#FFA07A' if is_reflected else '#87CEEB', 'stripes': stripes_data}
        
    def _draw_tile_with_stripes(self, ax, tile_geometry, dominant_stripe_color=None, alpha=1.0, zorder=1):
        artists = []
        patch = ax.fill(*tile_geometry['cartesian_verts'], facecolor=tile_geometry['face_color'], edgecolor='black', lw=1.0, alpha=alpha, zorder=zorder)
        artists.extend(patch)
        if dominant_stripe_color:
            for stripe in tile_geometry['stripes']:
                if stripe['color'] == dominant_stripe_color:
                    p1c, p2c = stripe['p1c'], stripe['p2c']
                    line = ax.plot([p1c[0], p2c[0]], [p1c[1], p2c[1]], c=dominant_stripe_color, lw=2.5, alpha=min(1.0, alpha + 0.2), zorder=zorder + 0.1)
                    artists.extend(line)
        return artists

    def draw(self, tiler, title, analysis=None, forced_move_history=None):
        self.tiler = tiler
        for ax in self.ax_directs + self.ax_reflects + [self.ax_tally_direct, self.ax_tally_reflected]: ax.cla()
        
        # Clear all temporary artists, including previous frontier edges
        for artist in self.temporary_artists:
            if artist.get_figure(): # Check if artist is still associated with a figure
                artist.remove()
        self.temporary_artists = [] # Reset for this draw call

        # --- Plot all points with occupancy == 12 ---
        # occupancy_12_points_a2 = [point for point, count in tiler.occupancies.items() if count == 12]
        # if occupancy_12_points_a2:
        #     occupancy_12_points_cart = GEOMETRY.a2_to_cartesian(np.array(occupancy_12_points_a2))
            
        #     # Plot as small red circles
        #     scatter = self.ax_main.scatter(
        #         occupancy_12_points_cart[:, 0], 
        #         occupancy_12_points_cart[:, 1], 
        #         color='red', 
        #         s=20,  # Size of the points
        #         marker='o', 
        #         alpha=0.7, 
        #         zorder=11 # Ensure they are on top
        #     )
        #     self.temporary_artists.append(scatter) # Add to temporary artists for clearing

        if self.palette_xlim is None: self._calculate_palette_limits()

        all_placements = tiler.placed_tiles + tiler.initial_placements
        num_tiles = len(all_placements)
        
        if num_tiles > self.max_tiles_reached:
            self.max_tiles_reached = num_tiles

        direct_count = sum(1 for p in all_placements if p['iso_idx'] < 6)
        reflected_count = num_tiles - direct_count

        if num_tiles < self.decision_tile_count and self.dominant_color is not None:
            self.dominant_color = None

        if self.dominant_color is None and num_tiles >= self.decision_tile_count:
            if reflected_count > self.majority_threshold:
                self.dominant_color = 'red' 
            elif direct_count > self.majority_threshold:
                self.dominant_color = 'blue' 

        if self.dominant_color:
            draw_color = self.dominant_color
        else: 
            draw_color = 'blue' if direct_count > reflected_count else 'red'
        
        if draw_color != self.last_dominant_color:
            for artist_dict in [self.drawn_placed_artists, self.drawn_boundary_artists]:
                for key in list(artist_dict.keys()):
                    for artist in artist_dict.pop(key): artist.remove()
        self.last_dominant_color = draw_color

        elapsed = (time.time() - tiler.start_time) if tiler.start_time else 0

        if forced_move_history and isinstance(forced_move_history, list) and forced_move_history:
            forced_moves_str = f"{forced_move_history}"
            self.ax_main.set_title(f"Tiles: {num_tiles} | Max Tiles: {self.max_tiles_reached} | Time: {elapsed:.2f}s | {title} | {forced_moves_str}")
        else:
            self.ax_main.set_title(f"Tiles: {num_tiles} | Max Tiles: {self.max_tiles_reached} | Time: {elapsed:.2f}s | {title}")
        
        def sync_artists(placements, drawn_artists_dict, zorder_base_face):
            placements_by_key = {(p['tile_name'], p['iso_idx'], tuple(p['translation'])): p for p in placements}
            current_keys, drawn_keys = set(placements_by_key.keys()), set(drawn_artists_dict.keys())
            for key in drawn_keys - current_keys:
                for artist in drawn_artists_dict.pop(key): artist.remove()
            for key in current_keys - drawn_keys:
                p = placements_by_key[key]
                tile_geometry = self._get_tile_geometry(p)
                artists = self._draw_tile_with_stripes(self.ax_main, tile_geometry, dominant_stripe_color=draw_color, zorder=zorder_base_face)
                drawn_artists_dict[key] = artists
        
        sync_artists(tiler.initial_placements, self.drawn_boundary_artists, zorder_base_face=1)
        sync_artists(tiler.placed_tiles, self.drawn_placed_artists, zorder_base_face=1)

        if all_placements and self.ax_main.get_xlim() == (0.0, 1.0):
            all_verts = GEOMETRY.a2_to_cartesian(np.vstack([tiler.prototiles[p['tile_name']].visual_templates[p['iso_idx']]['verts']+p['translation'] for p in all_placements]))
            min_c, max_c = all_verts.min(axis=0), all_verts.max(axis=0)
            self.ax_main.set_xlim(min_c[0]-5,max_c[0]+5); self.ax_main.set_ylim(min_c[1]-5,max_c[1]+5)
        
        # --- Visualize all_frontier_edges ---
        all_frontier_edges_for_vis = [{'p1': p1, 'p2': p2} for (p1, p2), c in tiler.edge_registry.items() if c == 1]
        
        for edge_info in all_frontier_edges_for_vis:
            p1_a2, p2_a2 = np.array(edge_info['p1']), np.array(edge_info['p2'])
            p1_cart, p2_cart = GEOMETRY.a2_to_cartesian(p1_a2), GEOMETRY.a2_to_cartesian(p2_a2)
            
            line = self.ax_main.plot(
                [p1_cart[0], p2_cart[0]],
                [p1_cart[1], p2_cart[1]],
                color='green', linewidth=3.5, linestyle='-', alpha=0.8, zorder=10
            )
            self.temporary_artists.extend(line) # Add to temporary_artists for clearing

        counts = Counter(p['iso_idx'] for p in all_placements)
        for i in range(6):
            p_d = {'tile_name':'Turtle', 'iso_idx':i, 'translation':np.array([0,0,0])}
            self._draw_tile_with_stripes(self.ax_directs[i], self._get_tile_geometry(p_d), dominant_stripe_color=draw_color)
            self.ax_directs[i].text(3.5, 0, f"x{counts.get(i, 0)}", fontsize=14, va='center', ha='left'); self.ax_directs[i].set_title(f"Direct #{i}", fontsize=9)
            
            p_r = {'tile_name':'Turtle', 'iso_idx':i+6, 'translation':np.array([0,0,0])}
            self._draw_tile_with_stripes(self.ax_reflects[i], self._get_tile_geometry(p_r), dominant_stripe_color=draw_color)
            self.ax_reflects[i].text(3.5, 0, f"x{counts.get(i+6, 0)}", fontsize=14, va='center', ha='left'); self.ax_reflects[i].set_title(f"Reflected #{i+6}", fontsize=9)
            
        for ax in self.ax_directs + self.ax_reflects: ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(self.palette_xlim); ax.set_ylim(self.palette_ylim)
        self.ax_tally_direct.text(0.5, 0.5, f"Total: {direct_count}", ha='center', va='center', fontsize=14, weight='bold')
        self.ax_tally_reflected.text(0.5, 0.5, f"Total: {reflected_count}", ha='center', va='center', fontsize=14, weight='bold')
        for ax in [self.ax_tally_direct, self.ax_tally_reflected]: ax.axis('off')
        self.fig.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5)
        plt.draw(); plt.pause(0.001)


if __name__ == "__main__":
    GEOMETRY = A2Geometry()

    config_single = {
        'initial_placements': [
            {'tile_name':'Turtle', 'iso_idx':9, 'translation':np.array([0,0,0])},
            # {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([2,2,-4])},
            # {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([-2,-2,4])},
            # {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([4,4,-8])},
            # {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([6,6,-12])},
            # {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([-2,-2,4])},
            # {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([-2,-2,4])},
        ],
        'tiling_prototiles': ['Turtle'],
        'tiling_mode': 'plane_filling', 'max_tiles': 10000,
    }

    config_propeller = {
        'initial_placements': [
            {'tile_name':'Propeller', 'iso_idx':0, 'translation':np.array([0,0,0])},
            {'tile_name':'Turtle', 'iso_idx':8, 'translation':np.array([7,8,-15])},
            # {'tile_name':'Turtle', 'iso_idx':6, 'translation':np.array([-15,7,8])},
            # {'tile_name':'Turtle', 'iso_idx':10, 'translation':np.array([8,-16,7])},
        ],
        'tiling_prototiles': ['Turtle'],
        'tiling_mode': 'plane_filling', 'max_tiles': 10000,
    }

    config_dipole = {
        'initial_placements': [
            {'tile_name': 'Turtle', 'iso_idx': 9, 'translation': np.array([2,5,-7])},
            {'tile_name': 'Turtle', 'iso_idx': 6, 'translation': np.array([-2,-5,7])},
                {'tile_name':'Turtle', 'iso_idx':0, 'translation': np.array([0,3,-3])},
                {'tile_name':'Turtle', 'iso_idx':3, 'translation': np.array([0,-3,3])},
        ],
        'tiling_prototiles': ['Turtle'],
        'tiling_mode': 'plane_filling', 'max_tiles': 5000,
    }
        
    config_tripedal = {
        'initial_placements': [
            {'tile_name': 'Turtle', 'iso_idx': 9, 'translation': np.array([7, 1, -8])},
                # {'tile_name':'Turtle', 'iso_idx':0, 'translation': np.array([9, 3, -12])},
                # {'tile_name':'Turtle', 'iso_idx':0, 'translation': np.array([5, -1, -4])},
            {'tile_name': 'Turtle', 'iso_idx': 7, 'translation': np.array([-8, 7, 1])},
            {'tile_name': 'Turtle', 'iso_idx': 11, 'translation': np.array([1, -8, 7])},
            # {'tile_name': 'Turtle', 'iso_idx': 9, 'translation': np.array([1, 5, -6])},
            # {'tile_name': 'Turtle', 'iso_idx': 7, 'translation': np.array([-6, 1, 5])},
            # {'tile_name': 'Turtle', 'iso_idx': 11, 'translation': np.array([5, -6, 1])},
            # {'tile_name': 'Turtle', 'iso_idx': 0, 'translation': np.array([1, 5, -6])},
            # {'tile_name': 'Turtle', 'iso_idx': 2, 'translation': np.array([-6, 1, 5])},
            # {'tile_name': 'Turtle', 'iso_idx': 4, 'translation': np.array([5, -6, 1])},
            # {'tile_name': 'Turtle', 'iso_idx': 0, 'translation': np.array([3, 7, -10])},
            # {'tile_name': 'Turtle', 'iso_idx': 9, 'translation': np.array([5, 9, -14])},
            # {'tile_name': 'Turtle', 'iso_idx': 0, 'translation': np.array([7, 11, -18])},
            # {'tile_name': 'Turtle', 'iso_idx': 2, 'translation': np.array([-10, 3, 7])},
            # {'tile_name': 'Turtle', 'iso_idx': 4, 'translation': np.array([7, -10, 3])},
        ],
        'tiling_prototiles': ['Turtle'],
        'tiling_mode': 'plane_filling', 'max_tiles': 5000,
    }

    n = 16
    print('n =', n)
    config_cycle = {
        'initial_placements': [
            {'tile_name':'Turtle', 'iso_idx':9, 'translation':np.array([n-1, 1, -n])},
                {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([n+1, 3, -n-4])}, # add [2, 2, -4]
                {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([n-3, -1, -n+4])}, # add [-2, -2, 4]
            {'tile_name':'Turtle', 'iso_idx':7, 'translation':np.array([-n, n-1, 1])},
            {'tile_name':'Turtle', 'iso_idx':11, 'translation':np.array([1, -n, n-1])},
        ],
        'tiling_prototiles': ['Turtle'],
        'tiling_mode': 'plane_filling', 'max_tiles': 8000,
    }
    
    SELECTED_CONFIG = config_propeller
    
    vis = StaticVisualizer(show_coronas=False)
    tiler = HeeschTiler(config=SELECTED_CONFIG, visualizer=vis, enable_stripe_heuristic=True)
    
    tiler.tile()
