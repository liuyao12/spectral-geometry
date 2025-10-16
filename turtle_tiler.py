import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.lines as mlines
from collections import Counter, defaultdict
import time
import math
import random
import json
from datetime import datetime

class A2Geometry:
    def __init__(self):
        self.v1_cart, self.v2_cart = np.array([1.0, 0.0]), np.array([-0.5, np.sqrt(3) / 2])
        R = np.array([[ 1, 0,  1], [ 0, 0, -1], [-1, 0,  0]], dtype=int)
        S = np.array([[ 1, 0,  1], [-1, 0,  0], [ 0, 0, -1]], dtype=int)
        rotations = [np.identity(3, dtype=int)]; [rotations.append(R @ rotations[-1]) for _ in range(5)]
        self.isometry_matrices = rotations + [S @ rot for rot in rotations]

    def a2_to_cartesian(self, v):
        v = np.array(v)
        if v.ndim == 1:
            return (v[0] * self.v1_cart - v[2] * self.v2_cart)
        else:
            return (v[:, 0, np.newaxis] * self.v1_cart - v[:, 2, np.newaxis] * self.v2_cart)

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
                if p2y > p1y:
                    if val < 0:
                        inside = not inside
                else:
                    if val > 0:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def get_line_parameters(self, p1_a2, p2_a2):
        p1_trunc = self.a2_trunc(p1_a2)
        p2_trunc = self.a2_trunc(p2_a2)
        line_vec_cart = p2_trunc - p1_trunc
        n_cart_raw_x = -line_vec_cart[1]
        n_cart_raw_y = line_vec_cart[0]
        d_vec = p2_a2 - p1_a2
        n = np.array([d_vec[2]-d_vec[1], d_vec[0]-d_vec[2], d_vec[1]-d_vec[0]], dtype=int)
        if np.all(n == 0):
            return None, None
        gcd_val = math.gcd(math.gcd(n[0], n[1]), n[2])
        un = tuple(n // gcd_val)
        for x in un:
            if x != 0:
                if x < 0:
                    un = tuple(-y for y in un)
                break
        d = np.dot(un, p1_a2)
        return (un, d)

    def generate_visual_templates(self, prototile, allowed_iso_indices=None):
        if allowed_iso_indices is None:
            allowed_iso_indices = range(12)
        templates = []
        for i in range(12):
            if i not in allowed_iso_indices:
                templates.append(None)
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
            templates.append({'verts': verts, 'occupancies': occupancies})
        return templates

    def generate_dual_information(self, prototile, allowed_iso_indices=None):
        if allowed_iso_indices is None:
            allowed_iso_indices = range(12)
        dual_tiles = []
        for i in range(12):
            if i not in allowed_iso_indices:
                dual_tiles.append(None)
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

    # Modified to accept a list of heuristic functions and a single heuristic_data dict
    def are_heuristics_valid(self, placement, heuristic_data, heuristics_to_apply):
        for heuristic_func in heuristics_to_apply:
            if not heuristic_func(self, placement, heuristic_data):
                return False
        return True

# Define individual heuristic functions outside the Prototile class
# Each now expects the heuristic_data dict
def stripe_heuristic(prototile, placement, heuristic_data):
    global_stripe_occupancies = heuristic_data.get('global_stripe_occupancies')
    if global_stripe_occupancies is None: # Heuristic not relevant/enabled or data missing
        return True 

    if prototile.matching_rule_type == 'long_range_stripes':
        stripes_to_check = prototile.dual_info[placement['iso_idx']]
        t = placement['translation']
        for s in stripes_to_check:
            n, d0, c = s['params'][0], s['params'][1], s['color']
            d = d0 + np.dot(n, t)
            stripe_id = (n, d)
            if stripe_id in global_stripe_occupancies:
                registered_color = global_stripe_occupancies[stripe_id]
                if registered_color != c:
                    return False
    return True

def mod_6_heuristic(prototile, placement, heuristic_data):
    initial_mod_6_d_residues = heuristic_data.get('initial_mod_6_d_residues')
    if initial_mod_6_d_residues is None: # Heuristic not relevant/enabled or data missing
        return True

    stripes_to_check = prototile.dual_info[placement['iso_idx']]
    t = placement['translation']
    for s in stripes_to_check:
        n, d0, c = s['params'][0], s['params'][1], s['color']
        d = d0 + np.dot(n, t)
        
        if n in initial_mod_6_d_residues:
            expected_residue = initial_mod_6_d_residues[n]
            if d % 6 != expected_residue:
                return False
    return True

def TurtleTile(allowed_iso_indices=None):
    verts = [[0,-3,3],[1,-3,2],[2,-4,2],[3,-3,0],[2,-1,-1],[1,1,-2],[-1,2,-1],[-1,3,-2],[-2,4,-2],[-3,3,0],[-2,1,1],[-3,1,2],[-3,0,3],[-1,-1,2]]
    angles = [3,8,3,4,6,4,9,4,3,4,9,4,3,8]
    stripe_defs = [
        {'p1': [-1,-4,5], 'p2': [1,-2,1], 'color': 'blue'},
        {'p1': [1,-2,1], 'p2': [-1,2,-1], 'color': 'red'},
        {'p1': [2,-1,-1], 'p2': [-2,1,1], 'color': 'red'},
        {'p1': [-2,1,1], 'p2': [-4,-1,5], 'color': 'red'},
    ]
    return Prototile("Turtle", verts, angles, stripe_defs, matching_rule_type='long_range_stripes', allowed_iso_indices=allowed_iso_indices)

def PropellerTile(allowed_iso_indices=[0,1,6,7]):
    verts = [[1,0,-1],[2,0,-2],[2,1,-3],[0,2,-2],[-1,1,0],[-2,2,0],[-3,2,1],[-2,0,2],[0,-1,1],[0,-2,2],[1,-3,2],[2,-2,0]]
    angles = [9,4,3,4] * 3
    stripe_defs = [
        {'p1': verts[0], 'p2': verts[6], 'color': 'red'},
        {'p1': verts[4], 'p2': verts[10], 'color': 'red'},
        {'p1': verts[8], 'p2': verts[2], 'color': 'red'},
    ]
    return Prototile("Propeller", verts, angles, stripe_defs, matching_rule_type='long_range_stripes', allowed_iso_indices=allowed_iso_indices)

class HeeschTiler:
    def __init__(self, config, visualizer=None, heuristics_to_apply=None): # Changed heuristics_to_enable to heuristics_to_apply
        self.config = config
        self.live_visualizer = visualizer
        self.max_tiles, self.tiling_mode = config.get('max_tiles', 50), config.get('tiling_mode', 'plane_filling')
        
        # Now we directly assign the list of function objects
        self.heuristics_to_apply = heuristics_to_apply if heuristics_to_apply is not None else []
        
        # We also need to keep track of *which* data needs to be passed, based on the actual functions
        # This mapping allows us to build the 'heuristic_data' dict dynamically
        self.required_heuristic_data = set()
        for func in self.heuristics_to_apply:
            if func == stripe_heuristic:
                self.required_heuristic_data.add('global_stripe_occupancies')
            elif func == mod_6_heuristic:
                self.required_heuristic_data.add('initial_mod_6_d_residues')
            # Add mappings for other heuristics here as needed

        self.description = config.get('description', '')

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
        self.occupancies = {}

        # These remain as member variables to hold the global state
        self.global_stripe_occupancies = {}
        self.initial_mod_6_d_residues = {} 

        self.start_time = None
        self.edge_registry = Counter()
        self.choice_indent_level = 0
        self.last_saved_tile_count = 0
        timestamp = datetime.now().strftime("tiling_%Y-%m-%d_%H-%M-%S")
        if self.description:
            self.output_filename = f"{timestamp}_{self.description}.json"
        else:
            self.output_filename = f"{timestamp}.json"

    def _get_edges_of_placement(self, p):
        verts = self.prototiles[p['tile_name']].visual_templates[p['iso_idx']]['verts'] + p['translation']
        return [(tuple(verts[i-1]), tuple(verts[i])) for i in range(len(verts))]

    def _apply_placement_to_state(self, placement, is_initial=False):
        name, i, t = placement['tile_name'], placement['iso_idx'], placement['translation']
        prototile = self.prototiles[name]
        tile_data = prototile.visual_templates[i]

        occupancies_added = {}
        for pt in tile_data['occupancies']:
            val = tile_data['occupancies'][pt]
            current_count = self.occupancies.get(tuple(pt+t), 0)
            if current_count + val > 12:
                print(f"FATAL ERROR: Occupancy conflict at {tuple(pt+t)}: {current_count} + {val} > 12")
                return None
            self.occupancies[tuple(pt+t)] = current_count + val
            occupancies_added[tuple(pt+t)] = val

        newly_added_stripes_for_undo = []
        # Check if global_stripe_occupancies is required by any enabled heuristic
        if 'global_stripe_occupancies' in self.required_heuristic_data and prototile.matching_rule_type == 'long_range_stripes':
            for s in prototile.dual_info[i]:
                n, d0, c = s['params'][0], s['params'][1], s['color']
                d = d0 + np.dot(n, t)
                stripe_id = (n, d)

                if is_initial:
                    if stripe_id in self.global_stripe_occupancies and self.global_stripe_occupancies[stripe_id] != c:
                        print(f"FATAL ERROR: Initial placement {name} {t.tolist()} conflicts with already established stripe {stripe_id}. Existing color: {self.global_stripe_occupancies[stripe_id]}, New color: {c}")
                        for pt_coords, val_undo in occupancies_added.items():
                            self.occupancies[pt_coords] -= val_undo
                            if self.occupancies[pt_coords] == 0: del self.occupancies[pt_coords]
                        return None
                    self.global_stripe_occupancies[stripe_id] = c
                else:
                    if stripe_id not in self.global_stripe_occupancies:
                        self.global_stripe_occupancies[stripe_id] = c
                        newly_added_stripes_for_undo.append(stripe_id)

        # Check if initial_mod_6_d_residues is required by any enabled heuristic
        if is_initial and 'initial_mod_6_d_residues' in self.required_heuristic_data:
            for s in prototile.dual_info[i]:
                n, d0, c = s['params'][0], s['params'][1], s['color']
                d = d0 + np.dot(n, t)
                current_residue = d % 6
                if n in self.initial_mod_6_d_residues:
                    if self.initial_mod_6_d_residues[n] != current_residue:
                        print(f"FATAL ERROR: Initial placement {name} {t.tolist()} has stripe {n} with d%6={current_residue}, but another initial tile established d%6={self.initial_mod_6_d_residues[n]} for this 'n'.")
                self.initial_mod_6_d_residues[n] = current_residue

        edges_added_for_undo = []
        for edge in self._get_edges_of_placement(placement):
            p1, p2 = edge
            self.edge_registry[(p1, p2)] += 1
            self.edge_registry[(p2, p1)] -= 1
            edges_added_for_undo.append(edge)

        if is_initial: self.initial_placements.append(placement)
        else: self.placed_tiles.append(placement)

        return {
            'occupancies_added': occupancies_added,
            'newly_added_stripes': newly_added_stripes_for_undo,
            'edges_added': edges_added_for_undo,
            'is_initial': is_initial,
            'placement_ref': placement,
            'initial_mod_6_d_residues_snapshot': dict(self.initial_mod_6_d_residues) if is_initial else None
        }

    def _undo_placement(self, undo_info):
        placement_ref = undo_info['placement_ref']
        target_list = self.initial_placements if undo_info['is_initial'] else self.placed_tiles
        idx_to_remove = next((i for i, p in enumerate(target_list) if p['tile_name'] == placement_ref['tile_name'] and p['iso_idx'] == placement_ref['iso_idx'] and np.array_equal(p['translation'], placement_ref['translation'])), -1)
        if idx_to_remove != -1: target_list.pop(idx_to_remove)
        for p_coords, val_removed in undo_info['occupancies_added'].items():
            self.occupancies[p_coords] -= val_removed
            if self.occupancies[p_coords] == 0: del self.occupancies[p_coords]
        if not undo_info['is_initial']:
            for stripe_id in undo_info['newly_added_stripes']:
                if stripe_id in self.global_stripe_occupancies:
                    del self.global_stripe_occupancies[stripe_id]
        else:
            if undo_info['initial_mod_6_d_residues_snapshot'] is not None:
                 self.initial_mod_6_d_residues = undo_info['initial_mod_6_d_residues_snapshot']

        for edge in undo_info['edges_added']:
            p1, p2 = edge
            self.edge_registry[(p1, p2)] -= 1
            self.edge_registry[(p2, p1)] += 1
            if self.edge_registry[(p1, p2)] == 0: del self.edge_registry[(p1, p2)]
            if self.edge_registry[(p2, p1)] == 0: del self.edge_registry[(p2, p1)]

    def _get_candidate_placements(self, edge):
        cands, p1, p2 = [], np.array(edge['p1']), np.array(edge['p2']); target = p1 - p2
        for name, prototile in self.tiling_prototiles.items():
            for i in prototile.allowed_iso_indices:
                template = prototile.visual_templates[i]
                if template is None: continue
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
        
        # Construct heuristic_data based on self.required_heuristic_data
        heuristic_data = {}
        if 'global_stripe_occupancies' in self.required_heuristic_data:
            heuristic_data['global_stripe_occupancies'] = self.global_stripe_occupancies
        if 'initial_mod_6_d_residues' in self.required_heuristic_data:
            heuristic_data['initial_mod_6_d_residues'] = self.initial_mod_6_d_residues

        # Pass heuristic_data to the general heuristic validation method
        return prototile.are_heuristics_valid(p, heuristic_data, self.heuristics_to_apply)

    def _save_tiling_state(self):
        all_tiles_for_save = []
        for p in self.initial_placements + self.placed_tiles:
            all_tiles_for_save.append({
                'tile_name': p['tile_name'],
                'iso_idx': p['iso_idx'],
                'translation': p['translation'].tolist()
            })

        with open(self.output_filename, 'w') as f:
            json.dump(all_tiles_for_save, f, indent=4)
        print(f"Saved {len(all_tiles_for_save)} tiles to {self.output_filename}")

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
                total_tiles_count = len(self.placed_tiles) + len(self.initial_placements)
                if (total_tiles_count // 100) * 100 > self.last_saved_tile_count:
                    save_threshold = (total_tiles_count // 100) * 100
                    self._save_tiling_state()
                    self.last_saved_tile_count = save_threshold
                    if self.live_visualizer:
                        self.live_visualizer.draw(self, f"Autosaved {total_tiles_count} tiles", forced_move_history=all_forced_moves)

                if total_tiles_count >= self.max_tiles: return True

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
                        undo_info = self._apply_placement_to_state(p)
                        if undo_info is None:
                            for u in reversed(undo_stack_for_this_level): self._undo_placement(u)
                            return False
                        undo_stack_for_this_level.append(undo_info)
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
                if not self._is_placement_valid(placement):
                    # This case should ideally not be hit if _get_candidate_placements is robust,
                    # but it's a safeguard against complex interactions not caught earlier.
                    # print(f"DEBUG: Skipping invalid placement in choice branch. Tile: {placement['tile_name']} iso:{placement['iso_idx']} trans:{placement['translation'].tolist()}")
                    continue

                undo_info = self._apply_placement_to_state(placement)
                if undo_info is None:
                    print(f"FATAL ERROR during choice branch placement. Backtracking.")
                    continue

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

        for i, p_data in enumerate(self.config['initial_placements']):
            print(f"Attempting initial placement #{i+1}: {p_data['tile_name']} (iso_idx: {p_data['iso_idx']}, translation: {p_data['translation']})")
            current_mod_6_residues_snapshot = dict(self.initial_mod_6_d_residues)
            undo_info = self._apply_placement_to_state(p_data, is_initial=True)
            if undo_info is None:
                print(f"Tiling aborted due to fatal error in initial placement #{i+1}.")
                self.initial_mod_6_d_residues = current_mod_6_residues_snapshot
                return

            if self.live_visualizer:
                self.live_visualizer.draw(self, f"Initial Placement #{i+1} Applied", forced_move_history=[])

        self.global_center_emb = np.mean([p['translation'] for p in self.initial_placements], axis=0) if self.initial_placements else np.array([0,0,0])
        if self.live_visualizer:
            self.live_visualizer.draw(self, "Initial State Ready", forced_move_history=[])

        print("\n--- Starting Search ---"); self.start_time = time.time()
        success = self._search(last_choice_edge=None, level=0)
        print(f"\n--- Search Complete in {time.time() - self.start_time:.2f}s ---")
        print(f"Result: {'SUCCESS' if success else 'FAILURE'} | Total Tiles: {len(self.placed_tiles) + len(self.initial_placements)}")

        if (len(self.placed_tiles) + len(self.initial_placements)) > self.last_saved_tile_count:
            self._save_tiling_state()

        if self.live_visualizer:
            self.live_visualizer.draw(self, "Final Tiling - SUCCESS" if success else "Final State - FAILURE", forced_move_history=[])
            plt.ioff(); plt.show()

class StaticVisualizer:
    def __init__(self, show_coronas=True, initial_enable_stripe_heuristic=True, background_tiling_file=None):
        self.fig = plt.figure(figsize=(12, 10))
        gs = self.fig.add_gridspec(10, 8)
        self.ax_main = self.fig.add_subplot(gs[:, 0:7])
        self.ax_main.set_aspect('equal', adjustable='box')
        plt.ion()
        self.fig.canvas.manager.set_window_title("Heesch Tiler")
        self.tiler = None
        self.color_map = {0:'magenta', 1:'lime', 2:'yellow'}
        self.drawn_placed_artists = {}
        self.drawn_boundary_artists = {}
        self.temporary_artists = []
        self.palette_xlim, self.palette_ylim = None, None
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.dominant_color = None
        self.decision_tile_count = 10
        self.majority_threshold = 5
        self.last_dominant_color = None
        self.max_tiles_reached = 0
        self.background_tiling_file = background_tiling_file
        self.background_placements = []
        self.permanent_background_face_artists = [] 
        self.permanent_background_stripe_artists = []
        self.all_active_stripe_artists_info = []
        self.ax_radio_stripes = self.fig.add_subplot(gs[2:7, 7])
        radio_labels = ('Blue Stripes', 'Red Stripes', 'Both Stripes', 'No Stripes')
        self.radio_buttons = RadioButtons(self.ax_radio_stripes, radio_labels)
        self.radio_buttons.on_clicked(self._on_radio_select)
        self.current_stripe_display_mode = 'Blue Stripes'
        self.radio_buttons.set_active(0)
        self.ax_radio_stripes.set_facecolor('lightgrey')
        self.ax_radio_stripes.tick_params(axis='both', which='both', length=0)
        self.ax_radio_stripes.set_xticks([])
        self.ax_radio_stripes.set_yticks([])
        self.ax_radio_stripes.set_title('Stripe Display')

        if self.background_tiling_file:
            print(f"Loading background tiling from: {self.background_tiling_file}")
            self._load_and_draw_background()

    def _load_and_draw_background(self):
        if not self.background_tiling_file or self.permanent_background_face_artists:
            return

        try:
            with open(self.background_tiling_file, 'r') as f:
                data = json.load(f)
            background_prototiles = {}
            for p_data in data:
                tile_name = p_data['tile_name']
                if tile_name not in background_prototiles:
                    if tile_name == "Turtle":
                        background_prototiles[tile_name] = TurtleTile()
                    elif tile_name == "Propeller":
                        background_prototiles[tile_name] = PropellerTile()

            for p_data in data:
                translation_array = np.array(p_data['translation'], dtype=int)
                self.background_placements.append({
                    'tile_name': p_data['tile_name'],
                    'iso_idx': p_data['iso_idx'],
                    'translation': translation_array
                })

            print(f"Successfully loaded {len(self.background_placements)} tiles for background.")

            # Create a dummy tiler for background with no heuristics applied
            temp_tiler = HeeschTiler(config={'initial_placements': [], 'tiling_prototiles': list(background_prototiles.keys())}, heuristics_to_apply=[]) 
            temp_tiler.prototiles.update(background_prototiles)

            for p in self.background_placements:
                name, i, t = p['tile_name'], p['iso_idx'], p['translation']
                prototile = background_prototiles.get(name) 
                if prototile:
                    temp_p = {'tile_name': name, 'iso_idx': i, 'translation': t}
                    tile_geometry = self._get_tile_geometry(temp_p, prototile=prototile)

                    patch = self.ax_main.fill(*tile_geometry['cartesian_verts'], facecolor=tile_geometry['face_color'],
                                              edgecolor='black', lw=1.0, alpha=0.3, zorder=0)
                    self.permanent_background_face_artists.extend(patch)

                    # for stripe in tile_geometry['stripes']:
                    #     line = self.ax_main.plot([stripe['p1c'][0], stripe['p2c'][0]],
                    #                               [stripe['p1c'][1], stripe['p2c'][1]],
                    #                               c=stripe['color'], lw=2.5, alpha=0.5, zorder=0.1)
                    #     for l in line:
                    #         l.set_visible(self._should_draw_stripe(stripe['color'])) 
                    #         self.permanent_background_stripe_artists.append({'artist': l, 'color': stripe['color']})
                    #         self.all_active_stripe_artists_info.append({'artist': l, 'color': stripe['color']})
                else:
                    print(f"Warning: Prototile '{name}' for background not found during loading.")
            self.fig.canvas.draw_idle() 

        except FileNotFoundError:
            print(f"Error: Background tiling file not found at {self.background_tiling_file}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.background_tiling_file}")
        except Exception as e:
            print(f"An unexpected error occurred while loading background tiling: {e}")

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

    def _on_radio_select(self, label):
        self.current_stripe_display_mode = label
        self._redraw_main_stripes() 
        self.fig.canvas.draw_idle()

    def _should_draw_stripe(self, stripe_color):
        if self.current_stripe_display_mode == 'Both Stripes':
            return True
        elif self.current_stripe_display_mode == 'No Stripes':
            return False
        elif self.current_stripe_display_mode == 'Blue Stripes' and stripe_color == 'blue':
            return True
        elif self.current_stripe_display_mode == 'Red Stripes' and stripe_color == 'red':
            return True
        return False

    def _redraw_main_stripes(self):
        for stripe_info in self.all_active_stripe_artists_info:
            stripe_info['artist'].set_visible(self._should_draw_stripe(stripe_info['color']))

    def _calculate_palette_limits(self):
        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')

        if self.tiler and self.tiler.prototiles:
            for p in self.tiler.initial_placements + self.tiler.placed_tiles:
                key = (p['tile_name'], p['iso_idx'], tuple(p['translation']))
                if key in self.drawn_placed_artists:
                    tile_info = self.drawn_placed_artists[key]
                    verts_cart = tile_info['tile_geometry']['cartesian_verts'].T
                elif key in self.drawn_boundary_artists:
                    tile_info = self.drawn_boundary_artists[key]
                    verts_cart = tile_info['tile_geometry']['cartesian_verts'].T
                else:
                    tile_geometry = self._get_tile_geometry(p)
                    verts_cart = tile_geometry['cartesian_verts'].T

                min_x, max_x = min(min_x, verts_cart[:, 0].min()), max(max_x, verts_cart[:, 0].max())
                min_y, max_y = min(min_y, verts_cart[:, 1].min()), max(max_y, verts_cart[:, 1].max())

        if self.permanent_background_face_artists:
            for patch in self.permanent_background_face_artists:
                verts_cart = patch.get_path().vertices
                min_x, max_x = min(min_x, verts_cart[:, 0].min()), max(max_x, verts_cart[:, 0].max())
                min_y, max_y = min(min_y, verts_cart[:, 1].min()), max(max_y, verts_cart[:, 1].max())

        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        half_size = max(max_x-center_x, center_x-min_x, max_y-center_y, center_y-min_y) * 1.15
        if math.isinf(half_size):
            self.palette_xlim, self.palette_ylim = (-5, 5), (-5, 5)
        else:
            self.palette_xlim, self.palette_ylim = (center_x-half_size, center_x+half_size), (center_y-half_size, center_y+half_size)


    def _get_tile_geometry(self, p, prototile=None):
        name, t, i = p['tile_name'], p['translation'], p['iso_idx']
        if prototile is None:
            if self.tiler and name in self.tiler.prototiles:
                prototile = self.tiler.prototiles[name]
            else:
                print(f"Error: Prototile '{name}' not found for _get_tile_geometry.")
                return None
        
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


    def draw(self, tiler, title, analysis=None, forced_move_history=None):
        self.tiler = tiler
        for artist in self.temporary_artists:
            if artist.get_figure():
                artist.remove()
        self.temporary_artists = []

        all_placements = tiler.placed_tiles + tiler.initial_placements
        num_tiles = len(all_placements)
        if num_tiles > self.max_tiles_reached:
            self.max_tiles_reached = num_tiles

        elapsed = (time.time() - tiler.start_time) if tiler.start_time else 0
        if forced_move_history and isinstance(forced_move_history, list) and forced_move_history:
            forced_moves_str = f"{forced_move_history}"
            self.ax_main.set_title(f"Tiles: {num_tiles} | Max Tiles: {self.max_tiles_reached} | Time: {elapsed:.2f}s | {title} | {forced_moves_str}")
        else:
            self.ax_main.set_title(f"Tiles: {num_tiles} | Max Tiles: {self.max_tiles_reached} | Time: {elapsed:.2f}s | {title}")

        def sync_dynamic_tiles(placements, drawn_artists_dict, zorder_base_face, opacity=1.0):
            current_placement_keys = set()
            new_stripe_artists_to_add = []
            for p in placements:
                key = (p['tile_name'], p['iso_idx'], tuple(p['translation']))
                current_placement_keys.add(key)

                if key not in drawn_artists_dict:
                    tile_geometry = self._get_tile_geometry(p)
                    if tile_geometry is None: 
                        continue
                    face_patch = self.ax_main.fill(*tile_geometry['cartesian_verts'], facecolor=tile_geometry['face_color'],
                                                   edgecolor='black', lw=1.0, alpha=opacity, zorder=zorder_base_face)
                    stripe_artists_for_this_tile = []
                    for stripe in tile_geometry['stripes']:
                        line = self.ax_main.plot([stripe['p1c'][0], stripe['p2c'][0]],
                                                  [stripe['p1c'][1], stripe['p2c'][1]],
                                                  c=stripe['color'], lw=2.5, alpha=min(1.0, opacity + 0.2), zorder=zorder_base_face + 0.1)
                        for l in line:
                            l.set_visible(self._should_draw_stripe(stripe['color']))
                            stripe_artists_for_this_tile.append({'artist': l, 'color': stripe['color']})
                            new_stripe_artists_to_add.append({'artist': l, 'color': stripe['color']})
                    drawn_artists_dict[key] = {
                        'face_patches': face_patch,
                        'stripe_artists_info': stripe_artists_for_this_tile,
                        'tile_geometry': tile_geometry
                    }
                else:
                    # EXISTING TILE: Ensure properties are correct (e.g., alpha, zorder)
                    # For simplicity, we assume these don't change for already drawn tiles
                    # If they do, update like: drawn_artists_dict[key]['face_patches'][0].set_alpha(opacity)
                    pass
            keys_to_remove = set(drawn_artists_dict.keys()) - current_placement_keys
            for key in keys_to_remove:
                tile_info = drawn_artists_dict.pop(key)
                for patch in tile_info['face_patches']:
                    if patch.get_figure(): patch.remove()
                for stripe_info in tile_info['stripe_artists_info']:
                    if stripe_info['artist'].get_figure(): stripe_info['artist'].remove()
                    if stripe_info in self.all_active_stripe_artists_info:
                        self.all_active_stripe_artists_info.remove(stripe_info)
            self.all_active_stripe_artists_info.extend(new_stripe_artists_to_add)
        sync_dynamic_tiles(tiler.initial_placements, self.drawn_boundary_artists, zorder_base_face=1, opacity=0.7)
        sync_dynamic_tiles(tiler.placed_tiles, self.drawn_placed_artists, zorder_base_face=1, opacity=0.7)
        self._redraw_main_stripes()
        if all_placements and self.ax_main.get_xlim() == (0.0, 1.0) and self.ax_main.get_ylim() == (0.0, 1.0):
            self._calculate_palette_limits()
            if self.palette_xlim and self.palette_ylim:
                self.ax_main.set_xlim(self.palette_xlim)
                self.ax_main.set_ylim(self.palette_ylim)
        elif self.background_placements and not all_placements and \
             self.ax_main.get_xlim() == (0.0, 1.0) and self.ax_main.get_ylim() == (0.0, 1.0):
            self._calculate_palette_limits()
            if self.palette_xlim and self.palette_ylim:
                self.ax_main.set_xlim(self.palette_xlim)
                self.ax_main.set_ylim(self.palette_ylim)


        self.fig.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5)
        plt.draw(); plt.pause(0.001)

if __name__ == "__main__":
    GEOMETRY = A2Geometry()

    config_single = {
        'initial_placements': [
            {'tile_name':'Propeller', 'iso_idx':7, 'translation':np.array([0,0,0])},
            # {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([2,2,-4])},
            # {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([-2,-2,4])},
            # {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([4,4,-8])},
            # {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([6,6,-12])},
            # {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([-2,-2,4])},
            # {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([-2,-2,4])},
        ],
        'tiling_prototiles': ['Turtle'],
        'tiling_mode': 'plane_filling', 'max_tiles': 10000,
        'description': 'single',
    }

    config_propeller = {
        'initial_placements': [
            {'tile_name':'Propeller', 'iso_idx':7, 'translation':np.array([0,0,0])},
            {'tile_name':'Propeller', 'iso_idx':6, 'translation':np.array([6,8,-14])},
            # {'tile_name':'Turtle', 'iso_idx':7, 'translation':np.array([6,7,-13])},
            # {'tile_name':'Turtle', 'iso_idx':11, 'translation':np.array([-13,6,7])},
            # {'tile_name':'Turtle', 'iso_idx':9, 'translation':np.array([7,-13,6])},
        ],
        'tiling_prototiles': ['Turtle'],
        'tiling_mode': 'plane_filling', 'max_tiles': 10000,
        'description': 'two_propellers',
    }

    config_dipole = {
        'initial_placements': [
            {'tile_name': 'Turtle', 'iso_idx': 9, 'translation': np.array([4,7,-11])},
            {'tile_name': 'Turtle', 'iso_idx': 6, 'translation': np.array([-4,-7,11])},
                # {'tile_name':'Turtle', 'iso_idx':0, 'translation': np.array([0,3,-3])},
                # {'tile_name':'Turtle', 'iso_idx':3, 'translation': np.array([0,-3,3])},
        ],
        'tiling_prototiles': ['Turtle'],
        'tiling_mode': 'plane_filling', 'max_tiles': 5000,
        'description': 'dipole',
    }
    
    n = 15
    config_tripod = {
        'initial_placements': [
            {'tile_name':'Turtle', 'iso_idx':8, 'translation':np.array([n, 0, -n])},
            {'tile_name':'Turtle', 'iso_idx':6, 'translation':np.array([-n, n, 0])},
            {'tile_name':'Turtle', 'iso_idx':10, 'translation':np.array([0, -n, n])},
        ],
        'tiling_prototiles': ['Turtle'],
        'tiling_mode': 'plane_filling', 'max_tiles': 5000,
        'description': 'tripod',
    }

    # for n in [8]: # [8, 16, 24, 32, 40]:
    #     print('n =', n)

    n = 8
    config_cycle = {
        'initial_placements': [
            {'tile_name':'Turtle', 'iso_idx':8, 'translation':np.array([n-1, 1, -n])},
            {'tile_name':'Turtle', 'iso_idx':6, 'translation':np.array([-n, n-1, 1])},
            {'tile_name':'Turtle', 'iso_idx':10, 'translation':np.array([1, -n, n-1])},
                # {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([n+1, 3, -n-4])}, 
                # {'tile_name':'Turtle', 'iso_idx':2, 'translation':np.array([-n-4, n+1, 3])},
                # {'tile_name':'Turtle', 'iso_idx':4, 'translation':np.array([3, -n-4, n+1])},
                # {'tile_name':'Turtle', 'iso_idx':0, 'translation':np.array([n-3, -1, -n+4])},
                # {'tile_name':'Turtle', 'iso_idx':2, 'translation':np.array([-n+4, n-3, -1])},
                # {'tile_name':'Turtle', 'iso_idx':4, 'translation':np.array([-1, -n+4, n-3])},
        ],
        'tiling_prototiles': ['Turtle'],
        'tiling_mode': 'plane_filling', 'max_tiles': 10000,
        'description': f'cycle_n={n}',
    }
    
    SELECTED_CONFIG = config_tripod
    
    background_file = "tiling_2025-10-10_09-21-53_cycle_n=16.json" 
    # background_file = "tiling_2025-10-13_18-10-28_cycle_n=8.json"
    
    vis = StaticVisualizer(show_coronas=False, background_tiling_file=None)
    tiler = HeeschTiler(config=SELECTED_CONFIG, visualizer=vis, 
                        heuristics_to_apply=[
                            mod_6_heuristic,  # Pass function objects directly
                            stripe_heuristic, # Pass function objects directly
                        ])
        
    tiler.tile()
        # plt.ioff()
        # plt.close('all')
