import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.lines as mlines
from collections import defaultdict
import time
import math
import random
import json
from datetime import datetime
import copy

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

    def cartesian_to_a2(self, cart_coords):
        cart_x, cart_y = cart_coords[0], cart_coords[1]
        v2_float = cart_y / (np.sqrt(3) / 2)
        v0_float = cart_x + 0.5 * v2_float
        v1_float = -v0_float - v2_float 
        return np.array([round(v0_float), round(v1_float), round(v2_float)], dtype=int)

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

    def are_heuristics_valid(self, placement, heuristic_data, heuristics_to_apply):
        for heuristic_func in heuristics_to_apply:
            if not heuristic_func(self, placement, heuristic_data):
                return False
        return True

def stripe_alignment_heuristic(prototile, placement, heuristic_data):
    global_stripe_occupancies = heuristic_data.get('global_stripe_occupancies') 
    if global_stripe_occupancies is None: 
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

def stripe_separation_heuristic(prototile, placement, heuristic_data):
    initial_mod_3_d_residues = heuristic_data.get('initial_mod_3_d_residues')
    if initial_mod_3_d_residues is None:
        return True

    stripes_to_check = prototile.dual_info[placement['iso_idx']]
    t = placement['translation']
    for s in stripes_to_check:
        n, d0, c = s['params'][0], s['params'][1], s['color']
        d = d0 + np.dot(n, t)
        
        if n in initial_mod_3_d_residues:
            expected_residue = initial_mod_3_d_residues[n]
            if d % 3 != expected_residue:
                return False
    return True

def TurtleTile(allowed_iso_indices=None):
    verts = [[0,-3,3],[1,-3,2],[2,-4,2],[3,-3,0],[2,-1,-1],[1,1,-2],[-1,2,-1],[-1,3,-2],[-2,4,-2],[-3,3,0],[-2,1,1],[-3,1,2],[-3,0,3],[-1,-1,2]]
    angles = [3,8,3,4,6,4,9,4,3,4,9,4,3,8]
    stripe_defs = [
        {'p1': [-1,-4,5], 'p2': [1,-2,1], 'color': 'blue'},
        {'p1': [2,-4,2], 'p2': [-2,4,-2], 'color': 'red'},
        {'p1': [2,-1,-1], 'p2': [-2,1,1], 'color': 'red'},
        {'p1': [-2,1,1], 'p2': [-4,-1,5], 'color': 'red'},
    ]
    return Prototile("Turtle", verts, angles, stripe_defs, matching_rule_type='long_range_stripes', allowed_iso_indices=allowed_iso_indices)

def PropellerTile(allowed_iso_indices=[0,3,6,9]):
    verts = [[1,0,-1],[2,0,-2],[2,1,-3],[0,2,-2],[-1,1,0],[-2,2,0],[-3,2,1],[-2,0,2],[0,-1,1],[0,-2,2],[1,-3,2],[2,-2,0]]
    angles = [9,4,3,4] * 3
    stripe_defs = [
        {'p1': verts[0], 'p2': verts[6], 'color': 'red'},
        {'p1': verts[4], 'p2': verts[10], 'color': 'red'},
        {'p1': verts[8], 'p2': verts[2], 'color': 'red'},
    ]
    return Prototile("Propeller", verts, angles, stripe_defs, matching_rule_type='long_range_stripes', allowed_iso_indices=allowed_iso_indices)

class HeeschTiler:
    def __init__(self, config, visualizer=None, heuristics_to_apply=None):
        self.config = config
        self.live_visualizer = visualizer
        self.max_tiles, self.tiling_mode = config.get('max_tiles', 50), config.get('tiling_mode', 'plane_filling')
        
        self.heuristics_to_apply = heuristics_to_apply if heuristics_to_apply is not None else []
        
        self.required_heuristic_data = set()
        for func in self.heuristics_to_apply:
            if func == stripe_alignment_heuristic:
                self.required_heuristic_data.add('global_stripe_occupancies')
            elif func == stripe_separation_heuristic:
                self.required_heuristic_data.add('initial_mod_3_d_residues')

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

        self.tiling_prototiles = {name: self.prototiles[name] for name in config['tiling_prototiles']}

        self.initial_placements = [] 

        self.start_time = None
        self.last_saved_tile_count = 0
        timestamp = datetime.now().strftime("tiling_%Y-%m-%d_%H-%M-%S")
        if self.description:
            self.output_filename = f"{timestamp}_{self.description}.json"
        else:
            self.output_filename = f"{timestamp}.json"
        
        self.largest_tiling_state = None
        self.max_tiles_found_so_far = 0

    def _get_edges_of_placement(self, p):
        verts = self.prototiles[p['tile_name']].visual_templates[p['iso_idx']]['verts'] + p['translation']
        return [(tuple(verts[i-1]), tuple(verts[i])) for i in range(len(verts))]

    def _apply_placement_to_state(self, placement, current_tiling_state, is_initial=False):
        name, i, t = placement['tile_name'], placement['iso_idx'], placement['translation']
        prototile = self.prototiles[name]
        tile_data = prototile.visual_templates[i]

        new_edge_registry = current_tiling_state['edge_registry'].copy()
        new_global_stripe_occupancies = current_tiling_state['global_stripe_occupancies'].copy()
        new_initial_mod_3_d_residues = current_tiling_state['initial_mod_3_d_residues'].copy()
        new_occupancies = current_tiling_state['occupancies'].copy()
        new_placed_tiles = list(current_tiling_state['placed_tiles'])

        occupancies_added_this_step = {}

        for pt_local, val in tile_data['occupancies'].items():
            pt_global = tuple(np.array(pt_local) + t)
            current_count = new_occupancies.get(pt_global, 0)
            if current_count + val > 12:
                print('FATAL ERROR: Placement exceeds occupancy limit at point', pt_global, '--- Current:', current_count, 'Existing:', val)
                if self.live_visualizer:
                    self.live_visualizer.draw(tiler=self, tiling_state_to_draw=current_tiling_state,
                                              title="FATAL ERROR: Occupancy conflict!",
                                              failed_candidate=placement,
                                              highlight_point=pt_global)
                    plt.pause(0) 
                    input("Press Enter to continue...")
                return None
            occupancies_added_this_step[pt_global] = val

        for pt_global, val in occupancies_added_this_step.items():
            new_occupancies[pt_global] = new_occupancies.get(pt_global, 0) + val


        if 'global_stripe_occupancies' in self.required_heuristic_data and prototile.matching_rule_type == 'long_range_stripes':
            for s in prototile.dual_info[i]:
                n, d0, c = s['params'][0], s['params'][1], s['color']
                d = d0 + np.dot(n, t)
                stripe_id = (n, d)
                if is_initial:
                    if stripe_id in new_global_stripe_occupancies and new_global_stripe_occupancies[stripe_id] != c:
                        print(f"FATAL ERROR: Initial placement {name} {t.tolist()} conflicts with already established stripe {stripe_id}. Existing color: {new_global_stripe_occupancies[stripe_id]}, New color: {c}")
                        if self.live_visualizer:
                            self.live_visualizer.draw(tiler=self, tiling_state_to_draw=current_tiling_state,
                                                      title="FATAL ERROR: Stripe alignment conflict!",
                                                      failed_candidate=placement,
                                                      highlight_stripe=(n, d, c))
                            plt.pause(0)
                            input("Press Enter to continue...")
                        return None 
                    new_global_stripe_occupancies[stripe_id] = c
                else:
                    if stripe_id in new_global_stripe_occupancies and new_global_stripe_occupancies[stripe_id] != c:
                        return None
                    new_global_stripe_occupancies[stripe_id] = c

        if is_initial and 'initial_mod_3_d_residues' in self.required_heuristic_data:
            for s in prototile.dual_info[i]:
                n, d0, c = s['params'][0], s['params'][1], s['color']
                d = d0 + np.dot(n, t)
                current_residue = d % 3
                if n in new_initial_mod_3_d_residues:
                    if new_initial_mod_3_d_residues[n] != current_residue:
                        print(f"FATAL ERROR: Initial placement {name} {t.tolist()} has stripe {n} with d%3={current_residue}, but another initial tile established d%3={new_initial_mod_3_d_residues[n]} for this 'n'.")
                        if self.live_visualizer:
                            self.live_visualizer.draw(tiler=self, tiling_state_to_draw=current_tiling_state,
                                                      title="FATAL ERROR: Stripe separation conflict!",
                                                      failed_candidate=placement,
                                                      highlight_stripe=(n, d, c))
                            plt.pause(0)
                            input("Press Enter to continue...")
                        return None 
                new_initial_mod_3_d_residues[n] = current_residue

        for edge in self._get_edges_of_placement(placement):
            p1, p2 = edge
            new_edge_registry[(p1, p2)] += 1
            new_edge_registry[(p2, p1)] -= 1
            if new_edge_registry[(p1,p2)] == 0:
                del new_edge_registry[(p1,p2)]
            if new_edge_registry[(p2,p1)] == 0:
                del new_edge_registry[(p2,p1)]


        placement_with_level = placement.copy()
        placement_with_level['_level'] = placement.get('_level', 0)

        if is_initial:
            self.initial_placements.append(placement_with_level)
        else:
            new_placed_tiles.append(placement_with_level)

        return {
            'edge_registry': new_edge_registry,
            'global_stripe_occupancies': new_global_stripe_occupancies,
            'initial_mod_3_d_residues': new_initial_mod_3_d_residues,
            'occupancies': new_occupancies,
            'placed_tiles': new_placed_tiles
        }

    def _is_placement_valid(self, p, tiling_state):
        name, i, t = p['tile_name'], p['iso_idx'], p['translation']
        prototile = self.prototiles[name]
        tile = prototile.visual_templates[i]
        
        current_occupancies = tiling_state['occupancies'] 

        for pt in tile['occupancies']:
            if current_occupancies.get(tuple(np.array(pt)+t), 0) + tile['occupancies'][tuple(pt)] > 12: 
                return False
        
        heuristic_data = {}
        if 'global_stripe_occupancies' in self.required_heuristic_data:
            heuristic_data['global_stripe_occupancies'] = tiling_state['global_stripe_occupancies']
        if 'initial_mod_3_d_residues' in self.required_heuristic_data:
            heuristic_data['initial_mod_3_d_residues'] = tiling_state['initial_mod_3_d_residues']

        return prototile.are_heuristics_valid(p, heuristic_data, self.heuristics_to_apply)

    def _get_candidate_placements(self, edge, level): 
        cands, p1, p2 = [], np.array(edge['p1']), np.array(edge['p2']); target = p1 - p2
        for name, prototile in self.tiling_prototiles.items():
            for i in prototile.allowed_iso_indices:
                template = prototile.visual_templates[i]
                if template is None: continue
                verts = template['verts']
                for j in range(len(verts)):
                    if np.array_equal(verts[j] - verts[j-1], target):
                        cands.append({'tile_name': name, 'iso_idx': i, 'translation': p1 - verts[j], '_level': level}) 
        return cands

    def _save_tiling_state(self, tiling_state, filename=None):
        all_tiles_for_save = []
        for p in self.initial_placements:
            tile_data = {
                'tile_name': p['tile_name'],
                'iso_idx': p['iso_idx'],
                'translation': p['translation'].tolist(),
                'level': p.get('_level', 0)
            }
            all_tiles_for_save.append(tile_data)
        
        for p in tiling_state['placed_tiles']:
            tile_data = {
                'tile_name': p['tile_name'],
                'iso_idx': p['iso_idx'],
                'translation': p['translation'].tolist(),
                'level': p.get('_level', 0)
            }
            all_tiles_for_save.append(tile_data)

        output_file = filename if filename else self.output_filename
        with open(output_file, 'w') as f:
            json.dump(all_tiles_for_save, f, indent=4)

    def _update_max_tiling(self, current_tiling_state, indent_str=''):
        current_total_tiles = len(current_tiling_state['placed_tiles']) + len(self.initial_placements)
        if current_total_tiles > self.max_tiles_found_so_far:
            self.max_tiles_found_so_far = current_total_tiles
            self.largest_tiling_state = {
                'edge_registry': current_tiling_state['edge_registry'].copy(),
                'global_stripe_occupancies': current_tiling_state['global_stripe_occupancies'].copy(),
                'initial_mod_3_d_residues': current_tiling_state['initial_mod_3_d_residues'].copy(),
                'occupancies': current_tiling_state['occupancies'].copy(),
                'placed_tiles': copy.deepcopy(current_tiling_state['placed_tiles'])
            }
            self._save_tiling_state(self.largest_tiling_state)
            if self.live_visualizer:
                combined_placements = self.initial_placements + self.largest_tiling_state['placed_tiles']
                self.live_visualizer.update_background_tiling(combined_placements, self.prototiles)
                print(f"{indent_str} Updating live background with {self.max_tiles_found_so_far} tiles")

    def _search(self, tiling_state, last_choice_edge, level=0, history_is_last_choice=()):
        indent_parts = []
        for i in range(level):
            if i < len(history_is_last_choice) and history_is_last_choice[i]:
                indent_parts.append(' ')
            else:
                indent_parts.append('│')
        indent_str = "".join(indent_parts)
        
        try:
            forced_moves_count = []

            while True:
                if self.live_visualizer and self.live_visualizer.quit_flag:
                    print(f"QUIT detected! Current tiles: {len(tiling_state['placed_tiles']) + len(self.initial_placements)}")
                    return False

                total_tiles_count = len(tiling_state['placed_tiles']) + len(self.initial_placements)
                if total_tiles_count >= self.max_tiles: return True

                current_frontier_edges_list = [{'p1': p1, 'p2': p2} for (p1, p2), c in tiling_state['edge_registry'].items() if c == 1]
                
                if not current_frontier_edges_list:
                    forced_moves_count.append(0) 
                    if self.live_visualizer:
                        self.live_visualizer.draw(tiler=self, tiling_state_to_draw=tiling_state,
                            title="Tiling Complete for Branch", forced_move_history=forced_moves_count)
                    self._update_max_tiling(tiling_state, indent_str)
                    return True

                found_forced_move_in_this_pass = False
                forced_moves_count_in_this_pass = 0
                
                choice_edges_for_this_scan = [] 

                for edge in list(current_frontier_edges_list): 
                    p1_tuple, p2_tuple = edge['p1'], edge['p2']
                    if tiling_state['edge_registry'].get((p1_tuple, p2_tuple), 0) != 1:
                        continue
                    placements = [tile for tile in self._get_candidate_placements(edge, level) if self._is_placement_valid(tile, tiling_state)]
                    if len(placements) == 0:
                        forced_moves_count.append(0) 
                        if self.live_visualizer:
                            self.live_visualizer.draw(tiler=self, tiling_state_to_draw=tiling_state,
                                title="Dead End on Frontier (No Candidates)", forced_move_history=forced_moves_count)
                        self._update_max_tiling(tiling_state, indent_str)
                        return False
                    elif len(placements) == 1:
                        placement = placements[0]
                        new_state_after_forced_move = self._apply_placement_to_state(placement, tiling_state)
                        if new_state_after_forced_move is None:
                            print('FATAL ERROR: A conflicting forced move during immediate application.')
                            self._update_max_tiling(tiling_state, indent_str)
                            return False
                        tiling_state = new_state_after_forced_move
                        forced_moves_count_in_this_pass += 1
                        found_forced_move_in_this_pass = True
                    else:
                        choice_edges_for_this_scan.append({'edge': edge, 'placements': placements})
                
                forced_moves_count.append(forced_moves_count_in_this_pass)
                
                if self.live_visualizer:
                    self.live_visualizer.draw(tiler=self, tiling_state_to_draw=tiling_state,
                        title=f"Applied {forced_moves_count_in_this_pass} Forced Moves",
                        forced_move_history=forced_moves_count)

                if not found_forced_move_in_this_pass:
                    break 

            if not choice_edges_for_this_scan: 
                forced_moves_count.append(0) 
                if self.live_visualizer:
                    self.live_visualizer.draw(tiler=self, tiling_state_to_draw=tiling_state,
                        title="Tiling Complete (No Choices Left)", forced_move_history=forced_moves_count)
                self._update_max_tiling(tiling_state, indent_str)
                return True 

            current_choice_info = choice_edges_for_this_scan[0] 
            num_choices = len(current_choice_info['placements'])
            N = len(tiling_state['placed_tiles']) + len(self.initial_placements)

            for i, placement in enumerate(current_choice_info['placements']):
                is_current_choice_last = (i == num_choices - 1)
                if is_current_choice_last:
                    print(f"{indent_str}└[{N}] branch {i+1}/{num_choices}")
                else:
                    print(f"{indent_str}├[{N}] branch {i+1}/{num_choices}")

                if not self._is_placement_valid(placement, tiling_state):
                    continue

                next_tiling_state = self._apply_placement_to_state(placement, tiling_state)
                if next_tiling_state is None:
                    print(f"FATAL ERROR during choice branch placement. Backtracking.")
                    self._update_max_tiling(tiling_state, indent_str)
                    continue 

                if self.live_visualizer:
                    self.live_visualizer.draw(tiler=self, tiling_state_to_draw=next_tiling_state,
                        title=f"Apply choice {i+1}/{num_choices}", forced_move_history=forced_moves_count)
                    if self.live_visualizer.quit_flag:
                        print(f"QUIT detected during choice branch. Current tiles: {len(next_tiling_state['placed_tiles']) + len(self.initial_placements)}")
                        return False

                next_history = history_is_last_choice + (is_current_choice_last,)

                if self._search(next_tiling_state, last_choice_edge=current_choice_info['edge'], level=level + 1, history_is_last_choice=next_history):
                    return True 
            return False 
        finally:
            pass

    def tile(self):
        print(f"\n--- Tiling with '{','.join(self.tiling_prototiles.keys())}' on A2 Geometry (Mode: {self.tiling_mode}) ---")

        current_tiling_state = {
            'edge_registry': defaultdict(int),
            'global_stripe_occupancies': {},
            'initial_mod_3_d_residues': {},
            'occupancies': defaultdict(int),
            'placed_tiles': []
        }

        self.largest_tiling_state = None
        self.max_tiles_found_so_far = 0
        self.initial_placements = []


        for i, p_data in enumerate(self.config['initial_placements']):
            p_data['_level'] = 0
            p_data['translation'] = np.array(p_data['translation'], dtype=int)
            print(f"Attempting initial placement #{i+1}: {p_data['tile_name']} (iso_idx: {p_data['iso_idx']}, translation: {p_data['translation'].tolist()}, level: {p_data['_level']})")


            updated_state = self._apply_placement_to_state(p_data, current_tiling_state, is_initial=True)
            if updated_state is None:
                print(f"Tiling aborted due to fatal error in initial placement #{i+1}.")
                return False 

            current_tiling_state = updated_state

            if self.live_visualizer:
                self.live_visualizer.draw(tiler=self, tiling_state_to_draw=current_tiling_state,
                    title=f"Initial Placement #{i+1} Applied", forced_move_history=[])
                if self.live_visualizer.quit_flag:
                    print(f"QUIT detected during initial placements. Current tiles: {len(current_tiling_state['placed_tiles']) + len(self.initial_placements)}")
                    return False

        self.global_center_emb = np.mean([p['translation'] for p in self.initial_placements], axis=0) if self.initial_placements else np.array([0,0,0])
        if self.live_visualizer:
            self.live_visualizer.draw(tiler=self, tiling_state_to_draw=current_tiling_state,
                title="Initial State Ready", forced_move_history=[])
            if self.live_visualizer.quit_flag:
                print(f"QUIT detected before search. Current tiles: {len(current_tiling_state['placed_tiles']) + len(self.initial_placements)}")
                return False

        print("\n--- Starting Search ---"); self.start_time = time.time()
        final_tiling_state_result = self._search(current_tiling_state, last_choice_edge=None, level=0)
        
        if self.live_visualizer and self.live_visualizer.quit_flag:
            print(f"Tiling search was quit by user. Largest tiling state found: {self.max_tiles_found_so_far} tiles.")
            return False

        success = final_tiling_state_result is not None and final_tiling_state_result is not False 

        print(f"\n--- Search Complete in {time.time() - self.start_time:.2f}s ---")
        
        if success and isinstance(final_tiling_state_result, dict):
            final_tiles_count = len(final_tiling_state_result['placed_tiles']) + len(self.initial_placements)
            print(f"Result: SUCCESS | Total Tiles: {final_tiles_count}")
            if final_tiles_count > self.last_saved_tile_count:
                self._save_tiling_state(final_tiling_state_result)
            if self.live_visualizer:
                self.live_visualizer.draw(tiler=self, tiling_state_to_draw=final_tiling_state_result,
                    title="Final Tiling - SUCCESS", forced_move_history=[])
        else:
            final_tiles_count = len(self.largest_tiling_state['placed_tiles']) + len(self.initial_placements)
            print(f"Result: FAILURE | Reported Tiles: {final_tiles_count} (Largest tiling state found)")
            if final_tiles_count > self.last_saved_tile_count:
                self._save_tiling_state(self.largest_tiling_state) 
            if self.live_visualizer:
                self.live_visualizer.draw(tiler=self, tiling_state_to_draw=self.largest_tiling_state,
                    title="Final State - FAILURE (Largest Tiling)", forced_move_history=[])

        return success


class StaticVisualizer:
    def __init__(self, background_tiling_file=None):
        self.fig = plt.figure(figsize=(12, 10))
        gs = self.fig.add_gridspec(10, 8)
        self.ax_main = self.fig.add_subplot(gs[:, 0:7])
        self.ax_main.set_aspect('equal', adjustable='box')
        plt.ion()
        self.fig.canvas.manager.set_window_title("Heesch Tiler")
        self.tiler = None
        self.color_map = {0:'magenta', 1:'lime', 2:'yellow'}
        
        self.drawn_dynamic_tiles = {} 
        
        self.permanent_background_face_artists = [] 
        self.permanent_background_stripe_artists = []
        self._background_drawn_once = False 

        self.temporary_artists = []
        self.palette_xlim, self.palette_ylim = None, None
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.dominant_color = None
        self.decision_tile_count = 10
        self.majority_threshold = 5
        self.last_dominant_color = None
        self.max_tiles_reached = 0
        self.background_tiling_file = background_tiling_file
        
        self.all_active_stripe_artists_info = [] 
        
        self.ax_radio_stripes = self.fig.add_subplot(gs[2:7, 7])
        radio_labels = ('Blue Stripes', 'Red Stripes', 'Both Stripes', 'No Stripes')
        self.radio_buttons = RadioButtons(self.ax_radio_stripes, radio_labels)
        self.radio_buttons.on_clicked(self._on_radio_select)
        self.current_stripe_display_mode = 'No Stripes' 
        self.radio_buttons.set_active(3) 
        self.ax_radio_stripes.set_facecolor('lightgrey')
        self.ax_radio_stripes.tick_params(axis='both', which='both', length=0)
        self.ax_radio_stripes.set_xticks([])
        self.ax_radio_stripes.set_yticks([])
        self.ax_radio_stripes.set_title('Stripe Display')

        self.show_background_stripes = False

        self.ax_button_quit = self.fig.add_subplot(gs[0:1, 7])
        self.quit_button = Button(self.ax_button_quit, 'QUIT', color='salmon', hovercolor='red')
        self.quit_button.on_clicked(self._on_quit_button_clicked)
        self.ax_button_quit.set_facecolor('lightgrey')
        self.ax_button_quit.tick_params(axis='both', which='both', length=0)
        self.ax_button_quit.set_xticks([])
        self.ax_button_quit.set_yticks([])

        self.quit_flag = False 

        if self.background_tiling_file:
            print(f"Loading initial static background from: {self.background_tiling_file}")
            self._load_and_draw_background_from_file()

    def _on_quit_button_clicked(self, event):
        self.quit_flag = True
        print("QUIT button clicked. Signaling current run to stop.")

    def update_background_tiling(self, placements_to_be_background, prototiles_dict):
        """
        Updates the permanent background layer with a new "largest tiling state".
        This will clear the *current* permanent background and redraw it.
        This method is called by the tiler when a new largest tiling is found.
        """
        if self.background_tiling_file and self._background_drawn_once:
            return

        for artist in self.permanent_background_face_artists:
            if artist.get_figure(): artist.remove()
        self.permanent_background_face_artists = []
        for artist in self.permanent_background_stripe_artists:
            if artist.get_figure(): artist.remove()
        self.permanent_background_stripe_artists = []

        for p in placements_to_be_background:
            name, i, t = p['tile_name'], p['iso_idx'], p['translation']
            prototile = prototiles_dict.get(name) 
            if prototile:
                temp_p = {'tile_name': name, 'iso_idx': i, 'translation': t} 
                tile_geometry = self._get_tile_geometry(temp_p, prototile=prototile)

                if tile_geometry:
                    patch = self.ax_main.fill(*tile_geometry['cartesian_verts'], facecolor=tile_geometry['face_color'],
                                              edgecolor='black', lw=0.5, alpha=0.3, zorder=0)
                    self.permanent_background_face_artists.extend(patch)
                    
                    if self.show_background_stripes:
                        for stripe in tile_geometry['stripes']:
                            line = self.ax_main.plot([stripe['p1c'][0], stripe['p2c'][0]],
                                                      [stripe['p1c'][1], stripe['p2c'][1]],
                                                      c=stripe['color'], lw=0.5, alpha=0.1, zorder=0.1)
                            self.permanent_background_stripe_artists.extend(line)
            else:
                print(f"Warning: Prototile '{name}' for background not found during drawing.")
        
        self._calculate_palette_limits() 
        if self.palette_xlim and self.palette_ylim:
            self.ax_main.set_xlim(self.palette_xlim)
            self.ax_main.set_ylim(self.palette_ylim)
        
        self.fig.canvas.draw_idle()
        self._background_drawn_once = True

    def _load_and_draw_background_from_file(self):
        """
        Loads background tiling data from a JSON file and draws it permanently.
        This is typically for an initial, pre-computed background.
        """
        if not self.background_tiling_file:
            return

        try:
            with open(self.background_tiling_file, 'r') as f:
                data = json.load(f)
            
            file_placements = []
            file_prototiles_temp = {}
            for p_data in data:
                translation_array = np.array(p_data['translation'], dtype=int)
                file_placements.append({
                    'tile_name': p_data['tile_name'],
                    'iso_idx': p_data['iso_idx'],
                    'translation': translation_array
                })
                tile_name = p_data['tile_name']
                if tile_name not in file_prototiles_temp:
                    if tile_name == "Turtle":
                        file_prototiles_temp[tile_name] = TurtleTile()
                    elif tile_name == "Propeller":
                        file_prototiles_temp[tile_name] = PropellerTile()

            print(f"Successfully loaded {len(file_placements)} tiles for initial background from file.")
            self.update_background_tiling(file_placements, file_prototiles_temp)

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
        
        for artist in self.permanent_background_stripe_artists:
            if self.current_stripe_display_mode == 'Both Stripes':
                artist.set_visible(True)
            elif self.current_stripe_display_mode == 'No Stripes':
                artist.set_visible(False)

    def _calculate_palette_limits(self):
        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')

        if self.tiler and self.tiler.prototiles:
            for key, tile_info in self.drawn_dynamic_tiles.items():
                verts_cart = tile_info['tile_geometry']['cartesian_verts'].T
                min_x, max_x = min(min_x, verts_cart[:, 0].min()), max(max_x, verts_cart[:, 0].max())
                min_y, max_y = min(min_y, verts_cart[:, 1].min()), max(max_y, verts_cart[:, 1].max())

        if self.permanent_background_face_artists:
            for patch in self.permanent_background_face_artists:
                verts_cart = patch.get_path().vertices
                if verts_cart.size > 0:
                    min_x, max_x = min(min_x, verts_cart[:, 0].min()), max(max_x, verts_cart[:, 0].max())
                    min_y, max_y = min(min_y, verts_cart[:, 1].min()), max(max_y, verts_cart[:, 1].max())
        
        if math.isinf(min_x) or math.isinf(max_x) or math.isinf(min_y) or math.isinf(max_y):
            self.palette_xlim, self.palette_ylim = (-5, 5), (-5, 5)
        else:
            center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
            half_size = max(max_x-center_x, center_x-min_x, max_y-center_y, center_y-min_y) * 1.15
            self.palette_xlim, self.palette_ylim = (center_x-half_size, center_x+half_size), (center_y-half_size, center_y+half_size)


    def _get_tile_geometry(self, p, prototile=None):
        name, t, i = p['tile_name'], p['translation'], p['iso_idx']
        global GEOMETRY 

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

    def draw(self, tiler, tiling_state_to_draw, title, analysis=None, forced_move_history=None, failed_candidate=None, highlight_point=None, highlight_stripe=None): # Add tiling_state_to_draw
        self.tiler = tiler 
        # Clear temporary artists only if not quitting (allows last state to persist)
        if not self.quit_flag:
            for artist in self.temporary_artists:
                if artist.get_figure():
                    artist.remove()
            self.temporary_artists = []

        all_placements_for_dynamic_drawing = tiling_state_to_draw['placed_tiles'] + tiler.initial_placements 
        num_tiles = len(all_placements_for_dynamic_drawing)
        if num_tiles > self.max_tiles_reached:
            self.max_tiles_reached = num_tiles

        elapsed = (time.time() - tiler.start_time) if tiler.start_time else 0
        if forced_move_history and isinstance(forced_move_history, list) and forced_move_history:
            forced_moves_str = f"{forced_move_history}"
            self.ax_main.set_title(f"Tiles: {num_tiles} | Max Tiles: {self.max_tiles_reached} | Time: {elapsed:.2f}s | {title} | {forced_moves_str}")
        else:
            self.ax_main.set_title(f"Tiles: {num_tiles} | Max Tiles: {self.max_tiles_reached} | Time: {elapsed:.2f}s | {title}")

        def sync_tiles_for_display(current_placements, drawn_artists_dict, zorder_base_face, opacity=1.0):
            current_placement_keys = set()
            newly_drawn_stripe_artists = []

            for p in current_placements:
                key = (p['tile_name'], p['iso_idx'], tuple(p['translation']), p.get('_level', 0))
                current_placement_keys.add(key)

                if key not in drawn_artists_dict: # Only draw if not already present
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
                    
                    drawn_artists_dict[key] = {
                        'face_patches': face_patch,
                        'stripe_artists_info': stripe_artists_for_this_tile,
                        'tile_geometry': tile_geometry
                    }
                    newly_drawn_stripe_artists.extend(stripe_artists_for_this_tile)
                else:
                    # If already drawn, just include its stripe info for redrawing stripes if visibility changes
                    newly_drawn_stripe_artists.extend(drawn_artists_dict[key]['stripe_artists_info'])
            
            # Remove artists for tiles that are no longer in current_placements
            keys_to_remove = set(drawn_artists_dict.keys()) - current_placement_keys
            for key in keys_to_remove:
                tile_info = drawn_artists_dict.pop(key)
                for patch in tile_info['face_patches']:
                    if patch.get_figure(): patch.remove()
                for stripe_info in tile_info['stripe_artists_info']:
                    if stripe_info['artist'].get_figure(): stripe_info['artist'].remove()
            
            return newly_drawn_stripe_artists 

        # Sync the dynamically placed tiles
        # The background (loaded from file OR largest_tiling_state) is handled separately now.
        self.all_active_stripe_artists_info = sync_tiles_for_display(all_placements_for_dynamic_drawing, self.drawn_dynamic_tiles, zorder_base_face=1, opacity=0.7)
        
        self._redraw_main_stripes() # Ensure stripe visibility is correct for all active stripes

        # Handle temporary visualizations (failed candidates, highlights)
        if failed_candidate:
            failed_tile_geometry = self._get_tile_geometry(failed_candidate)
            if failed_tile_geometry:
                face_patch = self.ax_main.fill(*failed_tile_geometry['cartesian_verts'], facecolor='red',
                                               edgecolor='red', lw=2.0, alpha=0.5, zorder=2)
                self.temporary_artists.extend(face_patch)

                for stripe in failed_tile_geometry['stripes']:
                    line = self.ax_main.plot([stripe['p1c'][0], stripe['p2c'][0]],
                                              [stripe['p1c'][1], stripe['p2c'][1]],
                                              c='red', lw=3.0, alpha=0.7, zorder=2.1)
                    self.temporary_artists.extend(line)
        
        if highlight_point is not None:
            cart_point = GEOMETRY.a2_to_cartesian(np.array(highlight_point))
            highlight_marker = self.ax_main.plot(cart_point[0], cart_point[1], 'o', color='yellow', markersize=15, markeredgecolor='black', zorder=3)
            self.temporary_artists.extend(highlight_marker)

        if highlight_stripe is not None:
            n, d, color = highlight_stripe
            if failed_candidate: # Only highlight stripe if it's related to a failed candidate
                failed_tile_geometry = self._get_tile_geometry(failed_candidate)
                if failed_tile_geometry:
                    for s in failed_tile_geometry['stripes']:
                        # Need to re-derive params for comparison as highlight_stripe uses (n, d)
                        temp_n, temp_d = GEOMETRY.get_line_parameters(GEOMETRY.cartesian_to_a2(s['p1c']), GEOMETRY.cartesian_to_a2(s['p2c']))
                        
                        if temp_n == n and temp_d == d and s['color'] == color: 
                            line = self.ax_main.plot([s['p1c'][0], s['p2c'][0]],
                                                      [s['p1c'][1], s['p2c'][1]],
                                                      c='yellow', lw=5.0, alpha=1.0, zorder=3)
                            self.temporary_artists.extend(line)
                            # Draw thicker dash for visibility
                            line = self.ax_main.plot([s['p1c'][0], s['p2c'][0]],
                                                      [s['p1c'][1], s['p2c'][1]],
                                                      c='yellow', lw=5.0, alpha=1.0, zorder=3, linestyle='--')
                            self.temporary_artists.extend(line)
                            break

        # Adjust axis limits if any tiles are present
        has_any_tiles = bool(all_placements_for_dynamic_drawing) or bool(self.permanent_background_face_artists)
        if has_any_tiles:
            # Recalculate limits only if they are not yet set or if content has changed significantly
            # For simplicity, we recalculate here, but in a very high-performance scenario,
            # you might only do this on a significant change in tile count or bounds.
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
            {'tile_name':'Propeller', 'iso_idx':9, 'translation':np.array([0,0,0])},
        ],
        'tiling_prototiles': ['Turtle'],
        'tiling_mode': 'plane_filling', 'max_tiles': 10000,
        'description': 'single_propeller',
    }

    config_double = {
        'initial_placements': [
            {'tile_name':'Propeller', 'iso_idx':9, 'translation':np.array([0,0,0])},
            {'tile_name':'Propeller', 'iso_idx':6, 'translation':np.array([6,8,-14])},
        ],
        'tiling_prototiles': ['Turtle'],
        'tiling_mode': 'plane_filling', 'max_tiles': 10000,
        'description': 'double',
    }

    config_dipole = {
        'initial_placements': [
            {'tile_name': 'Turtle', 'iso_idx': 9, 'translation': np.array([2,5,-7])},
            {'tile_name': 'Turtle', 'iso_idx': 6, 'translation': np.array([-2,-5,7])},
        ],
        'tiling_prototiles': ['Turtle'],
        'tiling_mode': 'plane_filling', 'max_tiles': 5000,
        'description': 'dipole',
    }
    
    # for n in [8, 16, 24, 36, 40]: 
    # for n in [2,4,6,8,10,12,14,16]:
    for n in [16]:
        print('Preparing config_cycle with n =', n)

        config_cycle = {
            'initial_placements': [
                {'tile_name':'Turtle', 'iso_idx':9, 'translation':np.array([n-1, 1, -n])},
                {'tile_name':'Turtle', 'iso_idx':7, 'translation':np.array([-n, n-1, 1])},
                {'tile_name':'Turtle', 'iso_idx':11, 'translation':np.array([1, -n, n-1])},
            ],
            'tiling_prototiles': ['Turtle'],
            'tiling_mode': 'plane_filling', 'max_tiles': 10000,
            'description': f'cycle_n={n}',
        }

        config_triple = {
            'initial_placements': [
                {'tile_name':'Propeller', 'iso_idx':9, 'translation':np.array([0,0,0])},
                {'tile_name':'Propeller', 'iso_idx':9, 'translation':np.array([8,8,-16])},
                {'tile_name':'Propeller', 'iso_idx':9, 'translation':np.array([-8,-8,16])},
            ],
            'tiling_prototiles': ['Turtle'],
            'tiling_mode': 'plane_filling', 'max_tiles': 10000,
            'description': 'triple',
        }
        
        config_tripod = {
            'initial_placements': [
                {'tile_name':'Propeller', 'iso_idx':6, 'translation':np.array([n, 0, -n])},
                {'tile_name':'Propeller', 'iso_idx':6, 'translation':np.array([-n, n, 0])},
                {'tile_name':'Propeller', 'iso_idx':6, 'translation':np.array([0, -n, n])},
                # {'tile_name':'Propeller', 'iso_idx':6, 'translation':np.array([-2*n, 0, 2*n])},
                # {'tile_name':'Propeller', 'iso_idx':6, 'translation':np.array([2*n, -2*n, 0])},
                # {'tile_name':'Propeller', 'iso_idx':6, 'translation':np.array([0, 2*n, -2*n])},
            ],
            'tiling_prototiles': ['Turtle'],
            'tiling_mode': 'plane_filling', 'max_tiles': 10000,
            'description': f'tripod_{n}',
        }
        
        SELECTED_CONFIG = config_tripod # Change this to test other configs
        
        background_file = "tiling_2025-10-22_18-04-21_tripod_8.json" # Keep None for no initial background from file
        background_file = None 
        
        vis = StaticVisualizer(background_tiling_file=background_file)
        tiler = HeeschTiler(config=SELECTED_CONFIG, visualizer=vis, 
                            heuristics_to_apply=[
                                stripe_separation_heuristic,
                                stripe_alignment_heuristic,
                            ])
            
        tiler.tile()
        plt.ioff()
        plt.close('all')

