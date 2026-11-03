#!/usr/bin/env python3
r"""Reproduce free-junction theta billiard nets in convex Euclidean polygons.

Search scope: one direct edge and two reflected edges, each with at most B
reflections, no consecutive repeats, no corners or grazing. Outward-direction
cones prune impossible words. Junction directions at the first vertex are
120 degrees apart. The terminal balance gives a 2x2 matrix equation in the
common orientation. Rank two forbids a net; rank one fixes its orientation;
rank zero fixes orientation by the sharp convex dual length bound.

Once directions are fixed, junction positions and all itinerary constraints
are linear. A positive-margin LP supplies a witness. A separate physical
ray tracer validates endpoints, reflection words and both balances along a
nonzero affine continuation. No isolated-root solver is used as evidence.
"""
from pathlib import Path
from dataclasses import dataclass
import argparse
import json
import math
import time
from collections import Counter
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research-data/polygon-families'
J = np.array([[0., -1.], [1., 0.]])


@dataclass
class Word:
    sequence: tuple
    matrix: np.ndarray
    shift: np.ndarray
    walls: tuple


class Polygon:
    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=float)
        edges = np.roll(self.vertices, -1, axis=0)-self.vertices
        self.lengths = np.linalg.norm(edges, axis=1)
        self.normals = np.column_stack([edges[:, 1], -edges[:, 0]])/self.lengths[:, None]
        self.constants = np.sum(self.normals*self.vertices, axis=1)
        self.reflections = [np.eye(2)-2*np.outer(n, n) for n in self.normals]
        self.shifts = [2*c*n for c, n in zip(self.constants, self.normals)]
        self.scale = float(max(np.linalg.norm(a-b) for a in self.vertices for b in self.vertices))

    def words(self, depth):
        empty = Word((), np.eye(2), np.zeros(2), ())
        levels = [[empty]]
        for _ in range(depth):
            nxt = []
            for previous in levels[-1]:
                for index in range(len(self.vertices)):
                    if previous.sequence and index == previous.sequence[-1]:
                        continue
                    normal = previous.matrix@self.normals[index]
                    a = previous.matrix@self.vertices[index]+previous.shift
                    b = previous.matrix@self.vertices[(index+1)%len(self.vertices)]+previous.shift
                    walls = previous.walls+((normal, a, b),)
                    normals = np.array([row[0] for row in walls])
                    angles = np.sort(np.mod(np.arctan2(normals[:, 1], normals[:, 0]), 2*math.pi))
                    gaps = np.diff(np.r_[angles, angles[0]+2*math.pi])
                    if max(gaps) <= math.pi+1e-10:
                        continue
                    matrix = previous.matrix@self.reflections[index]
                    if np.linalg.norm(matrix-np.eye(2)) < 1e-12:
                        matrix = np.eye(2)
                    shift = previous.matrix@self.shifts[index]+previous.shift
                    nxt.append(Word(previous.sequence+(index,), matrix, shift, walls))
            levels.append(nxt)
        return levels

    def ray_trace(self, start, direction, length):
        point, direction = np.asarray(start).copy(), np.asarray(direction).copy()
        remaining = float(length)
        path = [point.copy()]
        word = []
        clearance = self.scale
        for _ in range(60):
            denominator = self.normals@direction
            distance = np.full(len(self.vertices), np.inf)
            exiting = denominator > 1e-10
            distance[exiting] = (self.constants[exiting]-self.normals[exiting]@point)/denominator[exiting]
            distance[distance < self.scale*1e-10] = np.inf
            wall = int(np.argmin(distance))
            travel = distance[wall]
            if remaining < travel-self.scale*1e-9:
                endpoint = point+remaining*direction
                path.append(endpoint)
                return endpoint, direction, tuple(word), np.array(path), clearance
            if not np.isfinite(travel):
                return None
            point = point+travel*direction
            remaining -= travel
            path.append(point.copy())
            word.append(wall)
            clearance = min(clearance, np.linalg.norm(point-self.vertices[wall]),
                            np.linalg.norm(point-self.vertices[(wall+1)%len(self.vertices)]))
            direction = direction-2*np.dot(direction, self.normals[wall])*self.normals[wall]
        return None


def directions_for(words, handedness):
    angles = handedness*2*math.pi*np.arange(3)/3
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    balance_matrix = sum(w.matrix.T@np.column_stack([t, J@t]) for w, t in zip(words, base))
    _, singular, vt = np.linalg.svd(balance_matrix)
    if singular[-1] > 1e-9:
        return None, 'rank_two'
    if singular[0] < 1e-9:
        dual = np.array([sum(np.dot(t, w.shift) for w, t in zip(words, base)),
                         sum(np.dot(J@t, w.shift) for w, t in zip(words, base))])
        if np.linalg.norm(dual) < 1e-10:
            return None, 'zero_dual_length'
        orientation = dual/np.linalg.norm(dual)
        kind = 'free_orientation_dual'
    else:
        orientation = vt[-1]
        trial = orientation[0]*base+orientation[1]*(base@J.T)
        dual = sum(np.dot(t, w.shift) for w, t in zip(words, trial))
        if abs(dual) < 1e-10:
            return None, 'zero_dual_length'
        orientation = orientation*(1 if dual > 0 else -1)
        kind = 'fixed_orientation'
    tangents = orientation[0]*base+orientation[1]*(base@J.T)
    if np.linalg.norm(sum(w.matrix.T@t for w, t in zip(words, tangents))) > 1e-8:
        return None, 'balance_failure'
    return tangents, kind


def linear_witness(polygon, words, tangents):
    eq, rhs, inequalities, constants = [], [], [], []
    def positive(row, constant):
        norm = np.linalg.norm(row)
        if norm < 1e-12:
            return constant > polygon.scale*1e-10
        inequalities.append(-row/norm)
        constants.append(constant/norm)
        return True
    for vertex in (0, 1):
        for normal, constant in zip(polygon.normals, polygon.constants):
            row = np.zeros(4); row[2*vertex:2*vertex+2] = -normal
            positive(row, constant)
    for w, tangent in zip(words, tangents):
        perpendicular = J@tangent
        row = np.r_[-perpendicular, w.matrix.T@perpendicular]
        eq.append(row)
        rhs.append(-np.dot(perpendicular, w.shift))
        length_row = np.r_[-tangent, w.matrix.T@tangent]
        length_constant = float(np.dot(tangent, w.shift))
        if not positive(length_row, length_constant): return None, 'negative_length'
        last_row, last_constant = np.zeros(4), 0.
        for normal, a, b in w.walls:
            denominator = float(np.dot(normal, tangent))
            if denominator <= 1e-10:
                return None, 'direction_incompatible'
            time_row = np.r_[-normal/denominator, [0., 0.]]
            time_constant = float(np.dot(normal, a)/denominator)
            if not positive(time_row-last_row, time_constant-last_constant):
                return None, 'wrong_hit_order'
            wall_tangent = (b-a)/np.linalg.norm(b-a)
            hit_matrix = np.column_stack([np.eye(2), np.zeros((2, 2))])+np.outer(tangent, time_row)
            hit_constant = tangent*time_constant
            along_row = wall_tangent@hit_matrix
            along_constant = float(np.dot(wall_tangent, hit_constant-a))
            if not positive(along_row, along_constant): return None, 'outside_side'
            if not positive(-along_row, np.linalg.norm(b-a)-along_constant): return None, 'outside_side'
            last_row, last_constant = time_row, time_constant
        if not positive(length_row-last_row, length_constant-last_constant):
            return None, 'endpoint_before_last_hit'
    equality, right = np.array(eq), np.array(rhs)
    a, b = np.array(inequalities), np.array(constants)
    result = linprog([0., 0., 0., 0., -1.], A_ub=np.column_stack([a, np.ones(len(a))]), b_ub=b,
                     A_eq=np.column_stack([equality, np.zeros(len(equality))]), b_eq=right,
                     bounds=[(None, None)]*4+[(0., None)], method='highs')
    if not result.success or result.x[-1] < polygon.scale*1e-6:
        return None, 'lp_no_positive_margin'
    return dict(z=result.x[:4], margin=result.x[-1], eq=equality, rhs=right,
                inequalities=a, constants=b), 'lp_feasible'


def crossings_of(paths,u,v,scale):
    segments=[]
    for edge,path in enumerate(paths):
        for index,(a,b) in enumerate(zip(path,path[1:])):
            segments.append((edge,index,np.array(a),np.array(b)))
    intersections=[];overlaps=0
    def cross(a,b): return a[0]*b[1]-a[1]*b[0]
    for index,(edge,part,a,b) in enumerate(segments):
        for other,otherpart,c,d in segments[:index]:
            if edge==other and abs(part-otherpart)==1: continue
            x,y=b-a,d-c;det=cross(x,y)
            if abs(det)<1e-10*scale*scale:
                if abs(cross(c-a,x))<1e-10*scale*scale:
                    interval=sorted([np.dot(c-a,x)/np.dot(x,x),np.dot(d-a,x)/np.dot(x,x)])
                    if min(1,interval[1])-max(0,interval[0])>1e-8: overlaps+=1
                continue
            first,second=cross(c-a,y)/det,cross(c-a,x)/det
            if -1e-9<=first<=1+1e-9 and -1e-9<=second<=1+1e-9:
                p=a+first*x
                if min(np.linalg.norm(p-u),np.linalg.norm(p-v))<scale*1e-7: continue
                if not any(np.linalg.norm(p-q)<scale*1e-7 for q in intersections): intersections.append(p)
    return [p.tolist() for p in intersections],overlaps


def validate_frame(polygon, words, tangents, z):
    u, v = z[:2], z[2:]
    paths, arrivals, lengths, endpoint_errors = [], [], [], []
    corner_clearance = polygon.scale
    junction_clearance = polygon.scale
    for word, tangent in zip(words, tangents):
        lifted = word.matrix@v+word.shift-u
        length = float(np.dot(tangent, lifted))
        if length <= polygon.scale*1e-8:
            return None
        traced = polygon.ray_trace(u, tangent, length)
        if traced is None: return None
        endpoint, arrival, observed_word, path, clearance = traced
        error = float(np.linalg.norm(endpoint-v))
        if observed_word != word.sequence or error > polygon.scale*2e-7 or clearance < polygon.scale*1e-7:
            return None
        for index, (a, b) in enumerate(zip(path, path[1:])):
            for label, node in [('u', u), ('v', v)]:
                if (label=='u' and index==0) or (label=='v' and index==len(path)-2): continue
                along = np.clip(np.dot(node-a, b-a)/np.dot(b-a, b-a), 0, 1)
                junction_clearance = min(junction_clearance, np.linalg.norm(node-(a+along*(b-a))))
        observed_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        paths.append(path.tolist()); arrivals.append(arrival); lengths.append(observed_length); endpoint_errors.append(error)
        corner_clearance = min(corner_clearance, clearance)
    residual = float(np.linalg.norm(np.sum(arrivals, axis=0)))
    if residual > 1e-7 or junction_clearance < polygon.scale*1e-7: return None
    crossing_points,overlaps=crossings_of(paths,u,v,polygon.scale)
    if overlaps: return None
    return dict(u=u.tolist(), v=v.tolist(), paths=paths, lengths=lengths, total_length=sum(lengths),
                endpoint_error=max(endpoint_errors), terminal_balance_residual=residual,
                corner_clearance=corner_clearance, intermediate_junction_clearance=float(junction_clearance),
                crossing_points=crossing_points,embedded=not bool(crossing_points))


def family_polygon(polygon, witness, kernel):
    if kernel.shape[1] != 2: return None
    matrix = witness['inequalities']@kernel
    right = witness['constants']-witness['inequalities']@witness['z']
    vertices = []
    for i in range(len(right)):
        for j in range(i):
            system = matrix[[i, j]]
            if abs(np.linalg.det(system)) < 1e-10: continue
            point = np.linalg.solve(system, right[[i, j]])
            if np.max(matrix@point-right)<polygon.scale*1e-8:
                vertices.append(point)
    if len(vertices)<3: return None
    vertices = np.unique(np.round(np.array(vertices), 12), axis=0)
    hull = ConvexHull(vertices)
    vertices = vertices[hull.vertices]
    junctions = witness['z'][:2]+vertices@kernel[:2].T
    area = float(hull.volume*abs(np.linalg.det(kernel[:2])))
    p = polygon.vertices
    domain_area = abs(np.dot(p[:, 0], np.roll(p[:, 1], -1))-np.dot(p[:, 1], np.roll(p[:, 0], -1)))/2
    return dict(first_junction_region=junctions.tolist(), area=area, domain_area=domain_area,
                fraction_of_polygon_area=area/domain_area)


def itinerary_path(polygon,word,tangent,z):
    point=z[:2].copy();direction=tangent.copy();path=[point.copy()]
    for wall in word.sequence:
        normal=polygon.normals[wall]
        travel=(polygon.constants[wall]-np.dot(normal,point))/np.dot(normal,direction)
        point=point+travel*direction;path.append(point.copy())
        direction=polygon.reflections[wall]@direction
    path.append(z[2:].copy())
    return np.array(path)


def topology_interval(polygon,words,tangents,z,direction,lower,upper):
    """Remove exact affine event parameters for nonincident junction hits and
    coincident bounces; the remaining open interval contains the LP center."""
    events=[];bounce_points=[]
    for word,tangent in zip(words,tangents):
        path=itinerary_path(polygon,word,tangent,z)
        slope=itinerary_path(polygon,word,tangent,z+direction)-path
        for part,(a,b) in enumerate(zip(path,path[1:])):
            ray=(b-a)/np.linalg.norm(b-a);normal=J@ray
            for label,node,dnode in [('u',z[:2],direction[:2]),('v',z[2:],direction[2:])]:
                if (label=='u' and part==0) or (label=='v' and part==len(path)-2): continue
                constant=np.dot(normal,node-a);derivative=np.dot(normal,dnode-slope[part])
                if abs(derivative)>1e-12:
                    parameter=-constant/derivative
                    if lower<parameter<upper:
                        p=node+parameter*dnode;pa=a+parameter*slope[part];pb=b+parameter*slope[part+1]
                        along=np.dot(p-pa,ray)
                        if -1e-9<=along<=np.linalg.norm(pb-pa)+1e-9: events.append(float(parameter))
                elif abs(constant)<1e-10:
                    for end in (part,part+1):
                        der=np.dot(ray,dnode-slope[end])
                        if abs(der)>1e-12:
                            parameter=-np.dot(ray,node-path[end])/der
                            if lower<parameter<upper: events.append(float(parameter))
        for part,wall in enumerate(word.sequence,1):
            bounce_points.append((wall,path[part],slope[part]))
    for i,(wall,point,slope) in enumerate(bounce_points):
        along=J@polygon.normals[wall]
        for other,otherpoint,otherslope in bounce_points[:i]:
            if wall!=other: continue
            derivative=np.dot(along,slope-otherslope)
            if abs(derivative)>1e-12:
                parameter=-np.dot(along,point-otherpoint)/derivative
                if lower<parameter<upper: events.append(float(parameter))
    for parameter in events:
        if parameter < -1e-9: lower=max(lower,parameter)
        elif parameter > 1e-9: upper=min(upper,parameter)
        else: return None
    return lower,upper,events


def continue_family(polygon, words, tangents, witness):
    z = witness['z']
    _, singular, vt = np.linalg.svd(witness['eq'], full_matrices=True)
    rank = int(np.sum(singular>1e-8))
    kernel = vt[rank:].T
    direction = kernel@np.array([1.]+[.371]*(kernel.shape[1]-1))
    direction /= np.linalg.norm(direction)
    lower, upper = -np.inf, np.inf
    for row, constant in zip(witness['inequalities'], witness['constants']):
        slope = float(np.dot(row, direction))
        gap = float(constant-np.dot(row, z))
        if slope > 1e-12: upper = min(upper, gap/slope)
        elif slope < -1e-12: lower = max(lower, gap/slope)
    if not (lower < 0 < upper and np.isfinite(lower+upper)):
        return None
    refined=topology_interval(polygon,words,tangents,z,direction,lower,upper)
    if refined is None:return None
    lower,upper,events=refined
    parameters = np.linspace(.75*lower, .75*upper, 9)
    frames = [validate_frame(polygon, words, tangents, z+parameter*direction) for parameter in parameters]
    if any(frame is None for frame in frames): return None
    values = [frame['total_length'] for frame in frames]
    if max(values)-min(values) > polygon.scale*1e-7: return None
    lengths = np.array([frame['lengths'] for frame in frames])
    return dict(words=[list(w.sequence) for w in words], tangents=tangents.tolist(),
                center=z.tolist(), family_dimension=4-rank, equality_singular_values=singular.tolist(),
                margin=float(witness['margin']), continuation_direction=direction.tolist(),
                full_parameter_interval=[lower, upper], sampled_parameters=parameters.tolist(), frames=frames,
                excluded_event_parameters=events,
                total_length_spread=max(values)-min(values),
                individual_length_spreads=np.ptp(lengths, axis=0).tolist(),
                maximum_endpoint_error=max(f['endpoint_error'] for f in frames),
                maximum_terminal_balance_residual=max(f['terminal_balance_residual'] for f in frames),
                minimum_corner_clearance=min(f['corner_clearance'] for f in frames))


def find_embedded_family(polygon,words,tangents,witness):
    """Try further positions in the feasible family rather than rejecting a
    whole family from its LP center. This finite sampling is not an embedded
    nonexistence certificate."""
    _,singular,vt=np.linalg.svd(witness['eq'],full_matrices=True)
    rank=int(np.sum(singular>1e-8));kernel=vt[rank:].T
    samples=[]
    if kernel.shape[1]==2:
        region=family_polygon(polygon,witness,kernel)
        if region is None:return None,0
        vertices=np.array(region['first_junction_region'])
        origin=vertices.mean(axis=0)
        # Triangulate the convex position region, then sample positive barycentric coordinates.
        for first,second in zip(vertices,np.roll(vertices,-1,axis=0)):
            for a in np.linspace(.05,.95,10):
                for b in np.linspace(.05,.95,10):
                    u=(1-a)*origin+a*((1-b)*first+b*second)
                    v=np.linalg.lstsq(witness['eq'][:,2:],witness['rhs']-witness['eq'][:,:2]@u,rcond=None)[0]
                    samples.append(np.r_[u,v])
    elif kernel.shape[1]==1:
        direction=kernel[:,0];lower,upper=-np.inf,np.inf
        for row,b in zip(witness['inequalities'],witness['constants']):
            slope=np.dot(row,direction);gap=b-np.dot(row,witness['z'])
            if slope>1e-12:upper=min(upper,gap/slope)
            elif slope< -1e-12:lower=max(lower,gap/slope)
        result=topology_interval(polygon,words,tangents,witness['z'],direction,lower,upper)
        if result:
            breaks=sorted([lower,upper]+result[2])
            samples=[witness['z']+(a+b)/2*direction for a,b in zip(breaks,breaks[1:])]
    checked=0
    for z in samples:
        checked+=1
        frame=validate_frame(polygon,words,tangents,z)
        if frame is None or not frame['embedded']:continue
        alternative=dict(witness,z=z)
        family=continue_family(polygon,words,tangents,alternative)
        if family and all(f['embedded'] for f in family['frames']):return family,checked
    return None,checked


def search(polygon, max_depth=4, max_witnesses=1, embedded_only=False):
    levels = polygon.words(max_depth)
    empty = levels[0][0]
    counters = Counter()
    found = []
    start = time.monotonic()
    for cost in range(2, 2*max_depth+1):
        for depth1 in range(1, max_depth+1):
            depth2 = cost-depth1
            if not depth1<=depth2<=max_depth: continue
            for i, first in enumerate(levels[depth1]):
                for j, second in enumerate(levels[depth2]):
                    if depth1==depth2 and i>=j: continue
                    if np.linalg.norm(first.matrix-second.matrix)+np.linalg.norm(first.shift-second.shift)<1e-10:
                        continue
                    counters['word_pairs'] += 1
                    words = [empty, first, second]
                    for handedness in (-1, 1):
                        tangents, kind = directions_for(words, handedness)
                        counters[kind] += 1
                        if tangents is None: continue
                        if embedded_only:
                            terminal=np.array([-w.matrix.T@t for w,t in zip(words,tangents)])
                            def orientation(t):
                                x,y=t[1]-t[0],t[2]-t[0]
                                return x[0]*y[1]-x[1]*y[0]
                            if orientation(tangents)*orientation(terminal)>0:
                                counters['nonplanar_rotation_system']+=1
                                continue
                        witness, state = linear_witness(polygon, words, tangents)
                        counters[state] += 1
                        if witness is None: continue
                        family = continue_family(polygon, words, tangents, witness)
                        if family is None:
                            counters['independent_trace_rejected'] += 1
                            continue
                        if embedded_only and not all(f['embedded'] for f in family['frames']):
                            alternative,checked=find_embedded_family(polygon,words,tangents,witness)
                            counters['additional_family_positions_checked']+=checked
                            if alternative is None:
                                counters['crossing_continuation_rejected']+=1
                                continue
                            family=alternative
                        family['orientation_case'] = kind
                        _, singular_eq, vt_eq = np.linalg.svd(witness['eq'], full_matrices=True)
                        rank_eq = int(np.sum(singular_eq>1e-8))
                        family['admissible_region'] = family_polygon(polygon, witness, vt_eq[rank_eq:].T)
                        family['reflections_total'] = cost
                        found.append(family)
                        print(json.dumps({'found': len(found), 'cost': cost, 'words': family['words'],
                                          'dimension': family['family_dimension'], 'length': family['frames'][0]['total_length'],
                                          'elapsed': time.monotonic()-start}), flush=True)
                        if len(found)>=max_witnesses:
                            return dict(max_reflections_per_non_direct_edge=max_depth,
                                        word_counts_by_depth=[len(level) for level in levels],
                                        counters=dict(counters), witnesses=found, elapsed=time.monotonic()-start)
    return dict(max_reflections_per_non_direct_edge=max_depth,
                word_counts_by_depth=[len(level) for level in levels], counters=dict(counters),
                witnesses=found, elapsed=time.monotonic()-start)


def polygons():
    angle = math.pi/2+2*math.pi*np.arange(5)/5
    pentagon = np.column_stack([np.cos(angle), np.sin(angle)])
    perturbed = pentagon.copy(); perturbed[0] += [.05, .03]
    return {
        'square': [[0, 0], [1, 0], [1, 1], [0, 1]],
        'rectangle': [[0, 0], [1, 0], [1, 3*math.sqrt(3)/4], [0, 3*math.sqrt(3)/4]],
        'rhombus_60': [[0, 0], [1, 0], [1.5, math.sqrt(3)/2], [.5, math.sqrt(3)/2]],
        'sheared_parallelogram': [[0, 0], [1, 0], [1.5, 1], [.5, 1]],
        'regular_pentagon': pentagon.tolist(),
        'house_pentagon': [[0, 0], [2, 0], [2, 1], [1, 2], [0, 1]],
        'perturbed_pentagon': perturbed.tolist(),
    }


def sweep(depth):
    rows = []
    for shear in [0., .25, .5, .75, 1.]:
        for height in [.5, 1., 1.5]:
            vertices = [[0, 0], [1, 0], [1+shear, height], [shear, height]]
            result = search(Polygon(vertices), depth)
            rows.append(dict(shear=shear, height=height, result=result))
            print(json.dumps({'sweep': [shear, height], 'found': bool(result['witnesses'])}), flush=True)
    return rows


def perturbation_comparison():
    sequences = [(), (0, 2, 0, 3), (4, 2, 0, 2)]
    rows = []
    for name in ['regular_pentagon', 'perturbed_pentagon']:
        polygon = Polygon(polygons()[name])
        by_word = {w.sequence: w for level in polygon.words(4) for w in level}
        if any(seq not in by_word for seq in sequences):
            rows.append(dict(shape=name, word_direction_cone_impossible=True)); continue
        words = [by_word[seq] for seq in sequences]
        for handedness in (-1, 1):
            angles = handedness*2*math.pi*np.arange(3)/3
            base = np.column_stack([np.cos(angles), np.sin(angles)])
            matrix = sum(w.matrix.T@np.column_stack([t, J@t]) for w, t in zip(words, base))
            rows.append(dict(shape=name, handedness=handedness, angular_balance_matrix=matrix.tolist(),
                             singular_values=np.linalg.svd(matrix, compute_uv=False).tolist(),
                             determinant=float(np.linalg.det(matrix))))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', default='all', choices=['all']+list(polygons()))
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--witnesses', type=int, default=1)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--embedded', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        result = dict(parallelograms=sweep(args.depth), same_words_under_pentagon_perturbation=perturbation_comparison(),
                      scope='Finite-depth direct-plus-two-reflected-edge search. Absence of a witness is not a global nonexistence theorem.')
        (OUT/'sweep.json').write_text(json.dumps(result, indent=2)+'\n')
        return
    for name, vertices in polygons().items():
        if args.shape != 'all' and name != args.shape: continue
        print(json.dumps({'starting': name, 'depth': args.depth}), flush=True)
        result = search(Polygon(vertices), args.depth, args.witnesses,args.embedded)
        result['name'] = name; result['vertices'] = vertices
        result['scope'] = 'One direct edge and two reflected edges; freely moving 120-degree junctions; fixed polygon; immersed paths; finite itinerary search.'
        result['embedded_only']=args.embedded
        (OUT/(name+('_embedded' if args.embedded else '')+'.json')).write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'completed': name, 'witnesses': len(result['witnesses']),
                          'counters': result['counters'], 'seconds': result['elapsed']}), flush=True)


if __name__ == '__main__':
    main()
