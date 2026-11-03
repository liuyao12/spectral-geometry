#!/usr/bin/env python3
"""Check the clean-family density and integrate the rhombus unfolded kernels.

The integrals here use plane Green kernels with specified unfolded isometries.
They are not the full Dirichlet/Neumann Green kernel of the rhombus.
"""
from pathlib import Path
import json
import math
import numpy as np
from scipy.special import roots_hermite, k0e
from polygon_theta_families import Polygon, polygons

ROOT = Path(__file__).resolve().parents[1]
J = np.array([[0., -1.], [1., 0.]])


def branch_data():
    polygon = Polygon(polygons()['rhombus_60'])
    words = {w.sequence: w for level in polygon.words(2) for w in level}
    return [words[()], words[(0, 3)], words[(1, 2)]]


def geometry(words, u):
    reflection = np.array([[.5, math.sqrt(3)/2], [math.sqrt(3)/2, -.5]])
    v = reflection@u
    displacements = np.array([w.matrix@v+w.shift-u for w in words])
    lengths = np.linalg.norm(displacements, axis=1)
    tangents = displacements/lengths[:, None]
    rows = np.array([np.r_[-J@t, w.matrix.T@(J@t)] for w, t in zip(words, tangents)])
    hessian = rows.T@np.diag(1/lengths)@rows
    eigenvalues = np.linalg.eigvalsh(hessian)
    normal_det = float(np.prod(eigenvalues[-2:]))
    density = 2/math.sqrt(float(np.prod(lengths))*normal_det)
    assert np.linalg.norm(hessian@np.vstack([np.eye(2), reflection])) < 1e-10
    return v, lengths, tangents, hessian, normal_det, density


def scaled_plane_integral(words, u, kappa, order=80):
    """Independent two-dimensional Gauss-Hermite quadrature over the other node.

    Returns kappa^(5/2) exp(3 kappa) times the exact plane-kernel integral.
    The Gaussian is an importance weight; the full Bessel factors are retained.
    """
    v, lengths, tangents, hessian, _, _ = geometry(words, u)
    normal = hessian[2:, 2:]
    eig, basis = np.linalg.eigh(normal)
    inverse_sqrt = (basis/np.sqrt(eig))@basis.T
    nodes, weights = roots_hermite(order)
    xx, yy = np.meshgrid(nodes, nodes, indexing='ij')
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    positions = v+math.sqrt(2/kappa)*grid@inverse_sqrt.T
    distances = np.array([np.linalg.norm(positions@w.matrix.T+w.shift-u, axis=1) for w in words])
    log_integrand = (np.log(k0e(kappa*distances)).sum(axis=0)-3*math.log(2*math.pi)
                     -kappa*(distances.sum(axis=0)-3)+np.sum(grid*grid, axis=1)
                     +1.5*math.log(kappa))
    quadrature_weights = np.outer(weights, weights).ravel()
    return float(2/math.sqrt(np.linalg.det(normal))*np.dot(quadrature_weights, np.exp(log_integrand)))


def main():
    words = branch_data()
    rng = np.random.default_rng(20261102)
    errors = dict(transverse_determinant=0., position_density=0., length=0.)
    for _ in range(500):
        y = rng.uniform(.01, math.sqrt(3)/2-.01)
        lo, hi = math.sqrt(3)*y, 1+y/math.sqrt(3)
        x = rng.uniform(lo+.01*(hi-lo), hi-.01*(hi-lo))
        _, lengths, _, _, determinant, density = geometry(words, np.array([x, y]))
        expected = 3*sum(lengths)/np.prod(lengths)
        errors['transverse_determinant'] = max(errors['transverse_determinant'], abs(determinant/expected-1))
        errors['position_density'] = max(errors['position_density'], abs(density-2/3))
        errors['length'] = max(errors['length'], abs(sum(lengths)-3))
    assert max(errors.values()) < 1e-10
    point_coefficient = 1/(4*math.sqrt(18*math.pi))
    rows = []
    for u in [[.75, .1], [.75, math.sqrt(3)/8], [.8, .2]]:
        values = []
        for kappa in [20, 40, 80, 160, 320]:
            value = scaled_plane_integral(words, np.array(u), kappa)
            refined = scaled_plane_integral(words, np.array(u), kappa, order=112)
            values.append(dict(kappa=kappa, scaled_integral=refined,
                               ratio_to_leading_coefficient=refined/point_coefficient,
                               quadrature_relative_change=abs(refined-value)/refined))
        assert abs(values[-1]['ratio_to_leading_coefficient']-1) < .001
        assert values[-1]['quadrature_relative_change'] < 1e-8
        rows.append(dict(first_junction=u, values=values))
    result = dict(scope='Plane Green kernels for one labeled, oriented unfolded rhombus itinerary triple. Not a complete billiard spectral trace.',
                  sampled_geometry_points=500, maximum_errors=errors,
                  position_density=2/3, first_junction_area=math.sqrt(3)/4,
                  intrinsic_family_weight=math.sqrt(3)/6,
                  fixed_first_junction_leading_coefficient=point_coefficient,
                  whole_family_leading_coefficient=1/(16*math.sqrt(6*math.pi)),
                  exact_net_term='K0(3*kappa)/(16*pi*kappa^2); subtract positive edge-collapse corrections.',
                  quadrature=rows)
    for target in ['research-data/theta-family-weights.json', 'docs/data/theta-family-weights.json']:
        (ROOT/target).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
