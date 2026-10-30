#!/usr/bin/env python3
r"""Geometry-first tests; no many-body Hamiltonian is selected.

An immersed theta net in a rectangle has unfolded edge vectors
    r, r-(2L,2H), r-(2L,-2H).
Its unweighted length is convex in r. For H in (sqrt(3)L/2,sqrt(3)L),
the critical difference is r=(2L-2H/sqrt(3),0). All junction angles are
120 degrees; the two common translations are genuine zero modes.

Diagnostic, NOT a stationary-net trace formula:
    I(kappa) = integral G_kappa(u,v)^3 du dv
             = sum_{i,j,k} C_ijk^2 / prod_a(lambda_a+kappa^2),
    C_ijk = integral phi_i phi_j phi_k.
The identity follows first with heat-regularized resolvents, then by
dominated convergence spatially and monotone convergence spectrally.
In two dimensions the logarithmic diagonal singularity is integrable.
Independent reflected-image cubature is compared with spectral cutoffs.
Cubature uncertainty is empirical, not a rigorous error certificate.

For each positive image length ell,
    integral_0^inf exp(-ell^2/(4t)-kappa^2*t)/(4*pi*t) dt
       = K_0(kappa*ell)/(2*pi).
Minimizing the exponent over every edge time gives kappa*total_length.
This yields ordinary length stationarity only at the exponential/action
level, not an exact replacement of vertex integrals by critical nets.
"""
from pathlib import Path
import json
import math
import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import k0, k0e
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / 'research-data/geodesic-first'


def length_derivatives(r, width, height):
    vectors = np.asarray(r) - np.array([[0., 0.], [2*width, 2*height], [2*width, -2*height]])
    lengths = np.linalg.norm(vectors, axis=1)
    tangents = vectors / lengths[:, None]
    gradient = tangents.sum(axis=0)
    hessian = sum((np.eye(2) - np.outer(t, t))/ell for t, ell in zip(tangents, lengths))
    return float(lengths.sum()), gradient, hessian, tangents, lengths


def geometry_checks():
    rows = []
    for width, ratio in [(1., 1.), (1., 3*math.sqrt(3)/4), (1., 1.5), (1.7, 1.2)]:
        height = width*ratio
        exact_r = np.array([2*width-2*height/math.sqrt(3), 0.])
        value, gradient, hessian, tangents, lengths = length_derivatives(exact_r, width, height)
        errors = []
        for start in [[.2*width, -.13*height], [.8*width, .17*height], [.5*width, 0.]]:
            fit = minimize(lambda r: length_derivatives(r, width, height)[0], start,
                           jac=lambda r: length_derivatives(r, width, height)[1],
                           hess=lambda r: length_derivatives(r, width, height)[2],
                           method='trust-exact', options={'gtol': 1e-11})
            errors.append(float(np.linalg.norm(fit.x-exact_r)))
        angles = [math.degrees(math.acos(np.clip(np.dot(tangents[i], tangents[j]), -1., 1.)))
                  for i, j in [(0, 1), (0, 2), (1, 2)]]
        full_hessian = np.block([[hessian, -hessian], [-hessian, hessian]])
        assert np.linalg.norm(gradient) < 1e-12
        assert max(abs(a-120) for a in angles) < 1e-10
        assert max(errors) < 2e-7
        rows.append(dict(width=width, height=height, vertex_difference=exact_r.tolist(),
                         total_length=value, edge_lengths=lengths.tolist(), angles_degrees=angles,
                         force_residual=float(np.linalg.norm(gradient)), optimizer_errors=errors,
                         full_hessian_eigenvalues=np.linalg.eigvalsh(full_hessian).tolist(),
                         common_translation_zero_modes=2))
    return rows


def triple_sine_squared(cutoff, side):
    modes = np.arange(1, cutoff+1)
    a, b, c = modes[:, None, None], modes[None, :, None], modes[None, None, :]
    def sine_integral(n):
        return np.divide(1-(-1.)**n, n, out=np.zeros(np.broadcast_shapes(n.shape, ())), where=n != 0)
    integral = (sine_integral(a+b-c)+sine_integral(a-b+c)+sine_integral(-a+b+c)-sine_integral(a+b+c))/4
    return ((2/side)**1.5 * side/math.pi * integral)**2


def spectral_theta(cutoff, width, height, kappa):
    modes = np.arange(1, cutoff+1)
    propagator = 1/((math.pi*modes[:, None]/width)**2 + (math.pi*modes[None, :]/height)**2 + kappa*kappa)
    return float(np.einsum('abc,def,ad,be,cf->', triple_sine_squared(cutoff, width),
                          triple_sine_squared(cutoff, height), propagator, propagator, propagator,
                          optimize=True))


def image_green(u, v, width, height, kappa, winding, boundary='dirichlet'):
    if boundary not in ('dirichlet', 'neumann'):
        raise ValueError('boundary must be dirichlet or neumann')
    value = np.zeros(len(u))
    for sx in (-1, 1):
        for sy in (-1, 1):
            for nx in range(-winding, winding+1):
                dx = u[:, 0]-sx*v[:, 0]+2*nx*width
                for ny in range(-winding, winding+1):
                    dy = u[:, 1]-sy*v[:, 1]+2*ny*height
                    character = sx*sy if boundary == 'dirichlet' else 1
                    value += character*k0(kappa*np.hypot(dx, dy))/(2*math.pi)
    return value


def image_theta(seed, power, winding, width, height, kappa, boundary='dirichlet'):
    points = qmc.Sobol(4, scramble=True, seed=seed).random_base2(power)
    t, angle = points[:, 0], 2*math.pi*points[:, 1]
    cs, sn = np.cos(angle), np.sin(angle)
    radial_max = np.minimum(width/np.abs(cs), height/np.abs(sn))
    radial = radial_max*t*t
    r = radial[:, None]*np.column_stack([cs, sn])
    overlap = np.array([width, height])-np.abs(r)
    u = np.maximum(0., -r)+points[:, 2:]*overlap
    v = u+r
    jacobian = 4*math.pi*radial_max**2*t**3*np.prod(overlap, axis=1)
    g = image_green(u, v, width, height, kappa, winding, boundary)
    return float(np.mean(jacobian*g**3))


def schwinger_checks():
    rows = []
    for ell, kappa in [(.5, 1.), (3., 1.), (.5, 3.), (3., 3.)]:
        # Logarithmic time coordinates avoid missing a narrow small-time peak.
        integrand = lambda s: math.exp(-ell*ell/(4*math.exp(s))-kappa*kappa*math.exp(s))/(4*math.pi)
        integrated, estimated_error = quad(integrand, -35., 25., epsabs=1e-14, epsrel=1e-12)
        bessel = float(k0(kappa*ell)/(2*math.pi))
        tstar = ell/(2*kappa)
        assert abs(integrated-bessel) < 1e-12
        rows.append(dict(length=ell, kappa=kappa, integral=integrated, bessel=bessel,
                         absolute_error=abs(integrated-bessel), quadrature_error_estimate=estimated_error,
                         stationary_time=tstar, minimized_exponent=ell*ell/(4*tstar)+kappa*kappa*tstar))
    return rows


def stationary_sector_checks():
    """Direct cubature of one specified labeled image sector, rescaled at its net.

    Translation integration gives overlap (1-|r_x|)(H-|r_y|).
    Hessian in relative coordinates at r=(1/2,0) is diag(1/2,13/6).
    Bessel asymptotics plus two-dimensional Laplace integration give
    A(kappa) ~ 3/(32 sqrt(13 pi)) kappa^(-5/2) exp(-13 kappa/2).
    No claim is made about cancellation with other signed image sectors.
    """
    height = 3*math.sqrt(3)/4
    predicted = 3/(32*math.sqrt(13*math.pi))
    rows = []
    for kappa in [10., 30., 100., 300., 1000.]:
        root = math.sqrt(kappa)
        def integrand(zy, zx):
            rx, ry = .5+zx/root, zy/root
            lengths = np.array([math.hypot(rx, ry), math.hypot(rx-2, ry-2*height),
                                math.hypot(rx-2, ry+2*height)])
            amplitude = np.prod(k0e(kappa*lengths)/(2*math.pi))
            return 2*kappa**1.5*(1-abs(rx))*(height-ry)*amplitude*math.exp(-kappa*(sum(lengths)-6.5))
        lo, hi = max(-12., -1.5*root), min(12., .5*root)
        ymax = min(12., height*root)
        def inner(zx):
            return quad(lambda zy: integrand(zy, zx), 0., ymax, epsabs=2e-12, epsrel=1e-9)[0]
        split = -.5*root
        pieces = [lo]+([split] if lo < split < hi else [])+[hi]
        value = sum(quad(inner, a, b, epsabs=2e-11, epsrel=1e-8)[0]
                    for a, b in zip(pieces, pieces[1:]))
        rows.append(dict(kappa=kappa, rescaled_sector_integral=value,
                         predicted_limit=predicted, ratio_to_predicted_limit=value/predicted))
    assert abs(rows[-1]['ratio_to_predicted_limit']-1) < .02
    return rows


def fold(value, side):
    reduced = np.mod(value, 2*side)
    return np.minimum(reduced, 2*side-reduced)


def draw_net():
    width, height = 1., 3*math.sqrt(3)/4
    u, v = np.array([.25, height/2]), np.array([.75, height/2])
    fig, ax = plt.subplots(figsize=(5.2, 6.1), layout='constrained')
    ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], color='#263d48')
    for index, target in enumerate([v, v-np.array([2*width, 2*height]), v-np.array([2*width, -2*height])]):
        path = u+np.linspace(0, 1, 1001)[:, None]*(target-u)
        ax.plot(fold(path[:, 0], width), fold(path[:, 1], height), lw=2,
                color=['#bf6536', '#007c83', '#6874a5'][index],
                label=[r'$\ell_1=1/2$', r'$\ell_2=3$', r'$\ell_3=3$'][index])
        # Check all unfolded boundary hits avoid corners.
        ts = []
        for coordinate, side in enumerate([width, height]):
            for wall in range(-4, 5):
                if abs(target[coordinate]-u[coordinate]) > 1e-12:
                    parameter = (wall*side-u[coordinate])/(target[coordinate]-u[coordinate])
                    if 1e-12 < parameter < 1-1e-12:
                        ts.append((parameter, coordinate))
        for parameter, coordinate in ts:
            other = 1-coordinate
            hit = u+parameter*(target-u)
            scaled = hit[other]/[width, height][other]
            assert abs(scaled-round(scaled)) > 1e-8
    ax.scatter([u[0], v[0]], [u[1], v[1]], color='#172f37', s=40, zorder=5)
    ax.text(u[0], u[1]+.045, r'$u$', ha='center')
    ax.text(v[0], v[1]+.045, r'$v$', ha='center')
    ax.set_aspect('equal')
    ax.set_title(r'Unweighted stationary theta net: $L=13/2$')
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3)
    fig.savefig(OUT/'theta-net.png', dpi=180)
    plt.close(fig)


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    geometry = geometry_checks()
    schwinger = schwinger_checks()
    print(json.dumps({'geometry': geometry, 'schwinger': schwinger}), flush=True)
    width, height, kappa = 1., 3*math.sqrt(3)/4, 3.
    spectral = []
    for cutoff in [8, 12, 16, 24, 32, 48, 64]:
        row = dict(cutoff_per_coordinate=cutoff, value=spectral_theta(cutoff, width, height, kappa))
        spectral.append(row)
        print(json.dumps({'spectral': row}), flush=True)
    replicas = [image_theta(seed, 14, 3, width, height, kappa) for seed in range(8)]
    winding_control = image_theta(0, 14, 4, width, height, kappa)
    cubature = dict(replicas=replicas, mean=float(np.mean(replicas)),
                    empirical_standard_error=float(np.std(replicas, ddof=1)/math.sqrt(len(replicas))),
                    samples_per_replica=2**14, winding_cutoff=3,
                    change_at_seed_zero_on_increasing_winding_to_four=abs(winding_control-replicas[0]))
    assert all(a['value'] < b['value'] for a, b in zip(spectral, spectral[1:]))
    stationary_sector = stationary_sector_checks()
    draw_net()
    result = dict(scope='Geometry-first; no many-body operator selected. Exact Green graph identity; stationary-net reduction remains unproved.',
                  geometry=geometry, schwinger=schwinger, stationary_sector=stationary_sector,
                  theta_spectral=spectral, theta_image_cubature=cubature,
                  caveats=['Two translation zero modes in the geometric net family.',
                           'Reflected curves are immersed; incidental crossings are not extra vertices.',
                           'Spectral triple overlaps are needed in addition to eigenvalues.',
                           'The spectral cutoff tail is not certified by these computations.',
                           'Randomized cubature error estimates are empirical, not rigorous bounds.',
                           'Critical total length is the large-parameter exponential action, not an exact graph-sum theorem.'],
                  runtime_seconds=time.monotonic()-start)
    (OUT/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'image_cubature': cubature, 'stationary_sector': stationary_sector,
                      'runtime_seconds': result['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
