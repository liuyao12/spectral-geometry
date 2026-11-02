#!/usr/bin/env python3
r"""Geometry-first exact theta-net decomposition and twisted spectral checks.

For three distinct noncollinear planar points, all triangle angles <120 deg,
the integral of three equal-mass Green functions has one explicit Bessel
term at the Steiner length, minus three 1D boundary corrections starting
at the broken-path lengths. The proof uses a circular level-set calculation
on the Schwinger-parameter simplex, NOT stationary phase.

The same amplitude is a Fourier coefficient in boundary twists of an exact
simplex average of resolvent-cube traces of constant-coefficient operators
on a two-particle flat configuration torus. No contact operator is assumed.
"""
from pathlib import Path
import json
import math
import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import k0, k1, roots_legendre, ndtri
from scipy.signal import fftconvolve
from scipy.stats import qmc

OUT = Path(__file__).resolve().parents[1] / 'research-data/twisted-nets'


def triangle_data(points):
    points = np.asarray(points, dtype=float)
    x, y = points[1]-points[0], points[2]-points[0]
    cross = abs(x[0]*y[1]-x[1]*y[0])
    if cross <= 1e-12:
        raise ValueError('noncollinear points required')
    opposite = np.array([np.dot(points[1]-points[2], points[1]-points[2]),
                         np.dot(points[0]-points[2], points[0]-points[2]),
                         np.dot(points[0]-points[1], points[0]-points[1])])
    sides = np.sqrt(opposite)
    angles = np.degrees(np.arccos(np.clip((np.sum(opposite)-2*opposite)/
                                        (2*np.prod(sides)/sides), -1., 1.)))
    zplus = np.sum(opposite)/2+math.sqrt(3)*cross
    zminus = np.sum(opposite)/2-math.sqrt(3)*cross
    return dict(opposite_squared=opposite, sides=sides, angles_degrees=angles,
                steiner_length=math.sqrt(zplus), zplus=zplus, zminus=max(0., zminus),
                broken_lengths=sides.sum()-sides)


def positive_schwinger(points, kappa, order=96):
    """Independent positive 2D quadrature after exact Gaussian vertex integration."""
    data = triangle_data(points)
    a, b, c = data['opposite_squared']
    nodes, weights = roots_legendre(order)
    u = (nodes[:, None]+1)/2
    v = (nodes[None, :]+1)/2
    s1 = np.broadcast_to(u, (order, order))
    s2 = (1-u)*v
    s3 = (1-u)*(1-v)
    determinant = s1*s2+s1*s3+s2*s3
    numerator = a*s1+b*s2+c*s3
    integrand = np.sqrt(numerator)/determinant**1.5*k1(kappa*np.sqrt(numerator/determinant))*(1-u)
    return float(np.sum(weights[:, None]*weights[None, :]*integrand)/(64*math.pi**2*kappa))


def exact_net_decomposition(points, kappa):
    data = triangle_data(points)
    if np.max(data['angles_degrees']) >= 120-1e-10:
        raise ValueError('the full-circle net formula requires all angles strictly below 120 degrees')
    zplus, zminus = data['zplus'], data['zminus']
    total = data['opposite_squared'].sum()
    net_term = float(k0(kappa*data['steiner_length'])/(4*math.pi*math.sqrt(3)*kappa*kappa))
    corrections = []
    errors = []
    for a, threshold in zip(data['opposite_squared'], data['broken_lengths']):
        def integrand(length):
            denominator = 2*math.sqrt((length*length-zplus)*(length*length-zminus))
            ratio = (length*length+total-3*a)/denominator
            angle = math.acos(np.clip(ratio, -1., 1.))
            return float(k1(kappa*length)*angle)
        value, error = quad(integrand, threshold, np.inf,
                            epsabs=max(1e-300, net_term*kappa*1e-11), epsrel=2e-10, limit=160)
        scale = 1/(4*math.pi**2*math.sqrt(3)*kappa)
        corrections.append(value*scale)
        errors.append(error*scale)
    return dict(value=net_term-sum(corrections), net_term=net_term,
                corrections=corrections, quadrature_error_estimate=sum(errors))


def steiner_optimizer(points, weights=None):
    points = np.asarray(points, dtype=float)
    weights = np.ones(3) if weights is None else np.asarray(weights)
    def objective(q):
        distances = np.linalg.norm(q-points, axis=1)
        return float(np.dot(weights, distances)), (weights[:, None]*(q-points)/distances[:, None]).sum(axis=0)
    fit = minimize(objective, points.mean(axis=0), jac=True, method='BFGS', options={'gtol': 1e-10})
    length, gradient = objective(fit.x)
    tangents = (points-fit.x)/np.linalg.norm(points-fit.x, axis=1)[:, None]
    return dict(vertex=fit.x.tolist(), length=length, force_residual=float(np.linalg.norm(gradient)),
                angles_degrees=[math.degrees(math.acos(np.clip(np.dot(tangents[i], tangents[j]), -1, 1)))
                                for i, j in [(0, 1), (0, 2), (1, 2)]])


def weighted_local_density_checks():
    cases = [([[0, 0], [2, 3*math.sqrt(3)/2], [2, -3*math.sqrt(3)/2]], [.8, 1, 1]),
             ([[0, 0], [2, 3*math.sqrt(3)/2], [2, -3*math.sqrt(3)/2]], [1.1, 1, 1]),
             ([[0, 0], [1, 0], [0, 1]], [1, .9, 1.05])]
    rows = []
    for points, weights in cases:
        points, weights = np.asarray(points, dtype=float), np.asarray(weights, dtype=float)
        semiperimeter = weights.sum()/2
        tension_area = math.sqrt(semiperimeter*np.prod(semiperimeter-weights))
        x, y = points[1]-points[0], points[2]-points[0]
        squared = (weights[1]**2*np.dot(x, x)+weights[2]**2*np.dot(y, y)
                   +(weights[0]**2-weights[1]**2-weights[2]**2)*np.dot(x, y)
                   +4*tension_area*abs(x[0]*y[1]-x[1]*y[0]))
        exact_length = math.sqrt(squared)
        optimizer = steiner_optimizer(points, weights)
        center = np.array(optimizer['vertex'])
        assert abs(optimizer['length']-exact_length) < 1e-9
        broken = np.array([np.dot(weights, np.linalg.norm(points-anchor, axis=1)) for anchor in points])
        gap = float(broken.min()-exact_length)
        assert gap > 1e-6
        epsilon = gap/4
        length = exact_length+epsilon
        exact_density = (length*math.acosh(length/exact_length)-math.sqrt((length-exact_length)*(length+exact_length)))/(16*math.pi*tension_area)
        replicas = []
        for seed in range(4):
            samples = qmc.Sobol(5, scramble=True, seed=seed).random_base2(15)
            angles = 2*math.pi*samples[:, 0]
            directions = np.column_stack([np.cos(angles), np.sin(angles)])
            def edge_lengths(radial):
                locations = center+radial[:, None]*directions
                return np.linalg.norm(locations[:, None, :]-points[None, :, :], axis=2)*weights[None, :]
            lo, hi = np.zeros(len(samples)), np.full(len(samples), 4*math.sqrt(epsilon))
            while np.any(edge_lengths(hi).sum(axis=1)<length):
                hi *= 2
            for _ in range(48):
                mid = (lo+hi)/2
                inside = edge_lengths(mid).sum(axis=1)<length
                lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
            limit = (lo+hi)/2
            lengths = edge_lengths(limit*np.sqrt(samples[:, 1]))
            excess = np.maximum(0., length-lengths.sum(axis=1))
            gaussian = ndtri(samples[:, 2:])
            simplex = gaussian**2/np.sum(gaussian**2, axis=1)[:, None]
            denominator = np.sqrt(np.prod(2*lengths+excess[:, None]*simplex, axis=1))
            replicas.append(float(np.mean(limit**2*np.sqrt(excess)/denominator)/(4*math.pi)))
        mean = float(np.mean(replicas))
        error = float(np.std(replicas, ddof=1)/2)
        row = dict(points=points.tolist(), weights=weights.tolist(), weighted_net_length=exact_length,
                   optimizer=optimizer, tension_triangle_area=tension_area, first_collapse_gap=gap,
                   tested_length_excess=epsilon, exact_local_density=exact_density,
                   direct_convolution_density=mean, empirical_standard_error=error,
                   relative_discrepancy=abs(mean-exact_density)/exact_density,
                   samples_per_replica=2**15, replicas=4)
        rows.append(row)
        assert abs(mean-exact_density) < max(8*error, 2e-5*exact_density)
    return rows


def spectral_fourier_coefficient(periods, winding1, winding2, kappa, cutoff, twist_grid):
    """Fine-grid quadrature of the unfolded Bloch spectral Fourier integral.

    The two independent real quasi-momenta u=p+alpha and v=q+beta are
    truncated to [-cutoff,cutoff]^2 and sampled at spacing 1/twist_grid.
    A zero-padded convolution evaluates the 4D sum. The third momentum is
    u+v and is not truncated separately. Errors are truncation/quadrature,
    not a certified continuum bound.
    """
    periods = np.asarray(periods)
    bound = cutoff*twist_grid
    modes = np.arange(-bound, bound+1)/twist_grid
    x, y = modes[:, None], modes[None, :]
    energy = (2*math.pi*x/periods[0])**2+(2*math.pi*y/periods[1])**2+kappa*kappa
    phase1 = np.exp(2j*math.pi*(x*winding1[0]+y*winding1[1]))
    phase2 = np.exp(2j*math.pi*(x*winding2[0]+y*winding2[1]))
    convolution = fftconvolve(phase1/energy, phase2/energy, mode='full')
    summed_modes = np.arange(-2*bound, 2*bound+1)/twist_grid
    third = 1/((2*math.pi*summed_modes[:, None]/periods[0])**2+
               (2*math.pi*summed_modes[None, :]/periods[1])**2+kappa*kappa)
    coefficient = np.sum(convolution*third)/(twist_grid**4*np.prod(periods)**2)
    return dict(real=float(coefficient.real), imaginary=float(coefficient.imag))


def universal_length_density(length, steiner_length):
    if length <= steiner_length:
        return 0.
    return (length*math.acosh(length/steiner_length)-math.sqrt(length*length-steiner_length**2))/(4*math.pi*math.sqrt(3))


def exact_length_density(points, length):
    data = triangle_data(points)
    start = data['steiner_length']
    if length <= start:
        return 0.
    def integrand(radial):
        z = radial*radial
        discriminant = math.sqrt((z-data['zplus'])*(z-data['zminus']))
        if discriminant == 0:
            angle = 2*math.pi
        else:
            ratios = (z+data['opposite_squared'].sum()-3*data['opposite_squared'])/(2*discriminant)
            angle = 2*math.pi-2*np.arccos(np.clip(ratios, -1, 1)).sum()
        return angle*math.sqrt((length-radial)*(length+radial))/radial
    breaks = sorted([start, length]+[float(t) for t in data['broken_lengths'] if start<t<length])
    value = sum(quad(integrand, a, b, epsabs=2e-14, epsrel=1e-10)[0]
                for a, b in zip(breaks, breaks[1:]))
    return value/(8*math.pi**2*math.sqrt(3))


def trace_simplex_check():
    """Compare finite coupled-kinetic resolvent traces with the product sum."""
    periods = np.array([2., 3*math.sqrt(3)/2])
    kappa = 1.2
    alpha, beta = np.array([.17, .31]), np.array([-.28, .07])
    modes = np.array([(i, j) for i in range(-3, 4) for j in range(-3, 4)])
    p = 2*math.pi*(modes+alpha)/periods
    q = 2*math.pi*(modes+beta)/periods
    e1 = np.broadcast_to(np.sum(p*p, axis=1)[:, None], (len(p), len(q))).ravel()
    e2 = np.broadcast_to(np.sum(q*q, axis=1)[None, :], (len(p), len(q))).ravel()
    e3 = np.sum((p[:, None, :]+q[None, :, :])**2, axis=2).ravel()
    product = np.sum(1/((e1+kappa*kappa)*(e2+kappa*kappa)*(e3+kappa*kappa)))/np.prod(periods)**2
    checks = []
    for order in [48, 80]:
        nodes, weights = roots_legendre(order)
        nodes, weights = (nodes+1)/2, weights/2
        value = 0.
        for x, wx in zip(nodes, weights):
            for y, wy in zip(nodes, weights):
                eigenvalues = x*e1+(1-x)*y*e2+(1-x)*(1-y)*e3+kappa*kappa
                value += 2*wx*wy*(1-x)*np.sum(eigenvalues**-3)/np.prod(periods)**2
        checks.append(dict(simplex_order=order, trace_average=value, product_sum=float(product),
                           relative_discrepancy=float(abs(value-product)/product)))
    assert checks[-1]['relative_discrepancy'] < 1e-9
    return checks


def draw_evidence():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), layout='constrained')
    transform = np.array([[1., .5], [0., math.sqrt(3)/2]])
    simplex = np.array([[0., 0.], [1., 0.], [.5, math.sqrt(3)/2], [0., 0.]])
    # Equal-sided physical anchors give a particularly legible parameter-circle picture.
    data = triangle_data([[0, 0], [1, 0], [.5, math.sqrt(3)/2]])
    for ax, radial, title in zip(axes[:2], [1.9, 2.2], ['Before the first collapse', 'After the first collapse']):
        z = radial*radial
        center = (z+data['opposite_squared'].sum()-3*data['opposite_squared'])/(3*z)
        radius = math.sqrt((z-data['zplus'])*(z-data['zminus'])/(3*z*z))
        theta = np.linspace(0, 2*math.pi, 1601)
        circle = transform@center[:2]+radius*np.column_stack([np.cos(theta), np.sin(theta)])
        bary = np.linalg.solve(transform, circle.T).T
        inside = (bary[:, 0]>=0)&(bary[:, 1]>=0)&(bary.sum(axis=1)<=1)
        ax.plot(simplex[:, 0], simplex[:, 1], color='#203d46', lw=1.7)
        for mask, color, label in [(inside, '#007c83', 'kept arc'), (~inside, '#ba6136', 'collapse correction')]:
            ax.plot(np.where(mask, circle[:, 0], np.nan), np.where(mask, circle[:, 1], np.nan),
                    color=color, lw=2.7, label=label)
        ax.scatter(*(transform@center[:2]), s=20, color='#203d46')
        ax.set_title(title+'\n'+rf'$r={radial},\quad L=\sqrt{{3}},\quad B_i=2$')
        ax.set_aspect('equal'); ax.set(xlim=(-.08, 1.08), ylim=(-.1, 1.0)); ax.axis('off')
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.02), frameon=False, fontsize=9)
    height = 3*math.sqrt(3)/4
    points = [[0, 0], [2, 2*height], [2, -2*height]]
    excess = np.geomspace(.001, .15, 75)
    ratios = [exact_length_density(points, 6.5+e)/universal_length_density(6.5+e, 6.5) for e in excess]
    ax = axes[2]
    ax.semilogx(excess, ratios, color='#007c83', lw=2, label='exact corrected density')
    original = json.loads((OUT.parent/'geodesic-first/reflection-signs.json').read_text())
    for index, row in enumerate(original['exact_neumann_orbit_length_density_checks']):
        epsilon = row['length_excess']
        universal = height*universal_length_density(6.5+epsilon, 6.5)
        ax.errorbar(epsilon, row['mean_density']/universal,
                    yerr=row['empirical_standard_error_of_scaled_density']*epsilon**1.5/universal,
                    marker='o', color='#ba6136', capsize=3, label='independent convolution' if index==0 else None)
    gap = math.sqrt(43)-6.5
    ax.axvline(gap, color='#667c82', ls='--', lw=1, label='first collapse')
    ax.set(xlabel=r'$t-L$', ylabel=r'$\rho(t)/\rho_{\mathrm{net}}(t)$', ylim=(.9, 1.01),
           title='The universal density is exact\nuntil the first collapse')
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    fig.savefig(OUT/'evidence.png', dpi=180)
    fig.savefig(OUT/'evidence.svg')
    plt.close(fig)


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    height = 3*math.sqrt(3)/4
    cases = [
        ('algebraic_rectangle', [[0, 0], [2, 2*height], [2, -2*height]]),
        ('square_reflections', [[0, 0], [2, 2], [2, -2]]),
        ('equilateral', [[0, 0], [1, 0], [.5, math.sqrt(3)/2]]),
        ('right_triangle', [[0, 0], [1, 0], [0, 1]]),
        ('scalene', [[0, 0], [2, 0], [.5, 1]]),
        ('near_120', [[0, 0], [1, 0], [-.4, .7]])]
    comparisons = []
    for name, points in cases:
        geom = triangle_data(points)
        optimizer = steiner_optimizer(points)
        assert abs(optimizer['length']-geom['steiner_length']) < 1e-9
        for kappa in [.5, 1., 2., 5.]:
            direct = positive_schwinger(points, kappa, order=112)
            finer = positive_schwinger(points, kappa, order=160)
            net = exact_net_decomposition(points, kappa)
            row = dict(case=name, points=points, kappa=kappa, steiner_length=geom['steiner_length'],
                       broken_lengths=geom['broken_lengths'].tolist(), triangle_angles=geom['angles_degrees'].tolist(),
                       positive_quadrature=finer, quadrature_refinement_change=abs(finer-direct),
                       net_decomposition=net, absolute_discrepancy=abs(finer-net['value']),
                       relative_discrepancy=abs(finer-net['value'])/finer)
            comparisons.append(row)
            print(json.dumps({'case': name, 'kappa': kappa, 'relative_decomposition_error': row['relative_discrepancy']}), flush=True)
        assert max(abs(a-120) for a in optimizer['angles_degrees']) < .01
    assert max(r['relative_discrepancy'] for r in comparisons) < 1e-9
    spectral = []
    points = cases[0][1]
    kappa = .7
    exact = exact_net_decomposition(points, kappa)['value']
    for cutoff, grid in [(8, 8), (16, 8), (32, 8), (16, 16), (32, 16)]:
        value = spectral_fourier_coefficient([2, 2*height], [1, 1], [1, -1], kappa, cutoff, grid)
        row = dict(cutoff=cutoff, twist_grid=grid, coefficient=value, exact_geometric_coefficient=exact,
                   absolute_discrepancy=abs(value['real']-exact), relative_discrepancy=abs(value['real']-exact)/exact)
        spectral.append(row)
        print(json.dumps({'spectral_coefficient_check': row}), flush=True)
    assert spectral[-1]['relative_discrepancy'] < .01
    threshold_reference = Path(__file__).resolve().parents[1]/'research-data/geodesic-first/reflection-signs.json'
    density_checks = []
    if threshold_reference.exists():
        original = json.loads(threshold_reference.read_text())
        for row in original['exact_neumann_orbit_length_density_checks']:
            epsilon = row['length_excess']
            predicted = height*universal_length_density(6.5+epsilon, 6.5)
            full_density = height*exact_length_density(points, 6.5+epsilon)
            density_checks.append(dict(length_excess=epsilon, below_first_collapse=bool(6.5+epsilon<min(triangle_data(points)['broken_lengths'])),
                                       independently_computed_density=row['mean_density'], exact_net_density=predicted,
                                       exact_full_density=full_density,
                                       relative_difference=abs(full_density-row['mean_density'])/row['mean_density']))
    rejected_obtuse = False
    try:
        exact_net_decomposition([[0, 0], [1, 0], [-.5, .5]], 1.)
    except ValueError:
        rejected_obtuse = True
    assert rejected_obtuse
    result = dict(status='Exact theorem for noncollinear triangles with all angles strictly below 120 degrees.',
                  decomposition_checks=comparisons, spectral_fourier_checks=spectral,
                  two_coordinate_trace_simplex_checks=trace_simplex_check(),
                  weighted_local_density_checks=weighted_local_density_checks(),
                  independent_prior_density_checks=density_checks, obtuse_formula_rejected=rejected_obtuse,
                  scope=['Auxiliary two-dimensional periodic doubled rectangle; not the fixed Dirichlet or Neumann billiard spectrum.',
                         'A boundary-twist and simplex family of two-coordinate kinetic operators, not one many-body Hamiltonian.',
                         'Exact net term plus explicit collapsed-path corrections; no general trace formula over arbitrary nets proved.',
                         'Numerical truncation and quadrature errors are observed, not rigorous certificates.'],
                  runtime_seconds=time.monotonic()-started)
    (OUT/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    draw_evidence()
    print(json.dumps({'max_relative_decomposition_error': max(r['relative_discrepancy'] for r in comparisons),
                      'prior_density_checks': density_checks, 'runtime_seconds': result['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
