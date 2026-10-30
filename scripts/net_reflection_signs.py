#!/usr/bin/env python3
r"""Test a complete common-reflection-parity orbit of a geometric theta net.

This is a specified subclass of the full reflected-image expansion. Mixed
reflection parities on the three edges are not included in this orbit test.
Consequently its asymptotics are not asserted for the complete graph invariant.

The full Neumann invariant is checked independently by cosine spectra and
reflected-image cubature. Neither invariant is an eigenvalue trace.
"""
from pathlib import Path
import json
import math
import time
import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import k0e, ndtri
from scipy.signal import fftconvolve
from scipy.stats import qmc
from geodesic_first import image_theta

OUT = Path(__file__).resolve().parents[1] / 'research-data/geodesic-first'


def signed_overlap(q, side):
    """Periodic difference between difference- and sum-coordinate densities."""
    z = np.mod(q, 2*side)
    distance = np.minimum(z, 2*side-z)
    return side-2*distance


def cosine_triples_squared(cutoff, side):
    modes = np.arange(cutoff)
    a, b, c = modes[:, None, None], modes[None, :, None], modes[None, None, :]
    count = sum((n == 0).astype(float) for n in [a+b+c, a+b-c, a-b+c, -a+b+c])
    norm = np.sqrt(np.where(modes == 0, 1., 2.)/side)
    triples = side/4*count*norm[:, None, None]*norm[None, :, None]*norm[None, None, :]
    return triples**2


def neumann_spectral(cutoff, width, height, kappa):
    modes = np.arange(cutoff)
    propagator = 1/((math.pi*modes[:, None]/width)**2+(math.pi*modes[None, :]/height)**2+kappa*kappa)
    return float(np.einsum('abc,def,ad,be,cf->', cosine_triples_squared(cutoff, width),
                          cosine_triples_squared(cutoff, height), propagator, propagator, propagator,
                          optimize=True))


def neumann_spectral_fft(cutoff, width, height, kappa):
    """Same positive cosine-mode sum, using signed-mode selection rules.

    In one coordinate, squared triple-cosine integrals equal the signed
    momentum-conserving count divided by 4L, counting both signs of zero,
    minus 1/L when all three modes vanish. Expanding this in two coordinates
    yields a 2D convolution, two 1D corrections, and a zero-mode correction.
    Zero-padding in fftconvolve avoids circular-convolution aliasing.
    """
    modes = np.arange(1-cutoff, cutoff)
    weights = np.where(modes == 0, 2., 1.)
    raw = 1/((math.pi*modes[:, None]/width)**2+(math.pi*modes[None, :]/height)**2+kappa*kappa)
    signed = raw*weights[:, None]*weights[None, :]
    txy = np.sum(signed*fftconvolve(signed, signed, mode='same'))
    tx = weights/((math.pi*modes/width)**2+kappa*kappa)
    ty = weights/((math.pi*modes/height)**2+kappa*kappa)
    xpart = np.sum(tx*fftconvolve(tx, tx, mode='same'))
    ypart = np.sum(ty*fftconvolve(ty, ty, mode='same'))
    return float((txy/16-(xpart+ypart)/4+1/kappa**6)/(width*height))


def orbit_integrals(kappa, height_ratio=3*math.sqrt(3)/4, radius=14.):
    """Rescaled positive and signed orbits; retain the same unfolded triangle.

    J_BC = integral_R2 W_BC(q) prod_e K0(kappa |q-a_e|)/(2pi) dq.
    a_e=(0,0),(2,2h),(2,-2h). W_N=h; W_D=w_1(qx) w_h(qy).
    The finite scaled box is a numerical truncation, not a certified tail bound.
    """
    height = height_ratio
    displacement = 2-2*height/math.sqrt(3)
    total_length = 2+2*math.sqrt(3)*height
    root = math.sqrt(kappa)
    def integrand(zy, zx):
        qx, qy = displacement+zx/root, zy/root
        lengths = np.array([math.hypot(qx, qy), math.hypot(qx-2, qy-2*height),
                            math.hypot(qx-2, qy+2*height)])
        base = 2*kappa**1.5*np.prod(k0e(kappa*lengths)/(2*math.pi))*math.exp(-kappa*(sum(lengths)-total_length))
        return base*np.array([height, signed_overlap(qx, 1.)*signed_overlap(qy, height)])
    # Split at triangle-weight kinks and projected kernel singularities.
    xbreaks = sorted({-radius, radius} | {(wall-displacement)*root for wall in range(-5, 6)
                                          if -radius < (wall-displacement)*root < radius})
    ybreaks = sorted({0., radius} | {wall*height*root for wall in range(1, 6)
                                   if 0 < wall*height*root < radius})
    def inner(zx):
        value = np.zeros(2)
        for lo, hi in zip(ybreaks, ybreaks[1:]):
            value += quad_vec(lambda zy: integrand(zy, zx), lo, hi, epsabs=2e-12, epsrel=2e-9)[0]
        return value
    value = np.zeros(2)
    error = 0.
    for lo, hi in zip(xbreaks, xbreaks[1:]):
        piece, estimate = quad_vec(inner, lo, hi, epsabs=2e-12, epsrel=2e-9)
        value += piece
        error += estimate
    ell = 4*height/math.sqrt(3)
    hx = 3/(2*ell)
    hy = 1/displacement+1/(2*ell)
    prefactor = 1/(8*math.sqrt(2*math.pi)*math.sqrt(hx*hy*displacement*ell*ell))
    predicted_n = prefactor*height
    predicted_d = prefactor*height*(1-2*displacement)
    return dict(kappa=kappa, height=height, total_length=total_length,
                rescaled_neumann=float(value[0]), rescaled_dirichlet=float(value[1]),
                predicted_neumann_limit=predicted_n, predicted_dirichlet_limit=predicted_d,
                dirichlet_neumann_ratio=float(value[1]/value[0]),
                kappa_times_dirichlet_neumann_ratio=float(kappa*value[1]/value[0]),
                outer_quadrature_error_estimate=error, scaled_box_radius=radius)


def length_density(epsilon, seed, power=16):
    """Independent exact-convolution cubature of the Neumann length density.

    A K0 edge is the Laplace transform of (s^2-ell^2)^(-1/2), s>ell.
    Three such factors convolve using a Dirichlet(1/2,1/2,1/2) average.
    Polar coordinates about the unique net, with vectorized bisection to
    F(q)=L_*+epsilon, cover exactly the sublevel set in relative position.
    No large-kappa or Bessel approximation is used in this calculation.
    """
    height = 3*math.sqrt(3)/4
    points = qmc.Sobol(5, scramble=True, seed=seed).random_base2(power)
    angle = 2*math.pi*points[:, 0]
    direction = np.column_stack([np.cos(angle), np.sin(angle)])
    center = np.array([.5, 0.])
    endpoints = np.array([[0., 0.], [2., 2*height], [2., -2*height]])
    def lengths_at(radial):
        q = center+radial[:, None]*direction
        return np.linalg.norm(q[:, None, :]-endpoints[None, :, :], axis=2)
    lo = np.zeros(len(points))
    hi = np.full(len(points), 4*math.sqrt(epsilon))
    assert np.all(lengths_at(hi).sum(axis=1) > 6.5+epsilon)
    for _ in range(48):
        mid = (lo+hi)/2
        below = lengths_at(mid).sum(axis=1) < 6.5+epsilon
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    radial_max = (lo+hi)/2
    radial = radial_max*np.sqrt(points[:, 1])
    lengths = lengths_at(radial)
    excess = np.maximum(0., 6.5+epsilon-lengths.sum(axis=1))
    gaussian = ndtri(points[:, 2:])
    simplex = gaussian**2/np.sum(gaussian**2, axis=1)[:, None]
    denominator = np.sqrt(np.prod(2*lengths+excess[:, None]*simplex, axis=1))
    density = height/(4*math.pi)*np.mean(radial_max**2*np.sqrt(excess)/denominator)
    return float(density)


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    central = []
    for kappa in [30., 100., 300., 1000., 3000., 10000., 30000.]:
        row = orbit_integrals(kappa)
        central.append(row)
        print(json.dumps({'orbit': row}), flush=True)
    control = orbit_integrals(300., radius=18.)
    radius_change = max(abs(control['rescaled_neumann']-central[2]['rescaled_neumann']),
                        abs(control['rescaled_dirichlet']-central[2]['rescaled_dirichlet']))
    aspects = [orbit_integrals(1000., height_ratio=height) for height in [1., 1.2, 1.5]]
    width, height, kappa = 1., 3*math.sqrt(3)/4, 3.
    spectrum = []
    algorithm_checks = []
    for cutoff in [8, 16]:
        direct = neumann_spectral(cutoff, width, height, kappa)
        fast = neumann_spectral_fft(cutoff, width, height, kappa)
        algorithm_checks.append(dict(cutoff=cutoff, direct=direct, convolution=fast, absolute_error=abs(direct-fast)))
        assert abs(direct-fast) < 1e-15
    for cutoff in [8, 16, 32, 64, 128, 256, 512]:
        row = dict(cutoff_per_coordinate=cutoff, value=neumann_spectral_fft(cutoff, width, height, kappa))
        spectrum.append(row)
        print(json.dumps({'neumann_spectral': row}), flush=True)
    replicas = [image_theta(seed, 18, 3, width, height, kappa, 'neumann') for seed in range(8)]
    cubature = dict(mean=float(np.mean(replicas)), replicas=replicas,
                    empirical_standard_error=float(np.std(replicas, ddof=1)/math.sqrt(len(replicas))),
                    samples_per_replica=2**18, winding_cutoff=3)
    thresholds = []
    threshold_coefficient = 1/(4*math.pi*math.sqrt(13))
    for epsilon in [.1, .03, .01, .003]:
        densities = [length_density(epsilon, seed) for seed in range(4)]
        scaled = np.asarray(densities)/epsilon**1.5
        row = dict(length_excess=epsilon, mean_density=float(np.mean(densities)),
                   scaled_density=float(np.mean(scaled)), predicted_threshold_coefficient=threshold_coefficient,
                   ratio_to_predicted_limit=float(np.mean(scaled)/threshold_coefficient),
                   empirical_standard_error_of_scaled_density=float(np.std(scaled, ddof=1)/2),
                   samples_per_replica=2**16, replicas=4)
        thresholds.append(row)
        print(json.dumps({'threshold_density': row}), flush=True)
    assert abs(central[-1]['rescaled_neumann']/central[-1]['predicted_neumann_limit']-1) < .003
    assert abs(central[-1]['kappa_times_dirichlet_neumann_ratio']-20/39) < .02
    assert radius_change < 1e-9
    assert all(a['value'] < b['value'] for a, b in zip(spectrum, spectrum[1:]))
    result = dict(scope='Complete common-parity winding orbit only; not the entire signed image expansion.',
                  central_rectangle_orbit=central, radius_control_max_absolute_change=radius_change,
                  other_aspect_ratios=aspects,
                  dirichlet_next_term_ratio_coefficient='20/39',
                  neumann_spectral_algorithm_checks=algorithm_checks,
                  neumann_full_graph_spectral=spectrum, neumann_full_graph_cubature=cubature,
                  exact_neumann_orbit_length_density_checks=thresholds,
                  caveats=['Mixed reflection parities are outside the orbit computation.',
                           'No claim that other net families cannot share and alter the same length contribution.',
                           'Finite spectral sums have an uncertified positive tail.',
                           'Randomized cubature uncertainty and radius controls are empirical.'],
                  runtime_seconds=time.monotonic()-start)
    (OUT/'reflection-signs.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'neumann_cubature': cubature, 'radius_control': radius_change,
                      'other_aspect_ratios': aspects, 'runtime_seconds': result['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
