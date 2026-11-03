#!/usr/bin/env python3
"""Intrinsic physical itinerary weight for the displayed rhombus family.

No patch cutoff is used. A Gaussian bridge representation gives a bounded
probability weight for the direct/(bottom,left)/(right,top) ray chamber.
This is a specular itinerary contribution, not the complete family-indexed
trace including diffraction.
"""
from pathlib import Path
import json
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import k0, k0e, kve, roots_genlaguerre, roots_legendre
from scipy.stats import geninvgauss
from polygon_theta_families import Polygon, polygons

ROOT=Path(__file__).resolve().parents[1]
HEIGHT=math.sqrt(3)/2
R=np.array([[-.5,-HEIGHT],[HEIGHT,-.5]])
REFLECT_LONG=np.array([[.5,HEIGHT],[HEIGHT,-.5]])
RADIUS=3.


def inside(polygon,points):
    return np.all(points@polygon.normals.T<polygon.constants[None,:],axis=1)


def visible_word(polygon,u,v,word):
    displacement=v@word.matrix.T+word.shift-u
    length=np.linalg.norm(displacement,axis=1)
    direction=displacement/length[:,None]
    point=u.copy();remaining=length.copy();mask=length>1e-12
    for wall in word.sequence:
        normal=polygon.normals[wall]
        denominator=direction@normal
        safe=np.where(abs(denominator)>1e-14,denominator,1.)
        flight=(polygon.constants[wall]-point@normal)/safe
        hit=point+flight[:,None]*direction
        edge=polygon.vertices[(wall+1)%4]-polygon.vertices[wall]
        fraction=(hit-polygon.vertices[wall])@edge/np.dot(edge,edge)
        mask&=(denominator>1e-14)&(flight>0)&(remaining>flight)&(fraction>0)&(fraction<1)
        point=hit
        remaining-=flight
        direction=direction-2*denominator[:,None]*normal
    endpoint=point+remaining[:,None]*direction
    return mask&(np.linalg.norm(endpoint-v,axis=1)<1e-7)


def chamber_probability(kappa,samples,seed):
    rng=np.random.default_rng(seed)
    polygon=Polygon(polygons()['rhombus_60'])
    by_word={w.sequence:w for level in polygon.words(2) for w in level}
    s=rng.dirichlet(np.ones(3),size=samples)
    total_time=geninvgauss.rvs(2,kappa*RADIUS,scale=RADIUS/(2*kappa),size=samples,random_state=rng)
    xi=rng.normal(size=(samples,2));eta=rng.normal(size=(samples,2))
    endpoint=np.array([RADIUS,0.])
    fluctuation=np.sqrt(2*total_time*s[:,0]*(1-s[:,0]))[:,None]*xi
    z0=s[:,0,None]*endpoint+fluctuation
    z1=(s[:,1,None]*endpoint-s[:,1,None]/(1-s[:,0,None])*fluctuation+
        np.sqrt(2*total_time*s[:,1]*s[:,2]/(1-s[:,0]))[:,None]*eta)
    u=(z0-z1@(R@R).T)@np.linalg.inv(np.eye(2)-R@R).T
    v=z1+u@R.T
    mask=inside(polygon,u)&inside(polygon,v)
    indices=np.flatnonzero(mask)
    selected=np.ones(len(indices),dtype=bool)
    for seq in [(0,3),(1,2)]:
        selected&=visible_word(polygon,u[indices],v[indices],by_word[seq])
    successes=int(np.sum(selected));probability=successes/samples
    # Exact whole-plane convolution coefficient, evaluated without underflow.
    scaled_prefactor=RADIUS**2/(48*math.pi)*math.sqrt(kappa)*kve(2,kappa*RADIUS)
    leading_coefficient=1/(16*math.sqrt(6*math.pi))
    return dict(kappa=kappa,samples=samples,successes=successes,probability=probability,
                standard_error=math.sqrt(probability*(1-probability)/samples),
                scaled_physical_itinerary_weight=scaled_prefactor*probability,
                ratio_to_leading_area_term=scaled_prefactor*probability/leading_coefficient,
                whole_plane_scaled_prefactor=scaled_prefactor)


def edge_integral(order):
    radial,wr=roots_genlaguerre(order,2)
    nodes,weights=roots_legendre(order)
    theta=(nodes+1)*math.pi/4;wt=weights*math.pi/4
    u=(nodes+1)/2;wu=weights/2
    radius=radial[:,None,None]
    factor=np.sqrt(np.cos(theta)[None,:,None]**2+
                   np.sin(theta)[None,:,None]**2*u[None,None,:]**2)
    distance=radius*factor
    first=k0e(distance)**2*k0e(radius)*np.exp(-2*distance)
    second=k0e(distance)*k0e(radius)**2*np.exp(-distance-radius)
    value=6/(2*math.pi)**3*np.einsum('i,j,k,ijk,j->',wr,wt,wu,first+second,np.sin(theta),optimize=True)
    return float(value)


def verify_stationary_geometry():
    polygon=Polygon(polygons()['rhombus_60'])
    by_word={w.sequence:w for level in polygon.words(2) for w in level}
    rng=np.random.default_rng(42)
    s=rng.dirichlet(np.ones(3),size=1000)
    z0=RADIUS*s[:,0,None]*np.array([1.,0.])
    z1=RADIUS*s[:,1,None]*np.array([1.,0.])
    u=(z0-z1@(R@R).T)@np.linalg.inv(np.eye(2)-R@R).T
    v=z1+u@R.T
    observed=inside(polygon,u)&inside(polygon,v)
    for seq in [(0,3),(1,2)]:observed&=visible_word(polygon,u,v,by_word[seq])
    expected=(s[:,0]<s[:,1])&(s[:,0]<s[:,2])
    assert np.array_equal(observed,expected)
    assert np.max(np.linalg.norm(v-u@REFLECT_LONG.T,axis=1))<1e-12
    return dict(points=1000,smallest_direct_edge_characterization='PASS')


def collapsed_weight(kappa,samples,seed):
    """Importance sampling of the actual direct/bottom/top chamber."""
    rng=np.random.default_rng(seed)
    polygon=Polygon(polygons()['rhombus_60'])
    by_word={w.sequence:w for level in polygon.words(1) for w in level}
    coordinates=rng.uniform(size=(samples,2))
    u=coordinates@np.array([[1.,0.],[.5,HEIGHT]])
    # The normalized proposal density for eta is kappa^2*g_kappa(|eta|).
    eta=np.sqrt(2*rng.exponential(size=samples))[:,None]*rng.normal(size=(samples,2))/kappa
    v=u+eta
    mask=inside(polygon,v)
    for seq in [(0,),(2,)]:
        mask&=visible_word(polygon,u,v,by_word[seq])
    selected=np.flatnonzero(mask)
    lengths=[]
    for seq in [(0,),(2,)]:
        word=by_word[seq]
        lengths.append(np.linalg.norm(v[selected]@word.matrix.T+word.shift-u[selected],axis=1))
    first,second=lengths
    values=np.zeros(samples)
    values[selected]=(HEIGHT*kappa/(4*math.pi**2)*k0e(kappa*first)*k0e(kappa*second)*
                      np.exp(-kappa*(first+second-2*HEIGHT)))
    return dict(kappa=kappa,samples=samples,scaled_weight=float(np.mean(values)),
                standard_error=float(np.std(values,ddof=1)/math.sqrt(samples)),
                leading_coefficient=1/32)


def main():
    cases=[1,2,5,20,80,320,1280,5120]
    samples=2**18
    rows=[chamber_probability(kappa,samples,20261103+i) for i,kappa in enumerate(cases)]
    bulk,error=quad(lambda r:r*k0(r)**3,0,np.inf,epsabs=1e-11,epsrel=1e-11,limit=250)
    bulk/=4*math.pi**2;error/=4*math.pi**2
    edge=3/32-1/(8*math.sqrt(3))
    edge_checks=[dict(order=n,direct_integral=edge_integral(n)) for n in [32,64,128]]
    assert abs(edge_checks[-1]['direct_integral']-edge)<5e-5
    result=dict(scope='Actual physical specular itinerary, not the complete trace or the full intrinsic diffraction regrouping.',
                reflection_words_zero_based=[[],[0,3],[1,2]],length=3,
                exact_weight='W(kappa)=9*K2(3*kappa)/(48*pi*kappa^2) * P(kappa).',
                limiting_probability_large_mass='1/3',limiting_probability_small_mass='0',
                monte_carlo=rows,
                simultaneous_99_percent_hoeffding_half_width=math.sqrt(math.log(2*len(cases)/.01)/(2*samples)),
                statistical_scope='The Hoeffding statement assumes ideal independent sampling, and is not an interval arithmetic certificate.',
                stationary_geometry=verify_stationary_geometry(),
                collapsed_chamber_monte_carlo=[collapsed_weight(k,samples,20261130+i) for i,k in enumerate([20,80,320,1280])],
                local_coefficients=dict(bulk=bulk,bulk_quadrature_error_estimate=error,
                                        neumann_edge=edge,edge_exact='3/32-1/(8*sqrt(3))',
                                        edge_checks=edge_checks,
                                        collapsed_bottom_top_predicted_leading_coefficient_per_ordered_chamber='1/32',
                                        collapsed_bottom_top_length=math.sqrt(3)))
    for path in ['research-data/intrinsic-rhombus-weights.json','docs/data/intrinsic-rhombus-weights.json']:
        (ROOT/path).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
