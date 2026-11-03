#!/usr/bin/env python3
"""Exact finite-cutoff certificates for a rotational rhombus graph trace.

This is the C3 quotient of the triangular torus, not the reflecting rhombus.
The full Neumann rhombus needs the separate boundary correction.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json
import math
import numpy as np
from scipy.special import k1, kv

ROOT=Path(__file__).resolve().parents[1]
V=3*math.sqrt(3)/2
SCALE=16*math.pi**2/9


def orbit_certificate(radius, rho):
    modes=[(m,n) for m in range(-2*radius,2*radius+1) for n in range(-2*radius,2*radius+1)
           if m*m-m*n+n*n<=radius*radius]
    def rotate(p):return (-p[1],p[0]-p[1])
    orbit={};sizes=[];energies=[]
    for p in modes:
        if p in orbit:continue
        points={p,rotate(p),rotate(rotate(p))}
        index=len(sizes);sizes.append(len(points));energies.append(p[0]**2-p[0]*p[1]+p[1]**2)
        for q in points:orbit[q]=index
    counts=Counter()
    for p in modes:
        for q in modes:
            r=(-p[0]-q[0],-p[1]-q[1])
            if r in orbit:counts[orbit[p],orbit[q],orbit[r]]+=1
    d=[q+rho for q in energies]
    quotient=sum((F(3*count*count,sizes[i]*sizes[j]*sizes[k]*d[i]*d[j]*d[k])
                  for (i,j,k),count in counts.items()),F(0))
    torus=sum((F(count,d[i]*d[j]*d[k]) for (i,j,k),count in counts.items()),F(0))
    trace2=sum((F(size,denominator**2) for size,denominator in zip(sizes,d)),F(0))
    trace3=sum((F(size,denominator**3) for size,denominator in zip(sizes,d)),F(0))
    rhs=torus/3+2*trace2/rho+2*trace3/3
    assert quotient==rhs
    return dict(cutoff=radius,dimensionless_mass_squared=rho,torus_modes=len(modes),
                quotient_modes=len(sizes),normalized_quotient_trace=str(quotient),
                exact_difference=str(quotient-rhs),status='PASS')


def poisson_checks(kappa):
    limit=256
    m,n=np.meshgrid(np.arange(-2*limit,2*limit+1),np.arange(-2*limit,2*limit+1),indexing='ij')
    q=m*m-m*n+n*n
    denominators=SCALE*q[q<=limit*limit]+kappa*kappa
    spectral2=float(np.sum(denominators**-2))
    spectral3=float(np.sum(denominators**-3))
    index=np.arange(-24,25)
    m,n=np.meshgrid(index,index,indexing='ij')
    radii=np.sqrt(3*(m*m+m*n+n*n)).ravel()
    positive=radii>0
    plane2=np.empty_like(radii);plane3=np.empty_like(radii)
    plane2[~positive]=1/(4*math.pi*kappa*kappa)
    plane3[~positive]=1/(8*math.pi*kappa**4)
    r=radii[positive]
    plane2[positive]=r*k1(kappa*r)/(4*math.pi*kappa)
    plane3[positive]=r*r*kv(2,kappa*r)/(16*math.pi*kappa*kappa)
    geometric2=float(V*np.sum(plane2));geometric3=float(V*np.sum(plane3))
    assert abs(spectral2/geometric2-1)<1e-5
    assert abs(spectral3/geometric3-1)<1e-8
    return dict(kappa=kappa,spectral_trace_square=spectral2,image_trace_square=geometric2,
                relative_square_difference=abs(spectral2/geometric2-1),
                spectral_trace_cube=spectral3,image_trace_cube=geometric3,
                relative_cube_difference=abs(spectral3/geometric3-1),
                note='Finite spectral and image cutoffs; differences are not certified tail bounds.')


def main():
    result=dict(operator='Laplacian on R^2/(Lambda semidirect C3), represented by the rhombus with rotational edge identifications.',
                excluded_operator='This is not the full Dirichlet or Neumann reflecting rhombus.',
                lattice_columns=[[1.5,math.sqrt(3)/2],[0,math.sqrt(3)]],torus_area=V,
                finite_certificates=[orbit_certificate(n,rho) for n in [1,2,4,7] for rho in [1,3]],
                poisson_checks=[poisson_checks(kappa) for kappa in [.7,1.5,3.]])
    for path in ['research-data/rhombus-rotation-trace.json','docs/data/rhombus-rotation-trace.json']:
        (ROOT/path).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
