#!/usr/bin/env python3
"""Exact obtuse-wedge decomposition and finite-model checks of convergent gluing.

The wedge calculation uses continuum kernels. The global-series check is a
finite-element algebraic validation; it is not a continuum error certificate.
"""
from pathlib import Path
import json
import math
import shutil
import numpy as np
from scipy.integrate import quad
from scipy.special import iv, kv, k0
from scipy.linalg import solve
from rhombus_exact_trace import mesh, graph_trace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
ALPHA=2*math.pi/3
BETA=2*ALPHA
Q=math.pi/ALPHA


def diffraction_density(s,delta):
    delta=(delta+BETA/2)%BETA-BETA/2
    c=math.cos(Q*delta)
    if abs(c)<1e-13:
        return 0.  # Average of the two one-sided diffraction limits.
    w=math.exp(-Q*s)
    return 4/BETA*c*w*(1+w*w)/((1-w*w)**2+4*c*c*w*w)


def cone_resolvent(r,rp,delta,kappa):
    delta=(delta+BETA/2)%BETA-BETA/2
    geometric=0.
    for k in range(-2,3):
        angle=delta+k*BETA
        if abs(angle)<math.pi-1e-12:
            distance=math.sqrt(r*r+rp*rp-2*r*rp*math.cos(angle))
            geometric+=k0(kappa*distance)/(2*math.pi)
        elif abs(abs(angle)-math.pi)<1e-12:
            geometric+=k0(kappa*(r+rp))/(4*math.pi)  # Half image on an optical boundary.
    def integrand(s):
        length=math.sqrt(r*r+rp*rp+2*r*rp*math.cosh(s))
        return diffraction_density(s,delta)*k0(kappa*length)/(2*math.pi)
    diffractive,error=quad(integrand,0,40,epsabs=2e-13,epsrel=2e-12,limit=300)
    return geometric,diffractive,error


def wedge_resolvent(r,rp,theta,thetap,kappa):
    terms=[cone_resolvent(r,rp,theta-sign*thetap,kappa) for sign in [1,-1]]
    geometric=sum(x[0] for x in terms);diffractive=sum(x[1] for x in terms)
    values=[]
    for count in [40,80]:
        value=iv(0,kappa*min(r,rp))*kv(0,kappa*max(r,rp))/ALPHA
        for n in range(1,count+1):
            nu=n*Q
            value+=2/ALPHA*math.cos(nu*theta)*math.cos(nu*thetap)*iv(nu,kappa*min(r,rp))*kv(nu,kappa*max(r,rp))
        values.append(float(value))
    assert abs(values[-1]-geometric-diffractive)<2e-11
    assert abs(values[-1]-values[0])<2e-11
    return dict(r=r,r_prime=rp,theta=theta,theta_prime=thetap,kappa=kappa,
                geometric=geometric,diffractive=diffractive,total=geometric+diffractive,
                angular_modes_40=values[0],angular_modes_80=values[-1],
                absolute_discrepancy=abs(values[-1]-geometric-diffractive),
                quadrature_error_estimate=sum(x[2] for x in terms))


def bump(s):
    out=np.zeros_like(s)
    inside=s<1
    out[inside]=np.exp(-1/(1-s[inside]))
    return out


def partition(points):
    height=math.sqrt(3)/2
    vertices=np.array([[0,0],[1,0],[1.5,height],[.5,height]])
    raw=[];patches=[]
    for v in vertices:
        distance=np.linalg.norm(points-v,axis=1)
        raw.append(bump((distance/.45)**2))
        patches.append(distance<.68)
    distances=[]
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        tangent=b-a;inward=np.array([-tangent[1],tangent[0]])
        along=(points-a)@tangent;normal=(points-a)@inward
        distances.append(normal)
        raw.append(bump(((along-.5)/.2)**2)*bump((normal/.1)**2))
        patches.append((along>.12)&(along<.88)&(normal<.27))
    distances=np.array(distances).T
    factor=np.zeros_like(distances)
    inside=distances>.03
    factor[inside]=np.exp(-1/(distances[inside]/.03-1)**2)
    raw.append(np.prod(factor,axis=1))
    patches.append(np.min(distances,axis=1)>1e-9)
    raw=np.array(raw);normalizer=np.sqrt(np.sum(raw*raw,axis=0))
    assert np.min(normalizer)>0
    return raw/normalizer,patches


def global_series_check():
    labels,stiffness,mass=mesh(6)
    points=np.array([[(i+j/2)/6,math.sqrt(3)*j/12] for i,j in labels])
    h=stiffness/np.sqrt(mass[:,None]*mass[None,:])
    chi,patches=partition(points);size=len(mass)
    scale=5.
    while True:
        parametrix=np.zeros_like(h)
        for weight,mask in zip(chi,patches):
            indices=np.flatnonzero(mask)
            if not len(indices):continue
            local=solve(h[np.ix_(indices,indices)]+scale*scale*np.eye(len(indices)),np.eye(len(indices)),assume_a='pos')
            parametrix[np.ix_(indices,indices)]+=weight[indices,None]*local*weight[None,indices]
        residual=(h+scale*scale*np.eye(size))@parametrix-np.eye(size)
        norm=float(np.linalg.norm(residual,2))
        if norm<.5:break
        scale*=2
    exact=solve(h+scale*scale*np.eye(size),np.eye(size),assume_a='pos')
    power=np.eye(size);approx=np.zeros_like(h);rows=[]
    for n in range(41):
        approx+=parametrix@power
        if n in [0,1,2,4,8,16,40]:
            kernel=approx/np.sqrt(mass[:,None]*mass[None,:])
            exact_kernel=exact/np.sqrt(mass[:,None]*mass[None,:])
            rows.append(dict(scattering_order=n,relative_resolvent_error=float(np.linalg.norm(approx-exact)/np.linalg.norm(exact)),
                             relative_graph_trace_error=abs(graph_trace(kernel,mass)/graph_trace(exact_kernel,mass)-1)))
        power=-power@residual
    assert rows[-1]['relative_resolvent_error']<1e-12
    # Continue to the original small mass with the explicit positive mass-shift series.
    kappa=1.5;a=scale*scale-kappa*kappa
    exact_small=solve(h+kappa*kappa*np.eye(size),np.eye(size),assume_a='pos')
    shifted=np.zeros_like(h);term=approx.copy();mass_rows=[]
    for m in range(4001):
        shifted+=term
        if m in [0,10,100,1000,4000]:
            mass_rows.append(dict(mass_shift_order=m,relative_resolvent_error=float(np.linalg.norm(shifted-exact_small)/np.linalg.norm(exact_small))))
        term=a*term@approx
    assert mass_rows[-1]['relative_resolvent_error']<1e-10
    return dict(scope='Finite-element local-subdomain resolvents, not an implementation of the continuum wedge patch series.',
                nodes=size,patches=len(patches),large_mass=scale,residual_operator_norm=norm,
                large_mass_checks=rows,target_mass=kappa,mass_shift_ratio=1-kappa*kappa/(scale*scale),mass_shift_checks=mass_rows)


def main():
    wedge=[wedge_resolvent(.4,.9,theta,thetap,kappa)
           for theta,thetap,kappa in [(.3,1.5,.7),(.4,1.3,1.5),(1.,.6,3.),
                                      (.2,.2+math.pi/3-.01,1.5),(.2,.2+math.pi/3,1.5),
                                      (.2,.2+math.pi/3+.01,1.5)]]
    shadow=[dict(offset=float(offset),**wedge_resolvent(.4,.9,.2,.2+math.pi/3+offset,1.5))
            for offset in np.linspace(-.08,.08,81)]
    result=dict(operator='Actual full Neumann rhombus; local continuum Neumann wedge angle 2*pi/3.',
                continuum_wedge_checks=wedge,finite_global_series=global_series_check(),
                shadow_profile=shadow,
                scope='The exact continuum corner identity and finite-model convergence checks are distinct. No claim of a stationary-family-only exact sum.')
    for path in ['research-data/rhombus-corner-expansion.json','docs/data/rhombus-corner-expansion.json']:
        (ROOT/path).write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),layout='constrained')
    for key,label,color in [('geometric','Geometric','#6876a4'),('diffractive','Corner diffraction','#bd6335'),('total','Combined kernel','#007c83')]:
        axes[0].plot([x['offset'] for x in shadow],[x[key] for x in shadow],label=label,color=color,lw=2)
    axes[0].axvline(0,color='#8c9696',lw=.8,ls=':')
    axes[0].set(title='Exact cancellation at an optical boundary',xlabel=r"$\theta'-(\theta+\pi/3)$",ylabel='Green kernel')
    axes[0].legend(fontsize=9)
    checks=result['finite_global_series']['large_mass_checks']
    axes[1].semilogy([x['scattering_order'] for x in checks],[max(x['relative_resolvent_error'],1e-16) for x in checks],'-o',color='#007c83',label='Resolvent')
    axes[1].semilogy([x['scattering_order'] for x in checks],[max(x['relative_graph_trace_error'],1e-16) for x in checks],'-s',color='#bd6335',label='Cubic graph trace')
    axes[1].set(title='Finite Neumann rhombus: 49 nodes',xlabel='Local gluing series order',ylabel='Relative error')
    axes[1].legend(fontsize=9);axes[1].grid(alpha=.2)
    for extension in ['png','svg']:
        path=ROOT/'research-data'/('rhombus-corner-expansion.'+extension)
        fig.savefig(path,dpi=160)
        if extension=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        shutil.copy2(path,ROOT/'docs/data'/path.name)
    plt.close(fig)
    print(json.dumps(dict(wedge_maximum_error=max(x['absolute_discrepancy'] for x in wedge),
                         global_finite_check=result['finite_global_series']),indent=2))


if __name__=='__main__':main()
