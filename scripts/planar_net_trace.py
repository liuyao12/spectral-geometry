#!/usr/bin/env python3
"""Bounded planar Dirichlet tests, contact-insertion traces and rational nets.

This does NOT assume a continuum two-dimensional bare delta Hamiltonian exists.
The contact insertion tests and finite-cutoff Hamiltonians are labeled separately.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product, combinations_with_replacement
import json
import math
import platform
import time
import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.special import roots_legendre
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/"docs/data/planar-nets"


def interval_trace(t,L,images=False,cutoff=40):
    if images:
        return L/(2*math.sqrt(math.pi*t))*sum(math.exp(-m*m*L*L/t)
            for m in range(-cutoff,cutoff+1))-.5
    return sum(math.exp(-t*(math.pi*n/L)**2) for n in range(1,cutoff+1))


def diagonal_square_integral(t,L,images=False):
    k=interval_trace(t,L,images)
    return (k*k+.5*interval_trace(2*t,L,images))/L


def kernel(t,L,x,images=False,cutoff=40):
    if images:
        d=x[:,None]-x[None,:]; a=x[:,None]+x[None,:]
        ans=np.zeros_like(d)
        for m in range(-3,4):
            ans+=np.exp(-(d+2*m*L)**2/(4*t))-np.exp(-(a+2*m*L)**2/(4*t))
        return ans/math.sqrt(4*math.pi*t)
    n=np.arange(1,cutoff+1)
    phi=math.sqrt(2/L)*np.sin(np.pi*x[:,None]*n/L)
    return (phi*np.exp(-t*(np.pi*n/L)**2))@phi.T


def two_vertex(beta,t,L,images=True,qorder=160,cutoff=40):
    x,w=roots_legendre(qorder); x=(x+1)*L/2; w=w*L/2
    a=kernel(t,L,x,images,cutoff); b=kernel(beta-t,L,x,images,cutoff)
    return float(w@((a*a)*(b*b))@w)


def net_interval(beta,t,L,cutoff=6):
    """Exact infinite reflected-net theta series, with an explicit cube tail bound.

    Integer cycle coordinates, not interval eigenfunctions, label this sum.
    """
    tau=np.array([t,t,beta-t,beta-t])
    total=0.;error=0.
    for signs in product([-1,1],repeat=4):
        groups=[[i for i,s in enumerate(signs) if s==value] for value in (-1,1)]
        columns=[]
        for group in groups:
            for i in group[:-1]:
                col=np.zeros(4);col[i]=1;col[group[-1]]=-1;columns.append(col)
        Z=np.array(columns).T
        rank=Z.shape[1];G=Z.T@(tau[:,None]*Z)
        # Positive symmetric matrices; scipy handles the tiny solves reliably.
        from scipy.linalg import solve,det
        dual=np.array(list(product(range(-cutoff,cutoff+1),repeat=rank)),dtype=float)
        exponent=L*L*np.sum(dual*solve(G,dual.T,assume_a='pos').T,axis=1)
        amplitude=L**rank/(math.pi**(rank/2)*math.sqrt(det(G)))/(16*L*L)
        total+=math.prod(signs)*amplitude*float(np.exp(-exponent).sum())
        # lambda_max(G) <= trace(G), followed by a union bound outside the cube.
        aa=L*L/float(np.trace(G))
        one_tail=2*math.exp(-aa*(cutoff+1)**2)/(1-math.exp(-aa*(2*cutoff+3)))
        theta_bound=1+math.sqrt(math.pi/aa)
        error+=amplitude*rank*one_tail*theta_bound**(rank-1)
    return total,error


def spectral_contact_sum(beta,t,L,cutoff=40):
    # Exact four-sine integrals: no vertex quadrature or Poisson summation.
    n=np.arange(1,cutoff+1)
    aa,bb=np.meshgrid(n,n,indexing='ij')
    diff=np.abs(aa-bb).ravel();summ=(aa+bb).ravel()
    overlap=((diff[:,None]==diff[None,:])*np.where(diff[:,None]==0,1.,.5)
        -.5*(diff[:,None]==summ[None,:])-.5*(summ[:,None]==diff[None,:])
        +.5*(summ[:,None]==summ[None,:]))/L
    energy=((math.pi*aa/L)**2+(math.pi*bb/L)**2).ravel()
    return float(np.exp(-t*energy)@(overlap*overlap)@np.exp(-(beta-t)*energy))


def four_sines(a,b,c,d,L):
    d1=abs(a-b); d2=abs(c-d); s1=a+b; s2=c+d
    return ((1 if d1==0 else .5)*(d1==d2)-.5*(d1==s2)
            -.5*(s1==d2)+.5*(s1==s2))/L


def contact_matrix(M,Lx,Ly):
    modes=list(product(range(1,M+1),repeat=2))
    pairs=list(combinations_with_replacement(range(len(modes)),2))
    diagonal=np.array([sum((math.pi*modes[i][0]/Lx)**2+(math.pi*modes[i][1]/Ly)**2
                           for i in pair) for pair in pairs])
    W=np.zeros((len(pairs),len(pairs)))
    for i,(a,b) in enumerate(pairs):
        for j in range(i+1):
            c,d=pairs[j]
            norm1=math.sqrt(2/(1+(a==b))); norm2=math.sqrt(2/(1+(c==d)))
            value=2*norm1*norm2*four_sines(modes[a][0],modes[b][0],modes[c][0],modes[d][0],Lx)*four_sines(modes[a][1],modes[b][1],modes[c][1],modes[d][1],Ly)
            W[i,j]=W[j,i]=value
    return diagonal,W


def coord_patterns():
    # Edge durations (2/5,2/5,3/5,3/5); weights have common factor removed.
    w=[3,3,2,2]; out=[]
    for edges in product(list(product([-1,1],range(-2,3))),repeat=4):
        a=sum(w); b=-sum(wi*s for wi,(s,m) in zip(w,edges)); det=a*a-b*b
        if not det:continue
        r1=sum(wi*2*m for wi,(s,m) in zip(w,edges))
        r2=-sum(wi*s*2*m for wi,(s,m) in zip(w,edges))
        u=F(a*r1-b*r2,det);v=F(a*r2-b*r1,det)
        if 0<u<1 and 0<v<1 and u!=v:
            disp=[s*v+2*m-u for s,m in edges]
            if all(disp):out.append((sum(d*d for d in disp),u,v,edges,disp))
    return sorted(out)


def folded_path(u,v):
    times={F(0),F(1)}; wall_times=[]
    for axis in range(2):
        ts=set()
        lo,hi=sorted((u[axis],v[axis]))
        for k in range(math.floor(lo)+1,math.ceil(hi)):
            t=(F(k)-u[axis])/(v[axis]-u[axis])
            if 0<t<1:ts.add(t);times.add(t)
        wall_times.append(ts)
    if wall_times[0]&wall_times[1]:return None
    def fold(z):
        r=z%2
        return r if r<=1 else 2-r
    return [[fold(u[j]+t*(v[j]-u[j])) for j in range(2)] for t in sorted(times)]


def rational_net():
    patterns=coord_patterns(); time_weights=[F(5,2),F(5,2),F(5,3),F(5,3)]
    candidates=[]
    for xx in patterns:
        for yy in patterns:
            u=[xx[1],yy[1]];v=[xx[2],yy[2]]
            displacements=[[xx[4][i],yy[4][i]] for i in range(4)]
            if len({tuple(d) for d in displacements})<4:continue
            if not any(d[0]*displacements[0][1]!=d[1]*displacements[0][0] for d in displacements):continue
            paths=[]
            for i,d in enumerate(displacements):
                p=folded_path(u,[u[j]+d[j] for j in range(2)])
                if p is None:break
                paths.append(p)
            if len(paths)!=4:continue
            fu=[sum(time_weights[i]*displacements[i][j] for i in range(4)) for j in range(2)]
            fv=[sum(time_weights[i]*([xx,yy][j][3][i][0])*displacements[i][j] for i in range(4)) for j in range(2)]
            assert fu==[0,0] and fv==[0,0]
            energy=sum(time_weights[i]**2*sum(d*d for d in displacements[i])*(1 if i<2 else -1) for i in range(4))
            if energy:continue
            lengths2=[sum(d*d for d in vec) for vec in displacements]
            action=sum(lengths2[i]*time_weights[i]/4 for i in range(4))
            candidates.append((action,u,v,xx[3],yy[3],displacements,paths,lengths2))
    assert candidates
    action,u,v,xword,yword,ds,paths,lens=min(candidates)
    unweighted_u=[sum(float(ds[i][j])/math.sqrt(float(lens[i])) for i in range(4)) for j in range(2)]
    unweighted_v=[sum([xword,yword][j][i][0]*float(ds[i][j])/math.sqrt(float(lens[i])) for i in range(4)) for j in range(2)]
    return dict(one_coordinate_patterns=len(patterns),valid_ordered_planar_candidates=len(candidates),
        theta="2/5",u=list(map(str,u)),v=list(map(str,v)),
        xword=xword,yword=yword,displacements=[[str(a) for a in d] for d in ds],
        paths=[[[str(a) for a in point] for point in path] for path in paths],
        squared_lengths=list(map(str,lens)),action=str(action),
        weighted_residual_exact="0",time_stationarity_residual_exact="0",
        unweighted_u_residual=unweighted_u,unweighted_v_residual=unweighted_v)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=dict(geometry="Dirichlet rectangle (0,Lx)x(0,Ly), real reflecting walls",
        interaction="Contact insertions 2 delta^(2)(x1-x2); finite Fourier cutoff only for full interacting matrices",
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        first_contact=[],fixed_time=[],net_grid=[],uv=[],matrix_checks=[])
    for Lx,Ly in ((1.,1.),(1.,1.5)):
        for beta in (.02,.05,.1,.2):
            spectral=-2*beta*diagonal_square_integral(beta,Lx)*diagonal_square_integral(beta,Ly)
            image=-2*beta*diagonal_square_integral(beta,Lx,True)*diagonal_square_integral(beta,Ly,True)
            data['first_contact'].append(dict(Lx=Lx,Ly=Ly,beta=beta,spectral=spectral,images=image,
                absolute_error=abs(spectral-image),mode_cutoff=40,image_cutoff=40))
    for beta in (.05,.1,.2):
        t=.4*beta;Lx,Ly=1.,1.5
        image=two_vertex(beta,t,Lx)*two_vertex(beta,t,Ly)
        spectral=spectral_contact_sum(beta,t,Lx)*spectral_contact_sum(beta,t,Ly)
        check=two_vertex(beta,t,Lx,qorder=240)*two_vertex(beta,t,Ly,qorder=240)
        nx,ex=net_interval(beta,t,Lx);ny,ey=net_interval(beta,t,Ly)
        net=nx*ny;net_error_bound=ex*abs(ny)+ey*abs(nx)+ex*ey
        data['fixed_time'].append(dict(beta=beta,t=t,Lx=Lx,Ly=Ly,images=image,spectral=spectral,
            absolute_error=abs(image-spectral),quadrature_error_estimate=abs(check-image),
            quadrature_orders=[160,240],image_cutoff=3,mode_cutoff=40,
            net_sum=net,net_absolute_error=abs(net-spectral),net_tail_bound=net_error_bound,net_cutoff=6))
    for beta in (.03,.1,.25):
        for theta in (.17,.4,.67):
            t=beta*theta
            nx,ex=net_interval(beta,t,1.);ny,ey=net_interval(beta,t,1.5)
            reference=spectral_contact_sum(beta,t,1.)*spectral_contact_sum(beta,t,1.5)
            data['net_grid'].append(dict(beta=beta,theta=theta,net=nx*ny,spectral=reference,
                absolute_error=abs(nx*ny-reference),tail_bound=ex*abs(ny)+ey*abs(nx)+ex*ey))
    beta=.1;Lx,Ly=1.,1.5
    B=diagonal_square_integral(beta,Lx)*diagonal_square_integral(beta,Ly)
    for fraction in (.4,.2,.1,.05,.02,.01,.005,.002,.001):
        t=beta*fraction
        qorder=320 if fraction>=.01 else 640
        value=two_vertex(beta,t,Lx,qorder=qorder)*two_vertex(beta,t,Ly,qorder=qorder)
        data['uv'].append(dict(beta=beta,t=t,contact_trace=value,
            predicted_limit=B/(8*math.pi),scaled_value=t*value,ratio=t*value/(B/(8*math.pi)),
            quadrature_order=qorder,image_cutoff=3))
    last=data['uv'][-1];qcheck=two_vertex(beta,last['t'],Lx,qorder=960)*two_vertex(beta,last['t'],Ly,qorder=960)
    last['quadrature_error_estimate']=abs(qcheck-last['contact_trace'])
    for M in (4,6):
        diagonal,W=contact_matrix(M,1.,1.5)
        h=.0001
        tick=time.perf_counter()
        plus=eigh(np.diag(diagonal)+h*W,eigvals_only=True)
        minus=eigh(np.diag(diagonal)-h*W,eigvals_only=True)
        for beta in (.05,.1):
            fd=float((np.exp(-beta*plus).sum()-np.exp(-beta*minus).sum())/(2*h))
            derivative=-beta*float(np.diag(W)@np.exp(-beta*diagonal))
            full=-2*beta*diagonal_square_integral(beta,1.)*diagonal_square_integral(beta,1.5)
            t=.4*beta
            contact2=float(np.exp(-t*diagonal)@((W*W)/4)@np.exp(-(beta-t)*diagonal))
            reference2=next(r['net_sum'] for r in data['fixed_time'] if r['beta']==beta)
            data['matrix_checks'].append(dict(mode_cutoff=M,dimension=len(diagonal),beta=beta,
                h=h,finite_difference=fd,finite_matrix_derivative=derivative,continuum_insertion=full,
                finite_difference_error=abs(fd-derivative),cutoff_error=abs(derivative-full),
                fixed_time_matrix_trace=contact2,fixed_time_net_trace=reference2,
                fixed_time_cutoff_error=abs(contact2-reference2),
                solve_seconds=time.perf_counter()-tick))
    data['net']=rational_net()
    assert max(r['absolute_error'] for r in data['first_contact'])<1e-12
    assert max(r['absolute_error'] for r in data['fixed_time'])<1e-10
    assert max(r['net_absolute_error'] for r in data['fixed_time'])<1e-10
    assert max(r['absolute_error'] for r in data['net_grid'])<1e-9
    assert max(r['finite_difference_error'] for r in data['matrix_checks'])<2e-7
    (OUT/'results.json').write_text(json.dumps(data,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':10,
                        'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    net=data['net'];colors=['#006d77','#c06d39','#597db4','#8b568b']
    for i,path in enumerate(net['paths']):
        xy=np.array([[float(F(a)) for a in p] for p in path])
        axs[0].plot(xy[:,0],xy[:,1],color=colors[i],lw=1.7,alpha=.85,label=rf'$e_{i+1}$')
    uv=np.array([[float(F(a)) for a in net[k]] for k in ('u','v')])
    axs[0].scatter(uv[:,0],uv[:,1],s=45,c='#152f39',zorder=5)
    axs[0].annotate(r'$u$',uv[0],xytext=(5,7),textcoords='offset points')
    axs[0].annotate(r'$v$',uv[1],xytext=(5,7),textcoords='offset points')
    axs[0].set(xlim=(0,1),ylim=(0,1),xlabel=r'$x$',ylabel=r'$y$',title='Exact rational reflected net')
    axs[0].spines[['top','right']].set_visible(True);axs[0].set_aspect('equal')
    rows=data['fixed_time']
    axs[1].semilogy([r['beta'] for r in rows],[r['net_sum'] for r in rows],color='#006d77',label='net formula')
    axs[1].scatter([r['beta'] for r in rows],[r['spectral'] for r in rows],marker='x',s=60,c='#c06d39',label='spectral sum')
    axs[1].set(xlabel=r'$\beta$',ylabel=r'$C_\beta(2\beta/5)$',title='Exact two-contact trace identity')
    axs[1].legend(frameon=False)
    axs[2].semilogx([r['t'] for r in data['uv']],[r['ratio'] for r in data['uv']],'o-',color='#006d77')
    axs[2].axhline(1,color='#c06d39',ls='--')
    axs[2].set(xlabel=r'$t$',ylabel=r'$8\pi t\,C_\beta(t)/B_\beta$',title='Second insertion has a logarithmic divergence')
    fig.savefig(OUT/'evidence.png',dpi=170)
    fig.savefig(OUT/'evidence.svg')
    svg=OUT/'evidence.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    print(json.dumps(dict(first_contact_max_error=max(r['absolute_error'] for r in data['first_contact']),
        fixed_time_max_error=max(r['absolute_error'] for r in data['fixed_time']),
        net_sum_max_error=max(r['net_absolute_error'] for r in data['fixed_time']),
        matrix_derivative_max_error=max(r['finite_difference_error'] for r in data['matrix_checks']),
        uv_ratio=data['uv'][-1]['ratio'],net=data['net']),indent=2))


if __name__=='__main__':main()
