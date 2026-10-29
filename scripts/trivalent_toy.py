#!/usr/bin/env python3
"""Mathematical toy: two Dirichlet particles, Neumann heat-kernel pair potential.

No zero-range limit, physical mediator, or renormalization is assumed.
Compare exact trivalent graph sums with spectral coefficients and full matrices.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product,combinations_with_replacement,combinations
from functools import lru_cache
import math,json,time,platform
import numpy as np
import scipy
from scipy.linalg import solve,det,eigh
from scipy.special import roots_legendre
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/'docs/data/trivalent-toy'


def integer_kernel(matrix):
    """Primitive full integer kernel of a signed-incidence transpose.

    Rational RREF has denominators at most two. Enforce the resulting parity
    constraints instead of independently clearing denominators (which can miss
    integer flows).
    """
    A=[[F(int(x)) for x in row] for row in matrix]
    nr,nc=len(A),len(A[0]);piv=[];row=0
    for col in range(nc):
        pivot=next((i for i in range(row,nr) if A[i][col]),None)
        if pivot is None:continue
        A[row],A[pivot]=A[pivot],A[row]
        scale=A[row][col];A[row]=[x/scale for x in A[row]]
        for i in range(nr):
            if i!=row and A[i][col]:
                q=A[i][col];A[i]=[x-q*y for x,y in zip(A[i],A[row])]
        piv.append(col);row+=1
        if row==nr:break
    free=[j for j in range(nc) if j not in piv];r=len(free)
    R=[[A[i][j] for j in free] for i in range(len(piv))]
    assert all(x.denominator in (1,2) for rr in R for x in rr)
    C=[[int(2*x)%2 for x in rr] for rr in R]
    cp=[];cr=0
    for col in range(r):
        p=next((i for i in range(cr,len(C)) if C[i][col]),None)
        if p is None:continue
        C[cr],C[p]=C[p],C[cr]
        for i in range(len(C)):
            if i!=cr and C[i][col]:C[i]=[x^y for x,y in zip(C[i],C[cr])]
        cp.append(col);cr+=1
        if cr==len(C):break
    basis=[]
    for j in range(r):
        v=[0]*r
        if j in cp:v[j]=2
        else:
            v[j]=1
            for i,p in enumerate(cp):v[p]=C[i][j]
        basis.append(v)
    K=np.array(basis,dtype=int).T
    Z=np.zeros((nc,r),dtype=int)
    for i,j in enumerate(free):Z[j]=K[i]
    for i,j in enumerate(piv):
        for a in range(r):
            value=-sum(R[i][b]*int(K[b,a]) for b in range(r))
            assert value.denominator==1
            Z[j,a]=int(value)
    assert np.all(np.array(matrix,dtype=int)@Z==0)
    return Z


# Edges carry their boundary condition. Every node has two D incidences and one N.
GRAPHS={
    'first_direct':(2,((0,0,'D'),(1,1,'D'),(0,1,'N'))),
    'first_exchange':(2,((0,1,'D'),(0,1,'D'),(0,1,'N'))),
    'second_direct':(4,((0,2,'D'),(1,3,'D'),(2,0,'D'),(3,1,'D'),(0,1,'N'),(2,3,'N'))),
    'second_exchange':(4,((0,2,'D'),(1,3,'D'),(2,1,'D'),(3,0,'D'),(0,1,'N'),(2,3,'N')))
}


@lru_cache(None)
def sectors(name):
    V,edges=GRAPHS[name];E=len(edges);out=[]
    parity=np.zeros(V,dtype=int)
    for a,b,bc in edges:
        if bc=='D':parity[a]+=1;parity[b]+=1
    assert np.all(parity%2==0)
    for ss in product([-1,1],repeat=E):
        B=np.zeros((E,V),dtype=int);character=1
        for i,((a,b,bc),s) in enumerate(zip(edges,ss)):
            B[i,a]+=1;B[i,b]-=s
            if bc=='D':character*=s
        out.append((character,integer_kernel(B.T.tolist())))
    return out


@lru_cache(None)
def dual_grid(rank,W):
    return np.array(list(product(range(-W,W+1),repeat=rank)),dtype=float)


def net_interval(name,durations,L,W=6):
    V,edges=GRAPHS[name];E=len(edges);tau=np.array(durations)
    value=0.;bound=0.
    for character,Z in sectors(name):
        r=Z.shape[1];G=Z.T@(tau[:,None]*Z)
        w=dual_grid(r,W)
        energy=L*L*np.sum(w*solve(G,w.T,assume_a='pos').T,axis=1)
        amplitude=2.**(-E)*L**(V-E+r)/(math.pi**(r/2)*math.sqrt(det(G)))
        value+=character*amplitude*float(np.exp(-energy).sum())
        a=L*L/float(np.trace(G))
        tail=2*math.exp(-a*(W+1)**2)/(1-math.exp(-a*(2*W+3)))
        bound+=amplitude*r*tail*(1+math.sqrt(math.pi/a))**(r-1)
    return value,bound


def net_rectangle(name,durations,Lx,Ly,W=6):
    x,ex=net_interval(name,durations,Lx,W);y,ey=net_interval(name,durations,Ly,W)
    return x*y,ex*abs(y)+ey*abs(x)+ex*ey


def cosine_vertex(M,L):
    n=np.arange(1,M+1)
    T=np.zeros((M,M,2*M+1))
    T[:,:,0]=np.eye(M)/math.sqrt(L)
    for k in range(1,2*M+1):
        T[:,:,k]=((np.abs(n[:,None]-n[None,:])==k).astype(float)
                   -(n[:,None]+n[None,:]==k).astype(float))/math.sqrt(2*L)
    return T


def interval_potential(M,L,sigma):
    T=cosine_vertex(M,L);k=np.arange(2*M+1)
    return np.einsum('ika,a,jla->ijkl',T,np.exp(-sigma*(math.pi*k/L)**2),T,
                     optimize=True).reshape(M*M,M*M)


def spectral_graph(name,durations,L,M=24):
    i,j=np.meshgrid(np.arange(M),np.arange(M),indexing='ij')
    swap=(j*M+i).ravel()
    energies=((math.pi*(i+1)/L)**2+(math.pi*(j+1)/L)**2).ravel()
    if name.startswith('first'):
        beta,_,sigma=durations;V=interval_potential(M,L,sigma)
        if name=='first_direct':return float(np.diag(V)@np.exp(-beta*energies))
        return float(V[swap,np.arange(M*M)]@np.exp(-beta*energies))
    t,_,u,_,s1,s2=durations
    A=interval_potential(M,L,s1);B=interval_potential(M,L,s2)
    if name=='second_exchange':A=A[swap,:]
    return float(np.exp(-u*energies)@(A*B.T)@np.exp(-t*energies))


def boson_matrix(M,Lx,Ly,sigma,fermion=False):
    modes=np.array(list(product(range(M),repeat=2)))
    maker=combinations if fermion else combinations_with_replacement
    pairs=np.array(list(maker(range(M*M),2)));i,j=pairs.T
    x,y=modes[:,0],modes[:,1]
    Vx=interval_potential(M,Lx,sigma);Vy=interval_potential(M,Ly,sigma)
    def element(k,l):
        return Vx[(x[i]*M+x[j])[:,None],(x[k]*M+x[l])[None,:]]*Vy[(y[i]*M+y[j])[:,None],(y[k]*M+y[l])[None,:]]
    W=element(i,j)+(-1 if fermion else 1)*element(j,i)
    if not fermion:
        norm=np.sqrt(1+(i==j).astype(float));W/=norm[:,None]*norm[None,:]
    energy=(math.pi*(x+1)/Lx)**2+(math.pi*(y+1)/Ly)**2
    return energy[i]+energy[j],W


def matrix_coefficients(energies,W,beta):
    f=np.exp(-beta*energies);delta=energies[None,:]-energies[:,None]
    difference=f[:,None]-f[None,:]
    weight=np.empty_like(delta)
    mask=np.abs(delta)>1e-9
    np.divide(difference,delta,out=weight,where=mask)
    weight[~mask]=(beta*np.exp(-beta*(energies[:,None]+energies[None,:])/2))[~mask]
    return float(f.sum()),-beta*float(np.diag(W)@f),beta/2*float(np.sum(W*W*weight))


def net_coefficients(beta,sigma,Lx,Ly,order=20,W=6,fermion=False):
    sign=-1 if fermion else 1
    direct,_=net_rectangle('first_direct',(beta,beta,sigma),Lx,Ly,W)
    exchange,_=net_rectangle('first_exchange',(beta,beta,sigma),Lx,Ly,W)
    a1=-beta/2*(direct+sign*exchange)
    q,weights=roots_legendre(order);ts=(q+1)*beta/2
    values=[]
    for t in ts:
        durations=(t,t,beta-t,beta-t,sigma,sigma)
        d,_=net_rectangle('second_direct',durations,Lx,Ly,W)
        e,_=net_rectangle('second_exchange',durations,Lx,Ly,W)
        values.append((d+sign*e)/2)
    a2=beta*beta/4*float(weights@np.array(values))
    return a1,a2


def one_trace(beta,L):
    return sum(math.exp(-beta*(math.pi*n/L)**2) for n in range(1,100))


def free_trace(beta,Lx,Ly,fermion=False):
    k=one_trace(beta,Lx)*one_trace(beta,Ly)
    k2=one_trace(2*beta,Lx)*one_trace(2*beta,Ly)
    return (k*k+(-1 if fermion else 1)*k2)/2


def potential_norm(sigma,Lx,Ly):
    def corner(L):return (1+2*sum(math.exp(-sigma*(math.pi*n/L)**2) for n in range(1,100)))/L
    return corner(Lx)*corner(Ly)


def free_baseline_checks():
    rows=[]
    for Lx,Ly in ((1.,1.),(1.,1.5)):
        energy=[(math.pi*x/Lx)**2+(math.pi*y/Ly)**2
                for x in range(1,41) for y in range(1,41)]
        for beta in (.01,.03,.05):
            for fermion in (False,True):
                occupations=[1.]+[0.]*4
                for e in energy:
                    weight=math.exp(-beta*e);old=occupations.copy()
                    for n in range(5):
                        occupations[n]=math.fsum(old[n-j]*weight**j for j in range(min(n,1 if fermion else 4)+1))
                cycles=[1.]
                for n in range(1,5):
                    cycles.append(math.fsum(((-1)**(j-1) if fermion else 1)
                        *sum(math.exp(-j*beta*e) for e in energy)*cycles[n-j]
                        for j in range(1,n+1))/n)
                    if n>=2:
                        rows.append(dict(Lx=Lx,Ly=Ly,beta=beta,N=n,
                            statistics='antisymmetric' if fermion else 'symmetric',
                            cycle_trace=cycles[n],occupation_trace=occupations[n],
                            absolute_error=abs(cycles[n]-occupations[n]),
                            relative_error=abs(cycles[n]-occupations[n])/occupations[n],mode_cutoff=40))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    data=dict(model='H_g=-Delta_x^D-Delta_y^D+g k_sigma^N(x,y), bounded rectangle, sigma fixed positive',
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        graph_checks=[],coefficient_checks=[],full_traces=[],controls=[])
    data['free_baseline']=free_baseline_checks()
    assert max(r['relative_error'] for r in data['free_baseline'])<1e-9
    cases=[(1.,1.,.06,.04),(1.,1.5,.1,.04),(1.,3*math.sqrt(3)/4,.12,.02)]
    for Lx,Ly,beta,sigma in cases:
        for name in GRAPHS:
            durations=(beta,beta,sigma) if name.startswith('first') else (.37*beta,.37*beta,.63*beta,.63*beta,sigma,sigma)
            geom,tail=net_rectangle(name,durations,Lx,Ly)
            spectral=spectral_graph(name,durations,Lx)*spectral_graph(name,durations,Ly)
            data['graph_checks'].append(dict(graph=name,Lx=Lx,Ly=Ly,beta=beta,sigma=sigma,
                net=geom,spectral=spectral,error=abs(geom-spectral),winding_tail_bound=tail,
                spectral_cutoff=24,winding_cutoff=6))
    # Full finite interacting spectra, plus net expansion coefficients in a common model.
    for fermion in (False,True):
        Lx,Ly,beta,sigma=1.,1.5,.1,.04
        a1,a2=net_coefficients(beta,sigma,Lx,Ly,order=16,fermion=fermion)
        a1c,a2c=net_coefficients(beta,sigma,Lx,Ly,order=24,fermion=fermion)
        z0=free_trace(beta,Lx,Ly,fermion)
        for M in (4,6,8):
            e,W=boson_matrix(M,Lx,Ly,sigma,fermion)
            zm,b1,b2=matrix_coefficients(e,W,beta)
            data['coefficient_checks'].append(dict(statistics='fermion' if fermion else 'boson',
                mode_cutoff=M,dimension=len(e),beta=beta,sigma=sigma,Lx=Lx,Ly=Ly,
                free_trace=z0,matrix_free_trace=zm,net_a1=a1c,net_a2=a2c,matrix_a1=b1,matrix_a2=b2,
                a1_error=abs(a1c-b1),a2_error=abs(a2c-b2),quadrature_error_estimate=abs(a2c-a2),
                quadrature_orders=[16,24]))
            for g in (.25,.5,1.,2.,5.):
                vals=eigh(np.diag(e)+g*W,eigvals_only=True)
                exact=float(np.exp(-beta*vals).sum())
                first=z0+g*a1c;second=first+g*g*a2c
                norm=potential_norm(sigma,Lx,Ly)
                bound=z0*(beta*g*norm)**3/6
                data['full_traces'].append(dict(statistics='fermion' if fermion else 'boson',
                    mode_cutoff=M,dimension=len(e),g=g,beta=beta,sigma=sigma,Lx=Lx,Ly=Ly,
                    full_matrix_trace=exact,free_trace=z0,interaction_shift=exact-zm,
                    net_first_order=first,net_second_order=second,second_order_error=abs(second-exact),
                    remainder_bound=bound,lowest_energies=vals[:6].tolist()))
    # A constant kernel is an exact negative control for a geometry-only inference.
    area=1.5
    for g in (0.,.5,2.):
        beta=.1;z0=free_trace(beta,1.,1.5)
        data['controls'].append(dict(type='constant pair potential',g=g,beta=beta,
            Z0=z0,Zg=math.exp(-beta*g/area)*z0,geometry_changed=False))
    assert max(r['error'] for r in data['graph_checks'])<1e-8
    assert max(r['a2_error'] for r in data['coefficient_checks'] if r['mode_cutoff']==8)<1e-7
    data['seconds']=time.perf_counter()-started
    (OUT/'results.json').write_text(json.dumps(data,indent=2)+'\n')
    # Genuine 120-degree theta net in an algebraic rectangle.
    H=3*math.sqrt(3)/4;u=np.array([.25,H/2]);v=np.array([.75,H/2])
    displacements=[np.array([-1.5,2*H]),np.array([-1.5,-2*H]),np.array([.5,0.])]
    def foldpath(d):
        times={0.,1.}
        for j,L in enumerate((1.,H)):
            a,b=u[j]/L,(u[j]+d[j])/L
            for k in range(math.floor(min(a,b))+1,math.ceil(max(a,b))):
                t=(k-a)/(b-a)
                if 0<t<1:times.add(t)
        out=[]
        for t in sorted(times):
            p=u+t*d;q=p/np.array([1.,H]);q=q%2;q=np.where(q<=1,q,2-q)
            out.append(q*np.array([1.,H]))
        return np.array(out)
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':10})
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for i,d in enumerate(displacements):
        p=foldpath(d);axs[0].plot(p[:,0],p[:,1],color=('#006d77','#597db4','#c06d39')[i],lw=2)
    axs[0].scatter([u[0],v[0]],[u[1],v[1]],c='#172f37',zorder=5)
    axs[0].set(xlim=(0,1),ylim=(0,H),xlabel=r'$x$',ylabel=r'$y$',title='Trivalent kernel graph: specified incidence')
    axs[0].set_aspect('equal')
    for stat,color in (('boson','#006d77'),('fermion','#c06d39')):
        rows=[r for r in data['full_traces'] if r['statistics']==stat and r['mode_cutoff']==8]
        axs[1].plot([r['g'] for r in rows],[r['interaction_shift']/r['free_trace'] for r in rows],'o-',label=stat,color=color)
        axs[2].loglog([r['g'] for r in rows],[r['second_order_error'] for r in rows],'o-',label=stat,color=color)
    axs[1].set(xlabel=r'$g$',ylabel=r'$(Z_g-Z_0)/Z_0$',title='Interaction changes the full trace')
    axs[1].legend(frameon=False)
    axs[2].set(xlabel=r'$g$',ylabel=r'$|Z_g-Z_{\mathrm{net},\leq2}|$',title='Test the net expansion against spectra')
    axs[2].set_xticks([.25,.5,1,2,5],labels=[r'$1/4$',r'$1/2$',r'$1$',r'$2$',r'$5$'])
    axs[2].tick_params(axis='x',which='minor',bottom=False,labelbottom=False)
    axs[2].legend(frameon=False)
    fig.savefig(OUT/'evidence.png',dpi=170);fig.savefig(OUT/'evidence.svg')
    p=OUT/'evidence.svg';p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
    print(json.dumps(dict(max_graph_error=max(r['error'] for r in data['graph_checks']),
        coefficients=[r for r in data['coefficient_checks'] if r['mode_cutoff']==8],seconds=data['seconds']),indent=2))


if __name__=='__main__':main()
