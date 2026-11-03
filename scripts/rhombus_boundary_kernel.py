#!/usr/bin/env python3
"""Independent continuum-image/boundary and full-domain finite-element check."""
from pathlib import Path
import json
import math
import numpy as np
from scipy.special import k0
from scipy.integrate import quad
from scipy.linalg import solve
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

ROOT=Path(__file__).resolve().parents[1]
HEIGHT=math.sqrt(3)/2
B=np.array([[1.5,0.],[HEIGHT,2*HEIGHT]])
R=np.array([[-.5,-HEIGHT],[HEIGHT,-.5]])
REFLECT=np.diag([1.,-1.])
GROUP=[np.eye(2),R,R@R,REFLECT,R@REFLECT,R@R@REFLECT]
X=np.array([.2,.15]);Y=np.array([.65,.1])
SIGMA=np.array([[-.5,-HEIGHT],[-HEIGHT,.5]])
Y_OTHER=np.array([1.5,HEIGHT])+SIGMA@Y


def image_kernel(first,second,kappa,cutoff=10,omit_zero=False):
    result=np.zeros((len(first),len(second)))
    zeros=np.zeros_like(result,dtype=int)
    for matrix in GROUP:
        transformed=second@matrix.T
        for m in range(-cutoff,cutoff+1):
            for n in range(-cutoff,cutoff+1):
                target=transformed+B@np.array([m,n])
                distance=np.linalg.norm(first[:,None,:]-target[None,:,:],axis=2)
                singular=distance<1e-12
                if np.any(singular):
                    assert omit_zero
                    zeros+=singular
                    distance=np.where(singular,1.,distance)
                value=k0(kappa*distance)/(2*math.pi)
                result+=np.where(singular,0.,value)
    return result,zeros


def boundary_kernel(panels,kappa,cutoff=10):
    t=(np.arange(panels)+.5)/panels
    edge=np.column_stack([1-t/2,HEIGHT*t])
    kernel,zeros=image_kernel(edge,edge,kappa,cutoff,True)
    assert np.array_equal(zeros,2*np.eye(panels,dtype=int))
    width=1/panels
    self_average=2/width*quad(lambda r:k0(kappa*r)/math.pi,0,width/2,
                            epsabs=1e-12,epsrel=1e-12)[0]
    kernel[np.diag_indices(panels)]+=self_average
    # Integrate the universal logarithmic singularity over each source panel.
    delta=t[:,None]-t[None,:]
    def primitive(z):
        return np.where(z==0,0.,z*(np.log(np.maximum(abs(z),1e-300))-1))
    average_log=(primitive(delta+width/2)-primitive(delta-width/2))/width
    correction=-(average_log-np.log(np.maximum(abs(delta),1e-300)))/math.pi
    np.fill_diagonal(correction,0.)
    kernel+=correction
    node_boundary,_=image_kernel(np.array([X,Y]),edge,kappa,cutoff)
    correction_xy=node_boundary[0]@solve(kernel,node_boundary[1],assume_a='sym')
    triangle_xy=image_kernel(np.array([X]),np.array([Y]),kappa,cutoff)[0][0,0]
    return dict(panels=panels,image_cutoff=cutoff,triangle_green=float(triangle_xy),
                boundary_correction=float(correction_xy),
                same_half_rhombus_green=float(triangle_xy-correction_xy/2),
                opposite_half_rhombus_green=float(correction_xy/2))


def finite_element(n,kappa):
    index=lambda i,j:i*(n+1)+j
    count=(n+1)**2
    row=[];col=[];data=[];mass=np.zeros(count)
    for i in range(n):
        for j in range(n):
            for labels in [[(i,j),(i+1,j),(i,j+1)],[(i+1,j+1),(i,j+1),(i+1,j)]]:
                nodes=[index(*p) for p in labels]
                points=np.array([[(a+b/2)/n,HEIGHT*b/n] for a,b in labels])
                matrix=np.column_stack([np.ones(3),points])
                gradient=np.linalg.inv(matrix)[1:,:]
                area=abs(np.linalg.det(matrix))/2
                local=area*gradient.T@gradient
                for a in range(3):
                    for b in range(3):row.append(nodes[a]);col.append(nodes[b]);data.append(local[a,b])
                mass[nodes]+=area/3
    row.extend(range(count));col.extend(range(count));data.extend(kappa*kappa*mass)
    matrix=coo_matrix((data,(row,col)),shape=(count,count)).tocsc()
    def evaluation(point):
        second=point[1]/HEIGHT*n;first=(point[0]-.5*point[1]/HEIGHT)*n
        i,j=int(first),int(second);a,b=first-i,second-j
        if a+b<=1:
            nodes=[index(i,j),index(i+1,j),index(i,j+1)];weights=[1-a-b,a,b]
        else:
            nodes=[index(i+1,j+1),index(i,j+1),index(i+1,j)];weights=[a+b-1,1-a,1-b]
        vector=np.zeros(count);vector[nodes]=weights
        return vector
    source=evaluation(X)
    solution=spsolve(matrix,source)
    return dict(subdivisions=n,nodes=count,same_half_rhombus_green=float(evaluation(Y)@solution),
                opposite_half_rhombus_green=float(evaluation(Y_OTHER)@solution))


def main():
    kappa=1.5
    boundary=[boundary_kernel(n,kappa) for n in [16,32,64,128]]
    truncated=boundary_kernel(128,kappa,8)
    meshes=[finite_element(n,kappa) for n in [24,48,96]]
    last=boundary[-1];fe=meshes[-1]
    differences={key:abs(last[key]-fe[key]) for key in ['same_half_rhombus_green','opposite_half_rhombus_green']}
    assert max(differences.values())<5e-4
    assert max(abs(last[k]-truncated[k]) for k in differences)<1e-7
    result=dict(operator='Full scalar Neumann rhombus Laplacian',kappa=kappa,
                first_point=X.tolist(),same_half_second_point=Y.tolist(),opposite_half_second_point=Y_OTHER.tolist(),
                boundary_nystrom=boundary,full_domain_finite_elements=meshes,
                last_absolute_differences=differences,
                scope='Off-diagonal continuum image kernel plus logarithmic-panel boundary inversion, compared with independently assembled sparse full-rhombus finite elements. These are convergence checks, not certified error bounds.')
    for path in ['research-data/rhombus-boundary-kernel.json','docs/data/rhombus-boundary-kernel.json']:
        (ROOT/path).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
