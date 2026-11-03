#!/usr/bin/env python3
"""Exact finite-model checks of the full Neumann rhombus reduction.

P1 equilateral meshes and lumped vertex integration preserve reflection parity.
The identities hold algebraically on each mesh; continuum statements have
separate variational proofs. Mesh refinement is not a proof of convergence.
"""
from pathlib import Path
import json
import math
import numpy as np
from scipy.linalg import eigh, solve

ROOT = Path(__file__).resolve().parents[1]


def mesh(n, triangular=False):
    labels = [(i,j) for i in range(n+1) for j in range(n+1) if not triangular or i+j<=n]
    index = {p:k for k,p in enumerate(labels)}
    points = np.array([[(i+j/2)/n, math.sqrt(3)*j/(2*n)] for i,j in labels])
    elements = []
    for i in range(n):
        for j in range(n):
            for cells in [[(i,j),(i+1,j),(i,j+1)],[(i+1,j+1),(i,j+1),(i+1,j)]]:
                if all(p in index for p in cells):elements.append([index[p] for p in cells])
    stiffness = np.zeros((len(labels),len(labels)))
    mass = np.zeros(len(labels))
    for nodes in elements:
        p = points[nodes]
        matrix = np.column_stack([np.ones(3),p])
        gradients = np.linalg.inv(matrix)[1:,:]
        area = abs(np.linalg.det(matrix))/2
        stiffness[np.ix_(nodes,nodes)] += area*gradients.T@gradients
        mass[nodes] += area/3
    return labels, stiffness, mass


def graph_trace(kernel, weights):
    return float(np.einsum('i,j,ij->',weights,weights,kernel**3))


def check_mesh(n):
    labels,k,m = mesh(n)
    triangle,kt,mt = mesh(n,True)
    seam = np.array([j for j,(a,b) in enumerate(triangle) if a+b==n])
    interior = np.array([j for j,(a,b) in enumerate(triangle) if a+b<n])
    eigenvalues, eigenvectors = eigh(k/np.sqrt(m[:,None]*m[None,:]))
    eigenvectors /= np.sqrt(m[:,None])
    nt = np.linalg.eigvalsh(kt/np.sqrt(mt[:,None]*mt[None,:]))
    mixed = np.linalg.eigvalsh(kt[np.ix_(interior,interior)]/
                             np.sqrt(mt[interior,None]*mt[None,interior]))
    union = np.sort(np.r_[nt,mixed])
    spectral_error = float(np.max(abs(eigenvalues-union))/max(eigenvalues))
    assert spectral_error < 1e-12
    values = []
    for kappa in [.7,1.5,3.]:
        full = solve(k+kappa*kappa*np.diag(m),np.eye(len(m)),assume_a='pos')
        neumann = solve(kt+kappa*kappa*np.diag(mt),np.eye(len(mt)),assume_a='pos')
        correction = neumann[:,seam]@solve(neumann[np.ix_(seam,seam)],neumann[seam,:],assume_a='pos')
        constrained = neumann-correction
        direct_mixed = np.zeros_like(neumann)
        direct_mixed[np.ix_(interior,interior)] = solve(
            kt[np.ix_(interior,interior)]+kappa*kappa*np.diag(mt[interior]),
            np.eye(len(interior)),assume_a='pos')
        correction_error = float(np.max(abs(constrained-direct_mixed)))
        full_trace = graph_trace(full,m)
        tri_trace = graph_trace(neumann,mt)
        cross = float(np.einsum('i,j,ij->',mt,mt,neumann*constrained**2))
        parity_formula = tri_trace/2+3*cross/2
        first_seam = float(np.einsum('i,j,ij->',mt,mt,neumann**2*correction))
        second_seam = float(np.einsum('i,j,ij->',mt,mt,neumann*correction**2))
        seam_formula = 2*tri_trace-3*first_seam+1.5*second_seam
        spectral_trace = None
        if n==6:
            couplings = np.einsum('x,xi,xj,xk->ijk',m,eigenvectors,eigenvectors,eigenvectors,optimize=True)
            denominators = eigenvalues+kappa*kappa
            spectral_trace = float(np.einsum('ijk,i,j,k->',couplings**2,1/denominators,1/denominators,1/denominators))
            assert abs(spectral_trace/full_trace-1) < 1e-11
        assert abs(parity_formula/full_trace-1) < 1e-11
        assert abs(seam_formula/full_trace-1) < 1e-11
        assert correction_error < 1e-10
        values.append(dict(kappa=kappa,full_trace=full_trace,parity_formula=parity_formula,
                           neumann_triangle_trace=tri_trace,first_boundary_term=first_seam,
                           second_boundary_term=second_seam,
                           boundary_corrected_formula=seam_formula,spectral_triple_sum=spectral_trace,
                           boundary_kernel_maximum_error=correction_error,
                           incorrectly_dropping_boundary_correction=2*tri_trace,
                           relative_trace_discrepancy=abs(seam_formula/full_trace-1)))
    return dict(mesh_subdivisions=n,rhombus_nodes=len(m),triangle_nodes=len(mt),
                spectral_union_relative_error=spectral_error,
                first_rhombus_eigenvalues=eigenvalues[:8].tolist(),
                first_neumann_triangle_eigenvalues=nt[:5].tolist(),
                first_mixed_triangle_eigenvalues=mixed[:5].tolist(),checks=values)


def main():
    result = dict(operator='Full scalar Neumann Laplacian on the unit 60-degree rhombus.',
                  observable='Integral of the pointwise cube of its Green kernel; spectral weights are squared triple products.',
                  method='P1 finite elements with lumped vertex integration. All parity and Schur-complement identities are checked on the same mesh.',
                  scope='Finite-dimensional algebraic checks support, but do not replace, the continuum variational proof. Refinement values are not certified continuum error bounds.',
                  meshes=[check_mesh(n) for n in [6,12,20]])
    for path in ['research-data/rhombus-exact-trace.json','docs/data/rhombus-exact-trace.json']:
        (ROOT/path).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
