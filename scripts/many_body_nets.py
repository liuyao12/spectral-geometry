#!/usr/bin/env python3
"""Reproduce the circle experiments. Requires numpy, scipy and matplotlib.

Run from the repository root: python scripts/many_body_nets.py
No random sampling; all cutoffs and observed errors are saved in results.json.
"""
from pathlib import Path
import itertools
import json
import math
import platform
import time

import numpy as np
import scipy
from scipy.optimize import brentq, root
from scipy.linalg import eigh, cholesky, cho_solve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs/data/many-body"
L = 1.0


def one_trace(t, winding=False, cutoff=40):
    if winding:
        return L / math.sqrt(4 * math.pi * t) * math.fsum(
            math.exp(-(w * L) ** 2 / (4 * t)) for w in range(-cutoff, cutoff + 1))
    return math.fsum(math.exp(-t * (2 * math.pi * n / L) ** 2)
                     for n in range(-cutoff, cutoff + 1))


def canonical_cycles(n, beta, fermion=False):
    z = [1.0]
    for k in range(1, n + 1):
        z.append(math.fsum(((-1) ** (j - 1) if fermion else 1)
                          * one_trace(j * beta, True) * z[k - j]
                          for j in range(1, k + 1)) / k)
    return z[n]


def canonical_occupations(n, beta, fermion=False):
    # Independent occupation-number generating polynomial, not the cycle recurrence.
    z = np.zeros(n + 1)
    z[0] = 1
    for m in range(-40, 41):
        weight = math.exp(-beta * (2 * math.pi * m / L) ** 2)
        old = z.copy()
        z[:] = 0
        for k in range(n + 1):
            for j in range(min(k, 1 if fermion else n) + 1):
                z[k] += old[k - j] * weight ** j
    return float(z[n])


def relative_q(c, j, parity=0):
    # Stable root equations with no tangent poles. a=L/2.
    a = L / 2
    lo = (j + parity / 2) * math.pi / a
    hi = (j + (parity + 1) / 2) * math.pi / a
    if c == 0:
        return lo
    def f(q):
        u = q * a - j * math.pi
        if parity == 0:
            return q * math.sin(u) - c / 2 * math.cos(u)
        return -q * math.cos(u) - c / 2 * math.sin(u)
    return brentq(f, lo, hi, xtol=1e-13, rtol=1e-14)


def two_trace(c, beta):
    qs = [[relative_q(c, j, p) for j in range(41)] for p in (0, 1)]
    return math.fsum(math.exp(-beta * ((2 * math.pi * n / L) ** 2 / 2 + 2 * q*q))
                     for n in range(-40, 41) for q in qs[n % 2])


def fock_matrix(n, cutoff, total_momentum=0, c=1):
    modes = list(range(-cutoff, cutoff + 1))
    basis = [b for b in itertools.combinations_with_replacement(modes, n)
             if sum(b) == total_momentum]
    index = {b: i for i, b in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)))
    for col, b in enumerate(basis):
        occ = {m: b.count(m) for m in set(b)}
        matrix[col, col] += sum((2 * math.pi * m / L) ** 2 for m in b)
        # c/L sum_{p+q=r+s} a_p^dagger a_q^dagger a_r a_s.
        for r in occ:
            for s in occ:
                after = occ.copy()
                amp = math.sqrt(after[s])
                after[s] -= 1
                if after[r] == 0:
                    continue
                amp *= math.sqrt(after[r])
                after[r] -= 1
                for p in modes:
                    q = r + s - p
                    if q < -cutoff or q > cutoff:
                        continue
                    final = after.copy()
                    ap = math.sqrt(final.get(q, 0) + 1)
                    final[q] = final.get(q, 0) + 1
                    ap *= math.sqrt(final.get(p, 0) + 1)
                    final[p] = final.get(p, 0) + 1
                    target = tuple(sorted(m for m, count in final.items() for _ in range(count)))
                    matrix[index[target], col] += c / L * amp * ap
    assert np.max(np.abs(matrix - matrix.T)) < 1e-10
    return matrix


def bethe(numbers, c):
    numbers = np.array(numbers, dtype=float)
    def fun(k):
        d = k[:, None] - k[None, :]
        return L * k + 2 * np.arctan(d / c).sum(axis=1) - 2 * math.pi * numbers
    def jac(k):
        d = k[:, None] - k[None, :]
        w = 2 * c / (c*c + d*d)
        np.fill_diagonal(w, 0)
        return np.diag(L + w.sum(axis=1)) - w
    sol = root(fun, 2 * math.pi * numbers / L, jac=jac, tol=1e-11)
    residual = float(np.linalg.norm(fun(sol.x)))
    assert residual < 2e-10, (numbers, c, sol.message, residual)
    return dict(numbers=numbers.tolist(), rapidities=sol.x.tolist(),
                energy=float(sol.x @ sol.x), residual=residual,
                root_error_bound=residual / L)


def three_body_contact(cutoff):
    """Trace of a normalized three-boson plane wave onto x1=x2.

    Rows label the spectator momentum. This construction does not use the
    creation/annihilation implementation in fock_matrix.
    """
    modes = list(range(-cutoff, cutoff+1))
    basis = [b for b in itertools.combinations_with_replacement(modes, 3) if sum(b)==0]
    C = np.zeros((len(modes),len(basis)))
    for col,b in enumerate(basis):
        counts = {m:b.count(m) for m in set(b)}
        normalization = math.sqrt(math.prod(math.factorial(v) for v in counts.values())/6)
        for u in counts:
            denominator = math.prod(math.factorial(v-(m==u)) for m,v in counts.items())
            C[u+cutoff,col] = normalization*2/denominator
    diagonal = np.array([sum((2*math.pi*m/L)**2 for m in b) for b in basis])
    return C,diagonal


def winding_resolvent(c, s, cutoff=None, parity=0):
    kappa = math.sqrt(s / 2)
    if cutoff is None:
        a = kappa * L / 2
        if parity == 0:
            S = 1 / math.tanh(a)
            Sp = -L / (2 * math.sinh(a)**2)
        else:
            S = math.tanh(a)
            Sp = L / (2 * math.cosh(a)**2)
    else:
        S = math.fsum(((-1)**(parity*w)) * math.exp(-kappa * L * abs(w))
                      for w in range(-cutoff, cutoff + 1))
        Sp = math.fsum(-L * abs(w) * ((-1)**(parity*w)) * math.exp(-kappa*L*abs(w))
                       for w in range(-cutoff, cutoff + 1))
    D = 1 + c * S / (2 * kappa)
    derivative = c * Sp / (8 * kappa*kappa) - c * S / (8 * kappa**3)
    return derivative / D


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    data = dict(convention="L=1, hbar=2m=1, H=-sum d_j^2+2c sum delta_L",
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              scipy=scipy.__version__, matplotlib=matplotlib.__version__),
                free=[], contact_derivative=[], diagonalization=[], bethe=[], resolvent=[],
                collision_operator=[])
    for fermion in (False, True):
        for n in (1, 2, 3, 4):
            for beta in (.005, .02, .05):
                a = canonical_cycles(n, beta, fermion)
                b = canonical_occupations(n, beta, fermion)
                data["free"].append(dict(statistics="fermion" if fermion else "boson",
                    n=n, beta=beta, winding_cycle=a, occupations=b,
                    absolute_error=abs(a-b), relative_error=abs(a-b)/b,
                    mode_cutoff=40, winding_cutoff=40))
    assert max(r["relative_error"] for r in data["free"]) < 1e-8

    for beta in (.02, .05, .1):
        z0 = two_trace(0, beta)
        h = .001
        d1 = (two_trace(h, beta)-z0)/h
        d2 = (two_trace(h/2, beta)-z0)/(h/2)
        estimate = 2*d2-d1
        exact = -2 * beta / L * one_trace(beta)**2
        data["contact_derivative"].append(dict(beta=beta, h=h,
            first_contact=exact, finite_difference_extrapolation=estimate,
            absolute_error=abs(exact-estimate), momentum_cutoff=40, relative_mode_cutoff=40))
    assert max(r["absolute_error"] for r in data["contact_derivative"]) < 2e-8

    # Two-body full momentum blocks, and a genuinely three-body block.
    for n, momentum, cutoffs in ((2, 0, (8,16,32)), (2, 1, (8,16,32)),
                                (3, 0, (6,12,24))):
        for c in (.1, 1., 10.):
            if n == 2:
                exact = [((2*math.pi*momentum/L)**2/2 + 2*relative_q(c,j,momentum%2)**2)
                         for j in range(4)]
                reference = "Robin boundary roots"
            else:
                states = [bethe(I, c) for I in itertools.combinations(range(-6,7), 3)
                          if sum(I) == 0]
                states.sort(key=lambda r:r["energy"])
                data["bethe"].extend(dict(n=n,c=c,**r) for r in states)
                exact = [r["energy"] for r in states[:4]]
                reference = "Bethe roots; quantum numbers -6 through 6, sum zero"
            energies = []
            for cutoff in cutoffs:
                mat = fock_matrix(n, cutoff, momentum, c)
                vals = eigh(mat, subset_by_index=(0,3), eigvals_only=True)
                energies.append(vals)
                data["diagonalization"].append(dict(n=n, momentum=momentum,c=c,
                    cutoff=cutoff,dimension=len(mat),energies=vals.tolist(),reference=reference,
                    reference_energies=exact,absolute_errors=np.abs(vals-exact).tolist()))
            assert np.all(np.array(energies[2]) <= np.array(energies[1])+1e-8)
            assert np.all(np.array(energies[1]) <= np.array(energies[0])+1e-8)
            assert np.all(np.array(energies[2]) >= np.array(exact)-1e-8)

    # The trace difference has a proved O(J^-3) spectral tail, independent of windings.
    J = 2048
    for c in (1.,10.):
        for parity in (0,1):
            ec = np.array([2*relative_q(c,j,parity)**2 for j in range(J+1)])
            e0 = np.array([2*relative_q(0,j,parity)**2 for j in range(J+1)])
            for s in (2.,8.,32.):
                # Stable subtraction avoids loss at large j.
                spectral = float(np.sum((e0-ec)/((ec+s)*(e0+s))))
                exact = winding_resolvent(c,s,parity=parity)
                tail = c * L**3/(48 * math.pi**4 * J**3)
                assert abs(exact-spectral) < tail+2e-12
                for W in (0,1,2,4,8,16,32,64):
                    approx = winding_resolvent(c,s,W,parity)
                    data["resolvent"].append(dict(c=c,s=s,parity=parity,winding_cutoff=W,
                        spectral_cutoff=J,spectral=spectral,exact=exact,net=approx,
                        net_error=abs(approx-exact),spectral_error=abs(spectral-exact),
                        spectral_tail_bound=tail))
    # Exact finite-cutoff collision reduction, including an independent factor check.
    for M in (6,12,24,48):
        C, diagonal = three_body_contact(M)
        c,s = 1.,2.
        alpha = 6*c/L
        mat = fock_matrix(3,M,c=c)
        factor_error = float(np.max(np.abs(mat-np.diag(diagonal)-alpha*C.T@C)))
        assert factor_error < 1e-10
        tick = time.perf_counter()
        eigenvalues = eigh(mat,eigvals_only=True)
        full_time = time.perf_counter()-tick
        full_logdet = float(np.log(eigenvalues+s).sum()-np.log(diagonal+s).sum())
        full_resolvent = float(np.sum(1/(eigenvalues+s)-1/(diagonal+s)))
        tick = time.perf_counter()
        B = alpha * (C/(diagonal+s))@C.T
        Bprime = -alpha * (C/(diagonal+s)**2)@C.T
        chol = cholesky(np.eye(len(C))+B,lower=True)
        boundary_logdet = 2*np.log(np.diag(chol)).sum()
        boundary_resolvent = float(np.trace(cho_solve((chol,True),Bprime)))
        boundary_time = time.perf_counter()-tick
        assert abs(full_logdet-boundary_logdet) < 2e-8
        assert abs(full_resolvent-boundary_resolvent) < 2e-9
        data["collision_operator"].append(dict(c=c,s=s,cutoff=M,
            hilbert_dimension=len(mat),collision_dimension=len(C),factor_error=factor_error,
            full_logdet=full_logdet,collision_logdet=float(boundary_logdet),
            logdet_error=abs(full_logdet-boundary_logdet),full_resolvent=full_resolvent,
            collision_resolvent=boundary_resolvent,resolvent_error=abs(full_resolvent-boundary_resolvent),
            full_eigensolve_seconds=full_time,collision_calculation_seconds=boundary_time))
    # Ground-state curve shows why lengths alone cannot encode the interaction.
    cs = np.geomspace(.01,1000,121)
    data["ground_curve"] = [dict(c=float(c),energy=2*relative_q(c,0)**2) for c in cs]
    data["elapsed_seconds"] = time.perf_counter()-started
    (OUT/"results.json").write_text(json.dumps(data,indent=2)+"\n")

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "mathtext.fontset":"dejavusans","axes.spines.top":False,
                         "axes.spines.right":False})
    fig, axs = plt.subplots(1,3,figsize=(14,4.2),layout="constrained")
    axs[0].semilogx(cs,[r["energy"] for r in data["ground_curve"]],color="#006d77",lw=2)
    axs[0].axhline(2*math.pi**2,color="#ad6734",ls="--",label=r"$2\pi^2$")
    axs[0].set(xlabel=r"$c$",ylabel=r"$E_0$",title="Same circle, changing spectrum")
    axs[0].legend(frameon=False)
    for c in (1.,10.):
        rows=[r for r in data["diagonalization"] if r["n"]==3 and r["c"]==c]
        axs[1].loglog([r["cutoff"] for r in rows],[r["absolute_errors"][0] for r in rows],
                     "o-",label=rf"$c={c:g}$")
    axs[1].set(xlabel=r"$M$",ylabel=r"$|E_{0,M}-E_{0,\mathrm{Bethe}}|$",
               title="Three particles: independent matrices")
    axs[1].legend(frameon=False)
    for s in (2.,8.,32.):
        rows=[r for r in data["resolvent"] if r["c"]==1 and r["parity"]==0 and r["s"]==s]
        axs[2].semilogy([r["winding_cutoff"] for r in rows],
                        [max(r["net_error"],1e-17) for r in rows],"o-",label=rf"$s={s:g}$")
    axs[2].set(xlabel=r"$W$",ylabel=r"$|R_W(s)-R(s)|$",title="Winding sums converge off the spectrum")
    axs[2].legend(frameon=False)
    fig.savefig(OUT/"evidence.svg")
    svg = OUT/"evidence.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    fig.savefig(OUT/"evidence.png",dpi=170)
    print(json.dumps(dict(free_max_relative_error=max(r["relative_error"] for r in data["free"]),
          contact_max_error=max(r["absolute_error"] for r in data["contact_derivative"]),
          bethe_max_residual=max(r["residual"] for r in data["bethe"]),
          collision_max_resolvent_error=max(r["resolvent_error"] for r in data["collision_operator"]),
          seconds=data["elapsed_seconds"]),indent=2))


if __name__ == "__main__":
    main()
