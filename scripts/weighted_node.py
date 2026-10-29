#!/usr/bin/env python3
"""Check unequal-angle stationary nodes in a bounded mathematical kernel model."""
from pathlib import Path
import math,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trivalent_toy import net_rectangle,spectral_graph

OUT=Path(__file__).resolve().parents[1]/'docs/data/weighted-node'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    H=3*math.sqrt(3)/4;beta=.12;results=[]
    for sigma in (.02,.03):
        r=sigma/beta;d=4*r/(1+2*r);A=2-d;ell=math.hypot(A,2*H)
        wm=ell/(2*beta);wn=d/(2*sigma)
        dn=math.degrees(math.acos(-A/ell))
        dd=math.degrees(math.acos((A*A-4*H*H)/(ell*ell)))
        geom,tail=net_rectangle('first_exchange',(beta,beta,sigma),1,H)
        spec=spectral_graph('first_exchange',(beta,beta,sigma),1)*spectral_graph('first_exchange',(beta,beta,sigma),H)
        balance=wn-2*wm*A/ell
        assert abs(balance)<1e-12 and abs(2*dn+dd-360)<1e-10
        assert abs(geom-spec)<1e-12
        results.append(dict(beta=beta,sigma=sigma,u=[(1-d)/2,H/2],v=[(1+d)/2,H/2],
            displacement_x=A,particle_edge_length=ell,kernel_edge_length=d,
            particle_edge_weight=wm,kernel_edge_weight=wn,angle_D_N_degrees=dn,
            angle_D_D_degrees=dd,balance_residual=balance,net_trace=geom,
            spectral_trace=spec,trace_error=abs(geom-spec),winding_tail_bound=tail))
    (OUT/'results.json').write_text(json.dumps(dict(
        model='H_g,sigma=-Delta_x^D-Delta_y^D+g K_sigma^N(x,y)',
        scope='Stationarity and first-exchange trace graph at fixed beta; not a full weighted-length trace theorem',
        rectangle_height='3 sqrt(3)/4',results=results),indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':12})
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for idx,(ax,row) in enumerate(zip(axs,results)):
        A=row['displacement_x'];ell=row['particle_edge_length']
        vectors=np.array([[-A/ell,2*H/ell],[-A/ell,-2*H/ell],[1.,0.]])
        for j,vec in enumerate(vectors):
            ax.annotate('',xy=vec,xytext=(0,0),arrowprops=dict(arrowstyle='->',lw=2.4,color='#c06d39' if j==2 else '#006d77'))
        ax.scatter([0],[0],c='#172f37',s=30)
        ax.text(-.37,0,rf'${row["angle_D_D_degrees"]:.2f}^\circ$',ha='center',va='center')
        ax.text(.1,.30,rf'${row["angle_D_N_degrees"]:.2f}^\circ$',ha='center')
        ax.text(.1,-.40,rf'${row["angle_D_N_degrees"]:.2f}^\circ$',ha='center')
        ax.set_title(r'$w_1=w_2=w_3$' if idx==0 else r'$(w_1,w_2,w_3)\propto(\sqrt{307},\sqrt{307},16)$')
        ax.set(xlim=(-1.1,1.15),ylim=(-1.03,1.03));ax.set_aspect('equal');ax.axis('off')
    fig.savefig(OUT/'angles.png',dpi=170);fig.savefig(OUT/'angles.svg')
    p=OUT/'angles.svg';p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
