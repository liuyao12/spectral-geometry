#!/usr/bin/env python3
"""Bundle the polygon experiments and draw independently traced representatives."""
from pathlib import Path
import json
import shutil
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'research-data/polygon-families'
WEB=ROOT/'docs/data/polygon-families'
NAMES=['rhombus_60','sheared_parallelogram','regular_pentagon','house_pentagon','square','rectangle','perturbed_pentagon']
LABELS={'rhombus_60':'Rhombus with equal sides','sheared_parallelogram':'Sheared parallelogram',
        'regular_pentagon':'Regular pentagon','house_pentagon':'House-shaped pentagon','square':'Square control',
        'rectangle':'Rectangle control','perturbed_pentagon':'Perturbed pentagon'}


def read(name):return json.loads((DATA/(name+'.json')).read_text())


def main():
    WEB.mkdir(parents=True,exist_ok=True)
    certificates=read('certificates');sweep=read('sweep')
    cases=[]
    for name in NAMES:
        base=read(name);embedded=read(name+'_embedded')
        witness=base['witnesses'][0] if base['witnesses'] else None
        embedded_witness=embedded['witnesses'][0] if embedded['witnesses'] else None
        if name=='rhombus_60':
            witness=dict(witness,frames=certificates['exact_rhombus_slice']['frames'],
                         sampled_parameters=certificates['exact_rhombus_slice']['sampled_parameters'])
            embedded_witness=dict(witness)
        cases.append(dict(id=name,label=LABELS[name],vertices=base['vertices'],witness=witness,
                          embedded_witness=embedded_witness,depth=base['max_reflections_per_non_direct_edge'],
                          embedded_depth=embedded['max_reflections_per_non_direct_edge'],
                          search_counters=base['counters'],embedded_search_counters=embedded['counters']))
    bundle=dict(cases=cases,certificate=certificates['regular_pentagon_interval_certificate'],
                parallelogram_sweep=[dict(shear=r['shear'],height=r['height'],
                                          found=bool(r['result']['witnesses'])) for r in sweep['parallelograms']],
                pentagon_perturbation=sweep['same_words_under_pentagon_perturbation'],
                scope='Free 120-degree junctions; ordinary specular reflections; one direct edge. The slider is an exact affine continuation of corresponding path vertices, on a fixed-itinerary chamber. Negative searches are bounded and are not global nonexistence proofs.')
    for case in cases:
        for witness in [case['witness'],case['embedded_witness']]:
            if witness is None:continue
            frames=witness['frames'];lengths=[f['total_length'] for f in frames]
            assert max(lengths)-min(lengths)<1e-10
            assert max(f['endpoint_error'] for f in frames)<1e-10
            assert max(f['terminal_balance_residual'] for f in frames)<1e-10
            assert min(f['corner_clearance'] for f in frames)>1e-6
    (DATA/'report.json').write_text(json.dumps(bundle,indent=2)+'\n')
    shutil.copy2(DATA/'report.json',WEB/'report.json')
    for name in ['certificates','sweep']:
        shutil.copy2(DATA/(name+'.json'),WEB/(name+'.json'))
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':11})
    fig,axes=plt.subplots(2,3,figsize=(12.8,8.0),layout='constrained')
    colors=['#bd6335','#007c83','#6876a4']
    for row,name in enumerate(['rhombus_60','regular_pentagon']):
        case=next(c for c in cases if c['id']==name)
        vertices=np.array(case['vertices']);closed=np.vstack([vertices,vertices[0]])
        for col,index in enumerate([0,4,8]):
            ax=axes[row,col];frame=case['witness']['frames'][index]
            ax.plot(closed[:,0],closed[:,1],color='#18353e',lw=1.5)
            for edge,path in enumerate(frame['paths']):
                p=np.array(path);ax.plot(p[:,0],p[:,1],color=colors[edge],lw=1.8)
                ax.scatter(p[1:-1,0],p[1:-1,1],color=colors[edge],s=8)
            u,v=np.array(frame['u']),np.array(frame['v'])
            ax.scatter([u[0],v[0]],[u[1],v[1]],color='#18353e',s=32,zorder=6)
            ax.text(u[0],u[1]+.05,r'$u$',ha='center');ax.text(v[0],v[1]-.08,r'$v$',ha='center')
            ax.set_aspect('equal');ax.axis('off')
            label='Crossing-free rhombus' if row==0 else 'Immersed regular pentagon'
            ax.set_title(label+'\n'+rf'$L={frame["total_length"]:.9f}$')
    fig.savefig(DATA/'families.png',dpi=170)
    fig.savefig(DATA/'families.svg')
    plt.close(fig)
    svg=DATA/'families.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    for name in ['families.png','families.svg']:shutil.copy2(DATA/name,WEB/name)
    summary=[]
    for case in cases:
        w=case['witness']
        summary.append(dict(shape=case['id'],found=w is not None,
                            total_length=w['frames'][0]['total_length'] if w else None,
                            crossings=len(w['frames'][0]['crossing_points']) if w else None,
                            embedded_found=case['embedded_witness'] is not None,
                            depth=case['depth'],embedded_depth=case['embedded_depth']))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
