#!/usr/bin/env python3
"""Plot the reproducible physical-chamber samples; no fitted parameters."""
from pathlib import Path
import json
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
data=json.loads((root/'research-data/intrinsic-rhombus-weights.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
rows=data['monte_carlo']
axes[0].errorbar([r['kappa'] for r in rows],[r['probability'] for r in rows],
                 yerr=[2*r['standard_error'] for r in rows],fmt='o-',color='#176b67',capsize=3)
axes[0].axhline(1/3,color='#aa542e',ls='--',label=r'Proved limit $1/3$')
axes[0].set(xscale='log',xlabel=r'$\kappa$',ylabel=r'$P_{\mathcal{D}}(\kappa)$',title='Length-three family: exact weight factor')
axes[0].legend(frameon=False)
rows=data['collapsed_chamber_monte_carlo']
axes[1].errorbar([r['kappa'] for r in rows],[r['scaled_weight'] for r in rows],
                 yerr=[2*r['standard_error'] for r in rows],fmt='o-',color='#176b67',capsize=3)
axes[1].axhline(1/32,color='#aa542e',ls='--',label=r'Proved limit $1/32$')
axes[1].set(xscale='log',xlabel=r'$\kappa$',ylabel=r'$\kappa^3 e^{\sqrt{3}\kappa}W_{\mathrm{coll}}(\kappa)$',title='Collapsed direct / bottom / top sector')
axes[1].legend(frameon=False)
for ax in axes:
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
target=root/'docs/assets/intrinsic-rhombus-weights.png'
target.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(target,dpi=170,facecolor='#fff')
print(target)
