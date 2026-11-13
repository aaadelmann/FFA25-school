"""Recreate the school FFA four-current footprint from the retained NAFF tables."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ffa-school-mpl')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
HERE=Path(__file__).resolve().parent;OUT=HERE.parent/'figures'
f=pd.read_csv(HERE/'particle-tunes.csv');s=pd.read_csv(HERE/'summary.csv');labels=pd.read_csv(HERE/'initial-action-fractions.csv').set_index('source_row')
plt.rcParams.update({'font.size':11,'axes.labelsize':12,'savefig.dpi':300,'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(5.4,4.8),layout='constrained');ax.set_box_aspect(1)
for n,(case,g) in enumerate(f.groupby('case',sort=False)):
 t=g.pivot(index='source_row',columns='axis',values='tune');v=labels.loc[t.index,'initial_horizontal_normalized_action_fraction']
 sc=ax.scatter(t.x,t.y,c=v,cmap='viridis',vmin=0,vmax=1,s=8,alpha=.8,linewidths=0,rasterized=True)
 model=s[s.case==case].set_index('axis');ax.plot(model.loc['x','model'],model.loc['y','model'],marker='x',color='#D55E00',ms=8,mew=1.6,ls='none',label='Analytic KV' if n==0 else None,zorder=5)
 offset=[(-64,16),(-65,15),(-65,15),(16,-35)][n]
 ax.annotate(f'{g.current_mA.iloc[0]:.3f} mA',(t.x.median(),t.y.median()),xytext=offset,textcoords='offset points',fontsize=9,ha='left',va='bottom',arrowprops={'arrowstyle':'-','color':'.45','lw':.6})
ax.set(xlabel='$Q_x$',ylabel='$Q_y$');ax.ticklabel_format(useOffset=False);ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(5));ax.margins(.16);ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=9,frameon=False)
cb=fig.colorbar(sc,ax=ax,fraction=.048,pad=.035,ticks=np.linspace(0,1,5));cb.set_label(r'Initial horizontal normalized-action fraction $f_x$')
OUT.mkdir(exist_ok=True)
for ext in ['pdf','png','svg']:fig.savefig(OUT/('tune-footprint.'+ext))
plt.close(fig)
