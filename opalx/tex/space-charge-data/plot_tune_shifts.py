"""Plot differences of median NAFF tunes and KV tunes from their own zero-current baselines."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ffa-preliminary-mpl')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
OUT=BASE.parent/'figures'
s=pd.read_csv(BASE/'summary.csv');rows=[]
plt.rcParams.update({'font.size':11,'axes.labelsize':12,'savefig.dpi':300,'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
for axis in ['x','y']:
 g=s[s.axis==axis].sort_values('current_mA').copy();zero=g.loc[g.current_mA==0].iloc[0]
 # Subtract the zero-current median from every NAFF percentile. Bars describe
 # the shifted footprint, not propagated uncertainty or paired-particle shifts.
 g['delta_naff']=g['median']-zero['median'];g['delta_kv']=g['model']-zero['model']
 g['delta_naff_p05']=g.p05-zero['median'];g['delta_naff_p95']=g.p95-zero['median']
 rows.append(g[['axis','current_mA','delta_naff','delta_kv','delta_naff_p05','delta_naff_p95']])
 fig,ax=plt.subplots(figsize=(4.6,4.6),layout='constrained')
 ax.axhline(0,color='.6',linewidth=.7)
 ax.plot(g.current_mA,g.delta_kv,color='black',label='Linear KV model')
 ax.errorbar(g.current_mA,g.delta_naff,yerr=np.array([g.delta_naff-g.delta_naff_p05,g.delta_naff_p95-g.delta_naff]),fmt='o',capsize=4,label='NAFF, 15 turns (5–95%)')
 ax.set(xlabel='Bunch current (mA)',ylabel=r'$\Delta Q_'+axis+r'=Q_'+axis+r'(I)-Q_'+axis+r'(0)$')
 ax.grid(alpha=.2);ax.legend(fontsize=9)
 for ext in ['png','pdf','svg']:fig.savefig(OUT/('preliminary-dQ'+axis+'.'+ext))
 plt.close(fig)
pd.concat(rows).to_csv(BASE/'tune-shifts-from-zero.csv',index=False)
