"""Plot latest available record lengths without relabelling unfinished charged cases as complete."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ffa-latest-mpl')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;BASE=HERE;FIG=HERE.parent/'figures';s=pd.read_csv(BASE/'summary.csv')
plt.rcParams.update({'font.size':11,'axes.labelsize':12,'savefig.dpi':300,'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
rows=[]
for axis in ['x','y']:
 g=s[s.axis==axis].sort_values('current_mA').copy();zero=g[g.current_mA==0].iloc[0]
 g['delta_naff']=g['median']-zero['median'];g['delta_kv']=g.model-zero.model;g['delta_naff_p05']=g.p05-zero['median'];g['delta_naff_p95']=g.p95-zero['median'];rows.append(g[['axis','current_mA','turns','delta_naff','delta_kv','delta_naff_p05','delta_naff_p95']])
 for shift in [False,True]:
  med=g.delta_naff if shift else g['median'];model=g.delta_kv if shift else g.model;lo=g.delta_naff_p05 if shift else g.p05;hi=g.delta_naff_p95 if shift else g.p95
  fig,ax=plt.subplots(figsize=(4.6,4.6),layout='constrained');ax.plot(g.current_mA,model,'k-',label='Linear KV model')
  ax.errorbar(g.current_mA,med,yerr=np.array([med-lo,hi-med]),fmt='o',capsize=4,label=f'NAFF: zero {int(zero.turns)} turns\ncharged 15 turns (5–95%)')
  ax.set(xlabel='Bunch current (mA)',ylabel=('$\\Delta Q_'+axis+'$') if shift else ('$Q_'+axis+'$'));ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower left' if shift else 'upper right')
  for ext in ['png','pdf','svg']:fig.savefig(FIG/(('preliminary-dQ' if shift else 'latest-Q')+axis+'.'+ext))
  plt.close(fig)
pd.concat(rows).to_csv(BASE/'tune-shifts-from-zero.csv',index=False)
