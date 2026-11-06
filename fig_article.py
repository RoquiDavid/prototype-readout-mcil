# Figures 2 and 3.
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size':8})
B=[500,1000,2000]
D={'MELD':{'ER':([37.47,38.46,38.56],[1.66,2.47,1.81]),'DER++':([37.68,39.26,40.72],[1.75,1.74,0.61]),'ours':(41.40,1.60)},
   'IEMOCAP':{'ER':([52.48,53.15,53.17],[1.17,1.17,0.56]),'DER++':([52.40,53.44,53.51],[0.77,1.11,0.68]),'ours':(49.50,1.13)}}
fig,axs=plt.subplots(1,2,figsize=(7,2.6))
for ax,(ds,d) in zip(axs,D.items()):
    m,s=d['ours']
    ax.axhspan(m-s,m+s,color='#2f4b7c',alpha=0.15,lw=0)
    ax.axhline(m,color='#2f4b7c',lw=1.5,label='Ours (no memory)')
    ax.errorbar(B,d['ER'][0],yerr=d['ER'][1],marker='o',ms=4,capsize=2,color='#9e9e9e',label='ER')
    ax.errorbar(B,d['DER++'][0],yerr=d['DER++'][1],marker='s',ms=4,capsize=2,color='#d1495b',ls='--',label='DER++')
    ax.set_xscale('log'); ax.set_xticks(B); ax.set_xticklabels([str(b) for b in B]); ax.minorticks_off()
    ax.set_xlabel('Stored samples'); ax.set_title(ds,fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
axs[0].set_ylabel('Accuracy (%)'); axs[0].legend(frameon=False,fontsize=7,loc='lower right')
plt.tight_layout(); plt.savefig('figures/fig_buffer.pdf')
conds=['MELD\nno audio','MELD\nno text','IEMOCAP\nno audio','IEMOCAP\nno text']
vals={'Single set':[30.32,20.41,40.09,31.85],'Matched sets':[44.49,27.41,41.44,42.97],
      'Model trained on the\nremaining modality':[44.27,27.71,41.21,43.19],'ER':[34.60,20.12,48.57,31.47],'DER++':[32.71,20.25,48.79,32.32]}
cols=['#c7c7c7','#2f4b7c','#7aa6c2','#8c8c8c','#d1495b']; hatch=['','','//','','']
x=np.arange(4); w=0.16
fig,ax=plt.subplots(figsize=(7,2.8))
for k,(name,v) in enumerate(vals.items()):
    ax.bar(x+(k-2)*w,v,w,label=name,color=cols[k],hatch=hatch[k],edgecolor='white' if not hatch[k] else '#2f4b7c',lw=0.3)
ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0,58)
ax.legend(frameon=False,fontsize=6.5,ncol=5,loc='upper center',bbox_to_anchor=(0.5,1.13))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig('figures/fig_missing.pdf')
print('ok')
