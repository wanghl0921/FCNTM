"""Retain original FCNTM figure families and palette; fit 8–10 pt type at 31 pica.
Concept figures preserve their original nodes/semantics, not rejected preview designs.
"""
from pathlib import Path
import sys,json,csv,importlib.util,argparse
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'work/scipilot_deps'))
sys.path.insert(0,str(Path.home()/'.codex/skills/scipilot-figure-skill/scripts'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch,Circle
from matplotlib.path import Path as PlotPath
from visual_qa import audit_layout
from export_figure import export_figure
SRC=Path(r'C:/Users/20102/Desktop/FCNTM/AppliedIntelligence_submission/figures_src')
spec=importlib.util.spec_from_file_location('source_original',SRC/'make_orphan_figures.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
P=argparse.ArgumentParser();P.add_argument('--export',action='store_true');args=P.parse_args()
OUT=ROOT/'resized_figures';OUT.mkdir(exist_ok=True)
WIDTH=372/72.27;MM=1/25.4;WMM=WIDTH*25.4
INK='#1C1C1C';SUB='#58616D';HAIR='#C6CBD2';SHADOW='#E5E7EA'
BG='#3F5E86';BGF='#E9EFF6';SM='#3F7A63';SMF='#E6F1EC';FUS='#B0803F';FUSF='#F7EFE1';DEC='#4A5260';DECF='#ECEEF2';MAROON='#8C3A3A'
AUDIT={};DATA={};FONT_SIZES={}
def style():
 plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9.5,
  'axes.labelsize':10,'axes.titlesize':10,'xtick.labelsize':9,'ytick.labelsize':9,
  'legend.fontsize':9,'pdf.fonttype':42,'svg.fonttype':'none','savefig.bbox':None,
  'figure.constrained_layout.use':False,'axes.linewidth':.65,'xtick.minor.visible':False,
  'ytick.minor.visible':False,'text.color':INK,'xtick.direction':'out','ytick.direction':'out'})
style()
def text(ax,x,y,s,size=9.5,weight='normal',ha='center',color=INK):
 return ax.text(x,y,s,fontsize=size,fontweight=weight,ha=ha,va='center',color=color,zorder=6,linespacing=1.18)
def rect(ax,x,y,w,h,fill='white',edge=HAIR,accent=None,dash=False,shadow=False):
 if shadow:ax.add_patch(Rectangle((x+.45,y-.5),w,h,fc=SHADOW,ec='none',zorder=1))
 ax.add_patch(Rectangle((x,y),w,h,fc=fill,ec=edge,lw=.7,ls=(0,(3,2)) if dash else '-',zorder=2))
 if accent:ax.add_patch(Rectangle((x,y+h-.65),w,.65,fc=accent,ec='none',zorder=3))
def arrow(ax,points,dash=False,col='#78808A',lw=.9):
 path=PlotPath(points,[PlotPath.MOVETO]+[PlotPath.LINETO]*(len(points)-1))
 ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=7,color=col,lw=lw,
                            linestyle=(0,(3,2)) if dash else '-',zorder=4))
def canvas(h):
 style();fig=plt.figure(figsize=(WIDTH,h*MM));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,WMM),ylim=(0,h));ax.axis('off');return fig,ax
def finish(fig,i):
 from matplotlib.text import Text
 sizes=[x.get_fontsize() for x in fig.findobj(Text) if x.get_visible() and x.get_text().strip()]
 FONT_SIZES[f'Fig{i}']={'minimum_base_pt':min(sizes),'maximum_base_pt':max(sizes),'width_mm':WMM,'height_mm':fig.get_size_inches()[1]*25.4}
 issues=audit_layout(fig);AUDIT[f'Fig{i}']=issues
 fig.savefig(OUT/f'Fig{i}_preview.png',dpi=200)
 if args.export:
  assert not [x for x in issues if x[0]=='FAIL'],issues
  export_figure(fig,str(OUT/f'Fig{i}'),formats=['pdf','svg','png'],dpi=600,
                size_inches=tuple(fig.get_size_inches()),tight=False)
 plt.close(fig)
def panel(ax,x,n,title):
 rect(ax,x,2,40,108,edge=HAIR,accent=DEC,shadow=True)
 ax.add_patch(Circle((x+4,105),2.7,fc=DEC,ec='none',zorder=4))
 text(ax,x+4,105,str(n),9,'bold',color='white')
 text(ax,x+9,105,title,8,'bold',ha='left')
 ax.plot([x+2,x+38],[99.5,99.5],color=HAIR,lw=.6)
def figure1():
 fig,ax=canvas(113)
 panel(ax,2,1,'THE DATA\nREALITY');panel(ax,45,2,'FIELD-COOPERATIVE\nMODELLING');panel(ax,88,3,'ONE MODEL,\nTHREE REGIMES')
 text(ax,22,94,'OpenAlex civil-engineering\ncorpus',8.5,'bold')
 for y,miss in [(66,False),(38,True)]:
  rect(ax,5,y,34,22,dash=miss)
  text(ax,22,y+17.7,'Title-only record' if miss else 'Record with abstract',8.5,'bold')
  rect(ax,8,y+9.3,28,4.5,SMF,edge='none');text(ax,10,y+11.5,'Title / summary',8,ha='left',color=SM)
  rect(ax,8,y+2.6,28,4.5,'white' if miss else BGF,edge=HAIR if miss else 'none',dash=miss)
  text(ax,10,y+4.8,'Abstract missing' if miss else 'Abstract / background',8,ha='left',color=SUB if miss else BG)
 text(ax,22,21,'Missing fields change\nthe observation, not\nthe modelling goal.',8.5,'bold')
 # Original two chips, gated fusion and shared decoder, with vertical spacing.
 text(ax,55,93,'Background\nstream',8,color=SUB);text(ax,75,93,'Summary\nstream',8,color=SUB)
 rect(ax,48,77,14,9,BGF,BG,shadow=True);text(ax,55,81.5,r'$\theta_{\rm bg}$',10)
 rect(ax,68,77,14,9,SMF,SM,shadow=True);text(ax,75,81.5,r'$\theta_{\rm sm}$',10)
 ax.add_patch(Circle((62,77),1.7,fc='white',ec=MAROON,lw=.8,zorder=5));text(ax,62,77,r'$\oslash$',9,color=MAROON)
 arrow(ax,[(55,77),(55,73.5),(65,73.5),(65,71)])
 arrow(ax,[(75,77),(75,73.5),(65,73.5)])
 rect(ax,53.5,56,23,15,FUSF,accent=FUS,shadow=True)
 text(ax,65,67.3,'Gated fusion',8.7,'bold');text(ax,65,62.3,r'$\theta^{\rm fuse}$',9.5);text(ax,65,58.3,r'$g=\sigma(w)$',8.5,color=SUB)
 arrow(ax,[(65,56),(65,50.5)])
 rect(ax,53.5,35.5,23,15,DECF,accent=DEC,shadow=True)
 text(ax,65,45.8,'Shared decoder',8.5,'bold');text(ax,65,40,r'same $W$',9,color=SUB)
 text(ax,65,29,'One shared vocabulary',8,color=SUB)
 text(ax,65,17,'Field masking:\ndrop a side,\nkeep the model.',8.5,color=SUB)
 cases=[(74,'Complete record',r'$\theta_{\rm sm},\ \theta_{\rm bg}$',BG),
        (50,'Abstract missing',r'$\theta_{\rm sm}$ + prior fallback',FUS),
        (26,'Title-only record',r'$\theta_{\rm sm}$ only',SM)]
 for y,title,sub,col in cases:
  rect(ax,91,y,34,20);ax.add_patch(Rectangle((91,y),.7,20,fc=col,ec='none',zorder=3))
  text(ax,108,y+15.5,title,8.5,'bold');text(ax,108,y+9.6,sub,8.5);text(ax,108,y+4,'Same topic space',8,color=SUB)
 text(ax,108,22,'Unified read-out',8,'bold',color=DEC);text(ax,108,17.5,r'$\beta$ / class TF–IDF',8)
 rect(ax,91,3,34,10,BGF,accent=BG,shadow=True);text(ax,108,8,'Screening 8,000\ncivil articles',8.5,'bold')
 finish(fig,1)
def trapezoid(ax,x,y,w,h,col,fill,title,sub):
 pts=[(x,y),(x+w,y+2),(x+w,y+h-2),(x,y+h)]
 ax.add_patch(plt.Polygon([(a+.4,b-.5) for a,b in pts],fc=SHADOW,ec='none',zorder=1))
 ax.add_patch(plt.Polygon(pts,fc=fill,ec=HAIR,lw=.7,zorder=2))
 ax.add_patch(Rectangle((x,y),.7,h,fc=col,ec='none',zorder=3))
 text(ax,x+w/2,y+h*.67,title,8.5,'bold');text(ax,x+w/2,y+h*.22,sub,8.5)
def figure2():
 fig,ax=canvas(143)
 text(ax,3,137,'FIELD-SPECIFIC ENCODING AND FUSION',9,'bold',ha='left',color=SUB)
 ax.plot([3,127],[133,133],color=HAIR,lw=.6)
 text(ax,3,129,'Absent background stream: prior fallback',8,ha='left',color=SUB)
 for y,title,sy,col,fill in [(107,'Background','bg',BG,BGF),(81,'Summary','sm',SM,SMF)]:
  rect(ax,3,y,23,18,fill,accent=col,shadow=True)
  text(ax,14.5,y+12,title+'\nfield',8.5,'bold');text(ax,14.5,y+3.8,r'$x_{\rm '+sy+'}$',9)
  trapezoid(ax,34,y+1,30,16,col,fill,title+'\nencoder',r'$q_{\rm '+sy+r'}(z\mid x_{\rm '+sy+'})$')
  rect(ax,74,y+4,15,10,FUSF,FUS,shadow=True);text(ax,81.5,y+9,r'$\theta_{\rm '+sy+'}$',10)
  arrow(ax,[(26,y+9),(34,y+9)]);arrow(ax,[(64,y+9),(74,y+9)])
  arrow(ax,[(89,y+9),(99,y+9)])
 rect(ax,99,82,28,41,FUSF,accent=FUS,shadow=True)
 text(ax,113,117.5,'Gated fusion',9,'bold')
 text(ax,113,111.5,r'$\theta^{\rm fuse}=g\theta_{\rm bg}$',9)
 text(ax,113,106.5,r'$+(1-g)\theta_{\rm sm}$',9)
 text(ax,113,100.5,r'$g=\sigma(w),\ w\in\mathbb{R}$',8.5,color=SUB)
 text(ax,113,94,'Background missing\nor dropped:',8,color=SUB)
 text(ax,113,86.5,r'$\theta^{\rm fuse}=\theta_{\rm sm}$',9,color=SUB)
 # Original per-stream paths share W, now routed around the expanded nodes.
 arrow(ax,[(81.5,121),(81.5,131),(130,131),(130,31),(80,31),(80,42)],dash=True)
 arrow(ax,[(81.5,85),(81.5,77),(44,77),(44,52),(51,52)],dash=True)
 text(ax,3,71,'SHARED DECODING',9,'bold',ha='left',color=SUB)
 ax.plot([3,40],[67,67],color=HAIR,lw=.6)
 arrow(ax,[(113,82),(113,65),(69,65),(69,62)])
 rect(ax,51,42,36,20,DECF,accent=DEC,shadow=True)
 text(ax,69,56.5,'Shared decoder',9,'bold');text(ax,69,51,r'same $W$',9);text(ax,69,46,'BN + softmax',8.5,color=SUB)
 arrow(ax,[(87,52),(96,52)])
 rect(ax,96,42,31,20,DECF,accent=DEC,shadow=True)
 text(ax,111.5,55.5,'Cross-field\nreconstruction',8.5,'bold');text(ax,111.5,46,'All present fields',8)
 arrow(ax,[(3,54),(10,54)]);text(ax,13,54,'Fused path',8,ha='left')
 arrow(ax,[(3,47),(10,47)],dash=True);text(ax,13,47,'Per-stream decode',8,ha='left')
 text(ax,3,37,'All paths reuse one W.',8,ha='left',color=SUB)
 rect(ax,3,2,124,26,'#F7F8FA',accent=DEC)
 text(ax,65,24,'Training objective',9,'bold')
 text(ax,65,18,r'$\mathcal{L}_d=\frac{1}{|\mathcal{A}_d|}\sum_{f\in\mathcal{A}_d}[R_f+\lambda_{\rm KL}(t)\mathrm{KL}_f]+\lambda_fR_{\rm fuse}+\lambda_{\rm JS}\mathrm{JS}$',8.7)
 text(ax,65,10.8,r'$\mathcal{L}=|B|^{-1}\sum_{d\in B}\mathcal{L}_d+\lambda_{\rm orth}\Omega$',8.7)
 text(ax,65,5,'JS: deterministic topic proportions; both streams must be active.',8,color=SUB)
 finish(fig,2)
def axes_style(ax,grid=False):
 ax.tick_params(labelsize=10,length=2.5,width=.65)
 for s in ax.spines.values():s.set_color('#3A3F45');s.set_linewidth(.65)
 if grid:ax.grid(color='#C9CDD2',lw=.5);ax.set_axisbelow(True)
def figure3():
 style();d=old.fig3_data();DATA['Fig3']={c:{'ctfidf':a.tolist(),'beta':b.tolist()} for c,(a,b) in d.items()}
 fig,axs=plt.subplots(1,3,figsize=(WIDTH,81*MM));fig.subplots_adjust(left=.11,right=.98,bottom=.30,top=.73,wspace=.48)
 panels=[('civil','Civil (OpenAlex)\n6,000/2,000',(-.012,.0896),[0,.02,.04,.06,.08]),('github','GitHub issues\n6,000/2,000',(-.0421,.0656),[-.04,-.02,0,.02,.04,.06]),('agnews','AG News\n6,000/7,600',(-.0951,.0499),[-.08,-.04,0,.04])]
 for ax,(c,title,lim,ticks) in zip(axs,panels):
  a,b=d[c];x=np.arange(1,11);ax.axhline(0,color=INK,lw=.65,ls=(0,(4,2.4)))
  ax.plot(x,a,'o',ms=3.8,mfc=BG,mec=BG,ls='none',label='Class TF–IDF read-out')
  ax.plot(x,b,'s',ms=3.8,mfc='white',mec=FUS,ls='none',label=r'$\beta$ read-out')
  ax.set(xlim=(.4,10.8),ylim=lim,xticks=[1,3,5,7,10],yticks=ticks)
  ax.set_title(title,fontsize=9.5,pad=15)
  ax.text(.03,1.025,f'{a.mean():+.3f}',transform=ax.transAxes,color=BG,fontsize=9,ha='left')
  ax.text(.97,1.025,f'{b.mean():+.3f}',transform=ax.transAxes,color=FUS,fontsize=9,ha='right')
  ax.set_xlabel('Paired seed\n(ordered)',fontsize=9.5);axes_style(ax)
 axs[0].set_ylabel(r'$\Delta C_V$: FC-NTM − BTM',fontsize=10)
 h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,frameon=False,fontsize=9.5,bbox_to_anchor=(.5,.015))
 finish(fig,3)
def figure4():
 style();d=old.fig4_data();DATA['Fig4']=d
 fig,axs=plt.subplots(2,1,figsize=(WIDTH,129*MM),sharex=True)
 fig.subplots_adjust(left=.13,right=.98,bottom=.25,top=.92,hspace=.30)
 x=np.arange(11);width=.36
 for ax,(mi,si,lim,lab) in zip(axs,[(0,1,(0,.79),r'$C_V$ coherence'),(2,3,(-.105,.32),'NPMI coherence')]):
  for offset,c,col in [(-.19,'github',BG),(.19,'agnews',FUS)]:
   vals=[d[c][m][mi] for m in old.METHODS];sd=[d[c][m][si] for m in old.METHODS]
   ax.bar(x+offset,vals,width=width,color=col,zorder=3)
   ax.errorbar(x+offset,vals,yerr=sd,fmt='none',ecolor=INK,elinewidth=.7,capsize=2,zorder=4)
  ax.set_ylim(lim);ax.set_ylabel(lab,fontsize=10);axes_style(ax)
 axs[1].set_xticks(x);axs[1].set_xticklabels(old.METHODS,rotation=90,ha='right',rotation_mode='anchor',fontsize=9)
 fig.legend([Rectangle((0,0),1,1,fc=BG),Rectangle((0,0),1,1,fc=FUS)],['GitHub issues','AG News'],loc='upper center',ncol=2,frameon=False,fontsize=9.5)
 finish(fig,4)
def figure5():
 style();rows=list(csv.DictReader((ROOT.parent/'figure_preview/data/fig5_readouts.csv').open(encoding='utf-8')))
 models=['FC-NTM','BTM','ETM','NMF','ProdLDA','CTM-lite','LDA'];cols=[(c,ro) for c in ['github','agnews','civil'] for ro in ['beta','ctfidf']]
 fig,axs=plt.subplots(2,1,figsize=(WIDTH,137*MM));fig.subplots_adjust(left=.16,right=.89,bottom=.12,top=.94,hspace=.62)
 for ax,metric,title in zip(axs,['CV','NPMI'],[r'$C_V$ coherence','NPMI coherence']):
  mat=np.array([[float(next(r for r in rows if (r['model'],r['corpus'],r['readout'],r['metric'])==(m,c,ro,metric))['mean']) for c,ro in cols] for m in models]);DATA['Fig5_'+metric]=mat.tolist()
  lo,hi=mat.min()-.02,mat.max()+.02;im=ax.imshow(mat,cmap='YlGnBu',vmin=lo,vmax=hi,aspect='auto')
  ax.set_xticks(range(6)); ax.set_xticklabels(['GitHub\nβ','GitHub\ncTF','AG News\nβ','AG News\ncTF','Civil\nβ','Civil\ncTF'],fontsize=9,rotation=25,ha='right')
  ax.set_yticks(range(7),models,fontsize=9.5);ax.tick_params(which='both',length=0)
  ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,7,1),minor=True);ax.grid(which='minor',color='white',lw=1.2)
  for s in ax.spines.values():s.set_visible(False)
  for i in range(7):
   for j in range(6):
    rgba=im.cmap(im.norm(mat[i,j]));rgb=np.array(rgba[:3]);lin=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4);lum=lin@[.2126,.7152,.0722]
    ink='white' if 1.05/(lum+.05)>(lum+.05)/.05 else 'black'
    ax.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',fontsize=9.5,color=ink,fontweight='bold' if i==mat[:,j].argmax() else 'normal')
  ax.add_patch(Rectangle((-.5,-.5),6,1,fill=False,ec=MAROON,lw=1.1,zorder=10,clip_on=False));ax.set_title(title,fontsize=11,fontweight='bold',pad=8)
  pos=ax.get_position();cax=fig.add_axes([.91,pos.y0,.025,pos.height]);cb=fig.colorbar(im,cax=cax);cb.ax.tick_params(labelsize=10,length=0);cb.outline.set_visible(False)
 finish(fig,5)
def figure6():
 style();fig,axs=plt.subplots(1,3,figsize=(WIDTH,85*MM));fig.subplots_adjust(left=.11,right=.98,bottom=.31,top=.76,wspace=.72)
 X=np.array([0,1]);cols=[BG,SM];names=['GitHub','AG News']
 ax=axs[0];ax.bar(X,[.306,.245],width=.5,color=cols,zorder=3)
 for x,v in zip(X,[.306,.245]):ax.text(x,v+.015,f'{v:.3f}',ha='center',fontsize=9.5)
 ax.set(ylim=(0,.40),xticks=X,xticklabels=names,yticks=[0,.1,.2,.3,.4]);ax.set_ylabel(r'Held-out $\rho^2_{\rm cca}$',fontsize=10)
 ax=axs[1];ax.axhline(0,color=HAIR,lw=.7)
 for x,mu,sd,col in zip(X,[.009,.018],[.022,.013],cols):
  ax.errorbar(x,mu,yerr=sd,fmt='o',ms=4,color=col,capsize=3,elinewidth=.9);ax.text(x,.040,f'{mu:+.3f}\n±{sd:.3f}',ha='center',fontsize=9)
 ax.set(xlim=(-.6,1.6),ylim=(-.035,.061),xticks=X,xticklabels=names,yticks=[-.03,0,.03]);ax.set_ylabel(r'Occlusion $\Delta C_V$',fontsize=10)
 ax=axs[2];a=[.027,.019];b=[.028,-.009]
 ax.axhline(0,color=HAIR,lw=.7)
 ax.bar(X-.15,a,width=.24,color=cols,label='3 seeds');ax.bar(X+.15,b,width=.24,color=cols,alpha=.5,label='10 seeds')
 for x,v in zip(X-.15,a):ax.text(x,v+.002,f'{v:+.3f}',ha='center',fontsize=9)
 for x,v in zip(X+.15,b):ax.text(x,v+(.013 if v>0 else -.004),f'{v:+.3f}',ha='center',va='bottom' if v>0 else 'top',fontsize=9)
 ax.set(xlim=(-.8,1.8),ylim=(-.03,.063),xticks=X,xticklabels=names,yticks=[-.02,0,.02,.04,.06]);ax.set_ylabel(r'$C_V$ gap, cTFIDF − BTM',fontsize=9.5)
 ax.legend(loc='upper center',bbox_to_anchor=(.5,-.34),ncol=1,frameon=False,fontsize=9)
 for ax,title in zip(axs,['HELD-OUT\nREDUNDANCY','OCCLUSION\nDIAGNOSTIC',r'$C_V$ GAP'+'\nVS. BTM']):
  ax.set_title(title,fontsize=9.5,fontweight='bold',color=SUB,pad=12)
  ax.tick_params(axis='x',rotation=35);axes_style(ax);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#EDEFF2',lw=.5,zorder=0);ax.set_axisbelow(True)
 DATA['Fig6']={'redundancy':[.306,.245],'occlusion_mean':[.009,.018],'occlusion_sd':[.022,.013],'gap3':a,'gap10':b}
 finish(fig,6)
def figure7():
 style();d=old.fig7_data();DATA['Fig7']=d;fig,axs=plt.subplots(1,2,figsize=(WIDTH,98*MM));fig.subplots_adjust(left=.12,right=.98,bottom=.40,top=.95,wspace=.42)
 for m in old.MODELS7:
  y=np.array([d[m][p] for p in old.RATES]);st=old.MODEL_STYLE[m]
  for ax,v in zip(axs,[y,y/y[0]]):ax.plot(old.RATES,v,marker='o',ms=3.5,lw=1.1,label=m,**st)
 axs[0].set(ylim=(6.44,10.45),yticks=np.arange(6.5,10.1,.5));axs[0].set_ylabel('NLL of masked tokens',fontsize=10)
 axs[1].set(ylim=(.975,1.5),yticks=np.arange(1,1.5,.1));axs[1].set_ylabel('Relative degradation\nNLL(p) / NLL(0.25)',fontsize=10)
 for ax in axs:ax.set(xlim=(.2125,1.0375),xticks=old.RATES);ax.set_xlabel('Fraction of background-field\ntokens masked',fontsize=9.5);axes_style(ax,True)
 h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,fontsize=9.5,bbox_to_anchor=(.5,.015),handlelength=2.5)
 finish(fig,7)
def figure8():
 style();d=old.fig8_data();DATA['Fig8']=d;fig,axs=plt.subplots(1,3,figsize=(WIDTH,102*MM),sharey=True);fig.subplots_adjust(left=.15,right=.98,bottom=.42,top=.82,wspace=.25)
 for ax,c,title,ratio in zip(axs,['github','civil','agnews'],['GitHub issues','Civil literature','AG News'],[12.9,11.2,3.9]):
  for m in old.MODELS8:ax.plot(old.RATES,[d[c][m][p] for p in old.RATES],marker='o',ms=3.5,lw=1.1,label=m,**old.MODEL_STYLE[m])
  ax.set(xlim=(.2125,1.0375),ylim=(.97,1.55),xticks=old.RATES,yticks=[1,1.1,1.2,1.3,1.4,1.5]);ax.set_xticklabels(['.25','.50','.75','1.00'],fontsize=9)
  ax.set_title(title+' (10 seeds)\n'+f'bg/summary ratio ≈ {ratio}',fontsize=9.5,pad=8)
  ax.set_xlabel('Fraction of background-\nfield tokens masked',fontsize=9.5);axes_style(ax,True)
 axs[0].set_ylabel('Relative degradation\nNLL(p) / NLL(0.25)',fontsize=10)
 h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,fontsize=9.5,bbox_to_anchor=(.5,.015),handlelength=2.5)
 finish(fig,8)
for fn in [figure1,figure2,figure3,figure4,figure5,figure6,figure7,figure8]:fn()
(OUT/'layout_audit.json').write_text(json.dumps(AUDIT,indent=2),'utf-8')
(OUT/'plotted_data.json').write_text(json.dumps(DATA,indent=2),'utf-8')
(OUT/'base_font_sizes.json').write_text(json.dumps(FONT_SIZES,indent=2),'utf-8')
print(json.dumps(AUDIT,indent=2))
