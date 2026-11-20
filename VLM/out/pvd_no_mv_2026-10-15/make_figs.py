import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_curve, auc, confusion_matrix

FONT="/home/mzhao/华为项目/VLM/src/simhei中易黑体.ttf"
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"]=font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"]=False

RES="/home/mzhao/华为项目/VLM/out/pvd_no_mv_2026-10-15"
FIG="/home/mzhao/华为项目/VLM/figs"
os.makedirs(FIG, exist_ok=True)

items=[]
for fp in sorted(glob.glob(os.path.join(RES,"*.json"))):
    d=json.load(open(fp))
    if "results" not in d: continue
    yt,yp,pr=[],[],[]
    for r in d["results"]:
        lab=r["msg"]["conversations"][-1]["value"].strip()
        if lab not in ("是","否"): continue
        yt.append(1 if lab=="是" else 0)
        yp.append(1 if r["response"]=="是" else 0)
        pr.append(r["prob_y"])
    yt=np.array(yt); yp=np.array(yp); pr=np.array(pr)
    if len(yt)==0 or len(np.unique(yt))<2: continue
    items.append(dict(name=os.path.basename(fp)[:-5], yt=yt, yp=yp, pr=pr,
                      acc=(yt==yp).mean(),
                      f1=d["metrics"]["f1"], auc=d["metrics"]["auc"],
                      n=len(yt), npos=int(yt.sum())))

names=[x["name"] for x in items]
print("items:", len(items))

# 1) 每类指标条形图
x=np.arange(len(items)); w=0.2
fig,ax=plt.subplots(figsize=(max(14,len(items)*0.6),7))
ax.bar(x-1.5*w,[i["acc"] for i in items],w,label="Acc")
ax.bar(x-0.5*w,[i["f1"] for i in items],w,label="F1")
ax.bar(x+0.5*w,[i["auc"] for i in items],w,label="AUC")
ax.bar(x+1.5*w,[i.get("f1") for i in items],w,label="dummy",alpha=0)  # placeholder
ax.set_xticks(x); ax.set_xticklabels(names,rotation=60,ha="right",fontsize=9)
ax.set_ylim(0,1); ax.set_ylabel("分数"); ax.legend()
ax.set_title("各诊断项指标（Acc / F1 / AUC，阈值=0.5）")
ax.grid(True,axis="y",alpha=0.3); fig.tight_layout()
fig.savefig(os.path.join(FIG,"classification_metrics_by_diag.png"),dpi=200)
plt.close(fig)

# 2) ROC 网格
ncol=6; nrow=int(np.ceil(len(items)/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(3*ncol,3*nrow))
axes=np.atleast_1d(axes).ravel()
for ax,it in zip(axes,items):
    fpr,tpr,_=roc_curve(it["yt"],it["pr"]); a=auc(fpr,tpr)
    ax.plot(fpr,tpr,lw=1.5,label=f"AUC={a:.3f}")
    ax.plot([0,1],[0,1],"k--",lw=0.8)
    ax.set_title(it["name"],fontsize=10); ax.set_xlabel("FPR",fontsize=8); ax.set_ylabel("TPR",fontsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
for ax in axes[len(items):]: ax.axis("off")
fig.suptitle("各诊断项 ROC 曲线（正类=阳性）",fontsize=14)
fig.tight_layout(rect=[0,0,1,0.98])
fig.savefig(os.path.join(FIG,"roc_curves.png"),dpi=200)
plt.close(fig)

# 3) 混淆矩阵网格
fig,axes=plt.subplots(nrow,ncol,figsize=(2.6*ncol,2.4*nrow))
axes=np.atleast_1d(axes).ravel()
for ax,it in zip(axes,items):
    cm=confusion_matrix(it["yt"],it["yp"],labels=[0,1])
    im=ax.imshow(cm,cmap="Blues")
    for i,j in np.ndindex(cm.shape):
        ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                color="white" if cm[i,j]>cm.max()/2 else "black",fontsize=9)
    ax.set_title(it["name"],fontsize=10)
    ax.set_xticks([0,1]); ax.set_xticklabels(["否","是"],fontsize=8)
    ax.set_yticks([0,1]); ax.set_yticklabels(["否","是"],fontsize=8)
    ax.set_xlabel("预测",fontsize=8); ax.set_ylabel("真实",fontsize=8)
for ax in axes[len(items):]: ax.axis("off")
fig.suptitle("各诊断项混淆矩阵（阈值=0.5）",fontsize=14)
fig.tight_layout(rect=[0,0,1,0.98])
fig.savefig(os.path.join(FIG,"confusion_matrices.png"),dpi=200)
plt.close(fig)

# 4) 样本量/阳性数
fig,ax=plt.subplots(figsize=(max(14,len(items)*0.6),6))
ax.bar(x-0.5*w,[i["n"]-i["npos"] for i in items],w,label="阴性")
ax.bar(x+0.5*w,[i["npos"] for i in items],w,label="阳性")
ax.set_xticks(x); ax.set_xticklabels(names,rotation=60,ha="right",fontsize=9)
ax.set_ylabel("样本数"); ax.legend(); ax.set_title("各诊断项测试样本（阳性/阴性）")
ax.grid(True,axis="y",alpha=0.3); fig.tight_layout()
fig.savefig(os.path.join(FIG,"label_distribution.png"),dpi=200)
plt.close(fig)

print("figs saved to", FIG)
