import json, glob, os
import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

RES="/home/mzhao/华为项目/VLM/out/pvd_no_mv_2026-10-15_full"
FIG="/home/mzhao/华为项目/VLM/figs_selective"
os.makedirs(FIG, exist_ok=True)
FONT="/home/mzhao/华为项目/VLM/src/simhei中易黑体.ttf"
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"]=font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"]=False

# 载入全量逐条
items=[]
for fp in sorted(glob.glob(os.path.join(RES,"*.json"))):
    d=json.load(open(fp))
    if "results" not in d: continue
    yt=np.array([1 if r["msg"]["conversations"][-1]["value"].strip()=="是" else 0 for r in d["results"]])
    pr=np.array([float(r["prob_y"]) for r in d["results"]])
    items.append((d["diag_item"], yt, pr))

def evaluate(tau):
    per=[]; tot=0; cor=0; alln=0
    for it,yt,pr in items:
        alln+=len(yt)
        conf=(pr>=tau)|(pr<=1-tau)
        if conf.sum()==0:
            per.append((it,len(yt),0,np.nan,np.nan)); continue
        pred=(pr[conf]>0.5).astype(int); acc=(pred==yt[conf]).mean()
        per.append((it,len(yt),int(conf.sum()),acc,np.nan))
        tot+=conf.sum(); cor+=(pred==yt[conf]).sum()
    macro=np.nanmean([p[3] for p in per])
    return per, macro, tot/alln if alln else 0, (cor/tot if tot else np.nan)

taus=[0.0,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.925,0.95,0.975,0.99]
curve=[]
for t in taus:
    _,mac,cov,pool=evaluate(t)
    curve.append((t,cov,mac,pool))
cv=pd.DataFrame(curve,columns=["置信阈值τ","覆盖率","宏平均准确率","池化准确率"])

# 选满足 宏平均准确率>=0.95 的最小覆盖率(即最大τ里第一个达标)
cand=cv[cv["宏平均准确率"]>=0.95]
if len(cand):
    row=cand.sort_values("覆盖率").iloc[-1]; TAU=float(row["置信阈值τ"])
else:
    TAU=float(cv.sort_values("宏平均准确率").iloc[-1]["置信阈值τ"])
print("选定 τ*=",TAU, " 宏平均Acc=",float(cv[cv["置信阈值τ"]==TAU]["宏平均准确率"].iloc[0]))

per,mac,cov,pool=evaluate(TAU)
per_df=pd.DataFrame([(it,n,nc,round(nc/n,4),round(a,4) if a==a else None)
                     for it,n,nc,a,_ in per],
                    columns=["诊断项","总样本数","高置信样本数","覆盖率","高置信准确率"])

# 构造高置信子集 json（保留原记录 + 置信标记），选择依据=置信度(非对错)
subset=[]
for it,yt,pr in items:
    # 重新读原始记录以保留路径
    pass
sub=[]
for fp in sorted(glob.glob(os.path.join(RES,"*.json"))):
    d=json.load(open(fp))
    if "results" not in d: continue
    for r in d["results"]:
        p=float(r["prob_y"])
        if p>=TAU or p<=1-TAU:
            sub.append({"diag_item":d["diag_item"],"video":r["msg"]["video"][0],
                        "真实标签":r["msg"]["conversations"][-1]["value"].strip(),
                        "prob_y":round(p,6),"τ":TAU,
                        "预测":"是" if p>0.5 else "否"})
out_json=os.path.join(RES,"input","selective_subset_tau%.3f.json"%TAU)
os.makedirs(os.path.dirname(out_json),exist_ok=True)
json.dump({"说明":"选择性预测高置信子集；选择依据为模型置信度(>=τ 或 <=1-τ)，非按对错挑选；覆盖率见同目录 CSV；不得单独作为模型整体性能汇报",
           "tau":TAU,"coverage":cov,"macro_accuracy":mac,"pooled_accuracy":pool,"n":len(sub),
           "samples":sub}, open(out_json,"w"), ensure_ascii=False)
print("subset saved:", out_json, "n=",len(sub))

# xlsx
xls=os.path.join(RES,"selective_prediction.xlsx")
thin=Side(style="thin",color="D9D9D9"); border=Border(left=thin,right=thin,top=thin,bottom=thin)
hf=PatternFill("solid",fgColor="D9E1F2")
def fmt(ws,d):
    for j,c in enumerate(d.columns,1):
        w=max(len(str(c)),*(len(str(v)) for v in d[c].astype(str)))
        ws.column_dimensions[get_column_letter(j)].width=min(max(w+2,8),22)
    for cell in ws[1]:
        cell.font=Font(bold=True); cell.fill=hf; cell.border=border; cell.alignment=Alignment(horizontal="center",wrap_text=True)
    ws.freeze_panes="A2"; ws.auto_filter.ref=ws.dimensions
    for r in ws.iter_rows(min_row=2,max_row=ws.max_row):
        for cell in r: cell.border=border; cell.alignment=Alignment(horizontal="center")
with pd.ExcelWriter(xls, engine="openpyxl") as w:
    cv.round(4).to_excel(w,index=False,sheet_name="覆盖率-准确率曲线")
    per_df.to_excel(w,index=False,sheet_name="各诊断项@τ=%.3f"%TAU)
    fmt(w.sheets["覆盖率-准确率曲线"], cv.round(4)); fmt(w.sheets["各诊断项@τ=%.3f"%TAU], per_df)
print("xlsx:",xls)

# 图
fig,ax=plt.subplots(figsize=(8,5.5))
ax.plot(cv["覆盖率"],cv["宏平均准确率"],"o-",label="宏平均准确率")
ax.plot(cv["覆盖率"],cv["池化准确率"],"s--",label="池化准确率")
ax.axhline(0.95,color="r",ls=":",label="95%")
for _,r in cv.iterrows():
    ax.annotate(f"τ={r['置信阈值τ']:g}",(r["覆盖率"],r["宏平均准确率"]),fontsize=7,xytext=(3,3),textcoords="offset points")
ax.set_xlabel("覆盖率 (保留比例)"); ax.set_ylabel("准确率"); ax.set_title("选择性预测：覆盖率 vs 准确率（完整非平衡测试集）")
ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(FIG,"selective_coverage_accuracy.png"),dpi=200); plt.close(fig)
print("fig saved")
print(cv.round(4).to_string(index=False))
