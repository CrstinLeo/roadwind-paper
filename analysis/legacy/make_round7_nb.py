import nbformat as nbf
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 模型 A 第七轮：参数敏感性分析与模型改进

**问题**：第 6 轮模型（SMHI → TrV，高度 + 粗糙度 + 站点因子 + 路边站环境模型 + 多站合并）里有很多参数是我们自己定的，
比如上风向看多远、扇区多宽、粗糙度足迹多大、合并多少个站。这些值换一换，结果会变多少？哪个范围最合适？模型还能怎么改进？

**数据**：2026-11-13 20:00 至 11-23 15:00 UTC，共 236 小时；133 个 TrV 站；17 个 SMHI 站。SMHI 站点因子只用 2022-01 至 2026-11-13 19:00 的历史。

**评估方法（避免"在检验集上调参"）**
- **主评分：时间分块 × 留一站交叉验证。** 把 236 小时切成 4 段（每段约 59 小时）。每次用其中 3 段拟合路边站模型，预测第 4 段；
  同时被预测的路边站也不参与拟合。所以每个预测值都是"没见过的时间 + 没见过的站"。
- **指标**：每个路边站的风速 MAE（m/s）在 133 个站上的平均。比较两个设置时用**配对差值**（同一个站的 MAE 相减）及其标准误 SE。
- **选值规则（1-SE 规则）**：在一组取值里，凡是与最好值的差距小于 1 个 SE 的，都算"同样好"，组成推荐范围；范围内优先选物理上更有依据或更简单的值。
- **二次检查**：另用"前 124 小时拟合、后 112 小时检验"的切分，确认改进在完全没参与拟合的后半段也成立。
""")

code(r"""
import sys, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3de", "grid.linewidth": 0.6, "font.size": 9.5,
                     "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False})
C1, C2, C3, C4, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#1d1d1b", "#52514e", "#bdbcb6"
OUT = "/tmp/sww_out/"
S = pickle.load(open(OUT + "sensitivity_cv.pkl", "rb"))["res"]
CAND = pickle.load(open(OUT + "candidates_cv.pkl", "rb"))
TWO = pickle.load(open(OUT + "two_scale_cv.pkl", "rb"))
EX = pickle.load(open(OUT + "extra7.pkl", "rb"))
CO = pickle.load(open(OUT + "m6_coefs.pkl", "rb"))
base = S[("structure", "baseline")]["r"]["mae"]
def paired(a, b):
    d = a - b; ok = np.isfinite(d); return d[ok].mean(), d[ok].std(ddof=1) / np.sqrt(ok.sum())
print(f"基线（第 6 轮模型）交叉验证：站均 MAE {np.nanmean(base):.3f} m/s，中位数 {np.nanmedian(base):.3f}；"
      f"最近 SMHI × 0.912：中位数 {np.nanmedian(S[('structure','baseline')]['r']['mae_raw']):.3f} m/s")
""")

md(r"""
## 1. 单参数敏感性：每次只改一个参数，其余保持第 6 轮的设置

纵轴是"站均 MAE 相对第 6 轮基线的变化"（m/s，负值 = 更好），误差棒是配对 ±1 SE。
绿色底的取值是推荐范围（1-SE 规则，加上物理合理性），黑框是建议采用的值。作为量级参照：基线站均 MAE 约 0.74 m/s，
0.01 m/s 约等于 1.4%。
""")

code(r"""
groups = [
  ("z_blend", "混合高度 z_b（第 2 步）", ["30 m","40 m","60 m","80 m","100 m","150 m"], ["30 m","40 m","60 m","80 m","100 m","150 m"], "60 m"),
  ("c_tree", "树木粗糙度 z0 = c·h 中的 c", ["0.05","0.075","0.1","0.125","0.15"], ["0.05","0.075","0.1"], "0.1"),
  ("L_smhi", "SMHI 粗糙度足迹 L", ["25 m","50 m","100 m","200 m","400 m","1000 m","2000 m"], ["25 m","50 m","100 m","200 m","400 m"], "100 m"),
  ("L_trv", "路边站粗糙度足迹 L", ["25 m","50 m","100 m","200 m","400 m","1000 m","2000 m"], ["50 m","100 m","200 m","400 m","1000 m"], "100 m"),
  ("site", "SMHI 站点因子", ["none","mean","sector"], ["mean","sector"], "sector"),
  ("site_nsec", "站点因子的风向扇区数", ["4","8","16"], ["4","8","16"], "8"),
  ("site_min_net", "站点因子：站网风速门槛", ["1.0 m/s","2.0 m/s","3.0 m/s","4.0 m/s","5.0 m/s"], ["1.0 m/s","2.0 m/s","3.0 m/s","4.0 m/s"], "3.0 m/s"),
  ("tree_R", "上风向树木：半径 R（单尺度）", ["50 m","100 m","150 m","200 m","300 m","500 m","1000 m"], ["100 m","1000 m"], "100 m"),
  ("tree_H", "上风向树木：扇区半角", ["±11.25°","±22.5°","±45°","±90°","±180°"], ["±22.5°","±45°"], "±22.5°"),
  ("tree_soft", "树木：指数权重 exp(−r/L)", ["L=25 m","L=50 m","L=100 m","L=200 m","L=400 m"], ["L=25 m","L=400 m"], None),
  ("road_window", "沿路平均窗口（±S）", ["±0 m","±100 m","±250 m","±500 m","±750 m","±1000 m"], ["±0 m","±100 m"], "±0 m"),
  ("fit_min", "拟合用的最低 SMHI 风速", ["1.0 m/s","1.5 m/s","2.0 m/s","2.5 m/s","3.0 m/s","4.0 m/s"], ["1.5 m/s","2.0 m/s"], "2.0 m/s"),
  ("km", "合并半径", ["10 km","20 km","30 km","40 km","60 km","80 km","100 km","150 km"], ["80 km","100 km","150 km"], "150 km"),
  ("power", "合并权重 1/d^p（150 km 内全部）", ["0.0","0.5","1.0","1.5","2.0","3.0"], ["1.0","1.5","2.0"], "1.5"),
  ("min_n", "至少合并几个站（60 km）", ["1","2","3","4","6"], ["4","6"], None),
  ("dir_km", "风向合并半径（对风速无影响）", ["20 km","40 km","60 km","80 km","150 km"], ["20 km","40 km","60 km","80 km","150 km"], "60 km"),
]
fig, axes = plt.subplots(4, 4, figsize=(15, 13.5))
rows = []
for ax, (g, title, labels, rec, pick) in zip(axes.ravel(), groups):
    ys, es = [], []
    for l in labels:
        m, s = paired(S[(g, l)]["r"]["mae"], base); ys.append(m); es.append(s)
    x = np.arange(len(labels))
    for i, l in enumerate(labels):
        if l in rec: ax.axvspan(i - 0.45, i + 0.45, color=C3, alpha=0.13, lw=0)
    ax.errorbar(x, ys, yerr=es, fmt="o", color=C1, ms=5, lw=1.2, capsize=2)
    if pick in labels:
        i = labels.index(pick); ax.plot(i, ys[i], "s", ms=11, mfc="none", mec=INK, mew=1.4)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([l.replace(" m/s", "") for l in labels], rotation=40, fontsize=8)
    ax.set_title(title, fontsize=10); ax.set_ylim(min(-0.03, min(ys) - 0.01), max(0.03, max(ys) + max(es) + 0.005))
    rows.append(dict(参数=title, 试验范围=f"{labels[0]} – {labels[-1]}", 影响幅度=f"{max(ys) - min(ys):.3f}",
                     推荐范围=" / ".join(rec) if len(rec) <= 3 else f"{rec[0]} – {rec[-1]}",
                     建议值=pick or ("见第 2 节" if g == "tree_soft" else "—（合并全部站后无关）")))
for ax in axes[:, 0]: ax.set_ylabel("Δ MAE vs 基线 [m/s]")
fig.tight_layout(); plt.show()
pd.DataFrame(rows).style.hide(axis="index")
""")

md(r"""
**怎么读**："影响幅度"是这一组里最好与最差取值之差。小于 0.005 m/s（不到 1%）的参数可以认为不敏感，直接用文献值。

- **不敏感（物理换算部分）**：混合高度 30–150 m、粗糙度系数 c = 0.05–0.1、扇区半角 ±22.5–45° 的影响都在 0.005 m/s 以内；
  路边站足迹 L 的幅度是 0.008，但每个点的 SE 有 0.004–0.008，差别不显著。继续用文献值（60 m、0.1h），不需要调。
  SMHI 足迹短一些（25–400 m）比 1 km 稍好，但只差 0.001–0.004 m/s。
- **必不可少**：SMHI 站点因子。去掉后 MAE 增加 0.049 ± 0.013 m/s，是所有参数中影响最大的一项之一。
  但"按风向分 16 扇区"并不比"每站一个常数"显著更好（+0.0025 ± 0.006），4–8 个扇区就够了，建议 8 个（每格样本更多）。
- **合并半径影响大**：只用近处站（≤ 40 km）明显变差（+0.025 到 +0.05）；80 km 以上（实际就是所有 17 个站）最好，权重 1/d 到 1/d²，最好是 1/d^1.5。
  这和第 4、5 轮的结论一致：站点因子校正以后，"选哪个站"已不重要，重要的是用足够多的站平均掉单站噪声。
- **拟合门槛**：1.5–2 m/s 最好。低于 1.5 m/s 把大量微风噪声放进拟合；高于 2.5 m/s 又让模型只适合大风，平均偏差变大。
- **上风向树木的半径**出现两个最优点（100 m 和 1 km），中间的 150–500 m 反而更差。这说明路边站同时受两种尺度影响，见第 2 节。
""")

md(r"""
## 2. "风吹拂范围"：近处树木和大范围森林是两个不同的效应

把树木特征拆成两个：**近处**（上风向半径 R_near 内）和**远处**（R_far 内），同时放进模型。
下表是交叉验证 MAE 相对"100 m + 1 km"组合的差值（正值 = 更差）。
""")

code(r"""
ref = TWO[(100, "tree_R1000_H22.5")]["mae"]
nears = [50, 100, 150, 200]
fars = [("none", "无远处项"), ("tree_R300_H22.5", "300 m"), ("tree_R500_H22.5", "500 m"), ("tree_R1000_H22.5", "1 km"),
        ("tree_R500_H90", "500 m, ±90°"), ("tree_R1000_H90", "1 km, ±90°"), ("tree_R1000_H180", "1 km, 全方向"), ("treew_L400_H22.5", "exp(−r/400 m)")]
M = np.array([[paired(TWO[(n, f)]["mae"], ref)[0] for f, _ in fars] for n in nears])
E_ = np.array([[paired(TWO[(n, f)]["mae"], ref)[1] for f, _ in fars] for n in nears])
fig, ax = plt.subplots(figsize=(11, 3.6))
im = ax.imshow(M, cmap="RdBu_r", vmin=-0.02, vmax=0.02, aspect="auto")
for i in range(len(nears)):
    for j in range(len(fars)):
        ax.text(j, i, f"{M[i,j]:+.3f}\n±{E_[i,j]:.3f}", ha="center", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(nears))); ax.set_yticklabels([f"近处 {n} m" for n in nears])
ax.set_xticks(range(len(fars))); ax.set_xticklabels([l for _, l in fars], fontsize=8.5); ax.grid(False)
plt.colorbar(im, ax=ax, label="Δ MAE [m/s]"); ax.set_title("近处半径 × 远处范围（参照：100 m + 1 km = 0）"); plt.show()
""")

md(r"""
- **近处：50–100 m。** 100 m 最好，50 m 只差 0.003（小于 1 SE）；150 m 和 200 m 明显更差（+0.009、+0.012，都大于 3 SE）。
  第 6 轮用的 200 m 偏大。物理上，100 m 约为 4–7 倍树高（这里树高多为 15–25 m），与防风林、林缘研究中背风侧风速明显减弱的范围（几倍到十几倍障碍高度）同一量级。
- **远处：约 1 km 的森林覆盖率。** 300–500 m 的远处项几乎没有帮助，1 km 才有。扇区宽度（±22.5°、±90° 或全方向）和硬边界/指数权重（L ≈ 400 m）都差不多，
  说明这一项代表的是"整片区域有多少森林"，即景观尺度的粗糙度，与风向关系不大。这和 SMHI 站点因子中 1 km 足迹粗糙度显著的结果一致。
- **路边站粗糙度足迹 L_trv**（放在两尺度模型里再测）：25–1000 m 之间差别都小于 1–2 SE，保持 100 m。
""")

md(r"""
## 3. 路段长度：特征沿道路变化有多快？

车辆数据会按路段（硕士论文用 500 m）汇总，那么路段上的环境特征也要按路段平均。用 TrV 站点两侧沿道路方向每隔 100–250 m 重新计算上风向特征，看两个问题：
1. 离站点 s 米处的特征与站点处特征的相关有多高（特征的"相关长度"）；
2. 把站点特征换成"±S 米路段的平均值"后，对站点风速的预测变差多少。
""")

code(r"""
ac = EX["along_ac"]; s = np.array(sorted(ac)); r1 = np.array([ac[k][0] for k in s]); r2 = np.array([ac[k][1] for k in s])
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
ax[0].plot(s, r1, "-o", color=C1, ms=4, label="上风向树木占比（200 m）"); ax[0].plot(s, r2, "-o", color=C2, ms=4, label="上风向 ln z0（L = 100 m）")
ax[0].axhline(0.37, color=GREY, lw=0.8, ls="--"); ax[0].text(-1000, 0.39, "远处平台 ≈ 0.37", fontsize=8, color=INK2)
ax[0].set_xlabel("沿道路离站点的距离 s [m]"); ax[0].set_ylabel("与站点处特征的相关"); ax[0].legend(fontsize=8, frameon=False); ax[0].set_ylim(0, 1.05)
labs = ["±0 m", "±100 m", "±250 m", "±500 m", "±750 m", "±1000 m"]; xs = [0, 100, 250, 500, 750, 1000]
d = [paired(S[("road_window", l)]["r"]["mae"], base) for l in labs]
ax[1].errorbar([2 * x for x in xs], [v[0] for v in d], yerr=[v[1] for v in d], fmt="-o", color=C1, ms=5, capsize=2)
ax[1].axhline(0, color=GREY, lw=0.8); ax[1].set_xlabel("路段长度 2S [m]（特征取路段平均）"); ax[1].set_ylabel("Δ MAE vs 点特征 [m/s]")
ax[1].set_title("用路段平均特征预测点观测"); fig.tight_layout(); plt.show()
""")

md(r"""
- 特征的相关在 100 m 处还有 0.84–0.89，250 m 处降到约 0.6–0.7，500 m 处约 0.45–0.54，750 m 以外稳定在约 0.37（这部分是区域共性）。
  扣掉区域共性后，**局地部分的相关长度约 250 m**（在 250 m 处降到约 1/e）：超过这个距离，道路两侧的遮挡情况基本是另一回事了。
- 用 ±100 m（200 m 路段）的平均特征预测点观测，效果和点特征相同；±250 m（500 m 路段）及更长时有变差的趋势（+0.003 到 +0.012 m/s），
  但单个点只有 1–1.5 个 SE，证据不强。
- **对车辆路段长度的含义**：路段越长，路段内不同遮挡的地方被平均在一起，模型能区分的局地差异就越少；路段越短，车载测量噪声越大。
  NOR-XWIND 的结果（第 16 节）是噪声在 100–500 m 之间几乎不随长度减小，信噪比最高在 250–500 m。两者合起来，**建议路段长度 200–300 m，取 250 m**，
  500 m 作为敏感性对照（500 m 路段两端与中心的特征相关只剩约 0.6，路段内已经混进了不同的遮挡环境）。这一点最终要用 RV40 车辆数据确认。
""")

md(r"""
## 4. 模型结构改进：逐项加入

每一行都在上一行的基础上加一项，全部用同样的交叉验证。"不用粗糙度物理"和"不用站点因子"是消融对照；GBM（梯度提升树）用同样的特征再加地形，代表"非线性机器学习能做到多少"。
""")

code(r"""
R = CAND["res"]; b = R["M0 baseline (round 6)"]["mae"]
order = ["M0 baseline (round 6)", "M1 = M0 − lnU×tree", "M2 = M1 + stability", "M3 = M2 + cross-road×tree",
         "M4 = M2 + two-scale trees (100 m + 1 km)", "M5 = M4 + all stations, 1/d^1.5", "M6 = M5 + 8 sectors, L_smhi 100 m",
         "M7 = M6 + cross-road×tree", "M6 without roughness physics", "M0 without roughness physics", "M0 without site factors",
         "GBM upper bound (M7 features + Sx, TPI, h_mid)"]
zh = {"M0 baseline (round 6)": "M0 第 6 轮模型", "M1 = M0 − lnU×tree": "M1 = M0 去掉 lnU×tree",
      "M2 = M1 + stability": "M2 = M1 + 稳定度 (T_路面 − T_气温)", "M3 = M2 + cross-road×tree": "M3 = M2 + 横风×树木",
      "M4 = M2 + two-scale trees (100 m + 1 km)": "M4 = M2 + 两尺度树木 (100 m + 1 km)",
      "M5 = M4 + all stations, 1/d^1.5": "M5 = M4 + 全部站合并 1/d^1.5", "M6 = M5 + 8 sectors, L_smhi 100 m": "M6 = M5 + 8 扇区、SMHI 足迹 100 m",
      "M7 = M6 + cross-road×tree": "M7 = M6 + 横风×树木", "M6 without roughness physics": "M6 但不做粗糙度换算",
      "M0 without roughness physics": "M0 但不做粗糙度换算", "M0 without site factors": "M0 但不用站点因子",
      "GBM upper bound (M7 features + Sx, TPI, h_mid)": "GBM（M7 特征 + 地形）"}
tab = []
for k in order:
    r = R[k]; m, se = paired(r["mae"], b)
    tab.append(dict(模型=zh[k], 站均MAE=round(np.nanmean(r["mae"]), 3), MAE中位数=round(np.nanmedian(r["mae"]), 3),
                    相对M0=f"{m:+.3f} ± {se:.3f}", 对数偏差中位数=f"{np.nanmedian(r['bias_log']):+.3f}",
                    优于最近站=f"{int((r['mae'] < r['mae_raw']).sum())}/133"))
T = pd.DataFrame(tab)
fig, ax = plt.subplots(figsize=(10, 4.6))
y = np.arange(len(order))[::-1]
vals = [paired(R[k]["mae"], b) for k in order]
cols = [C1 if k.startswith("M") and "without" not in k else (C2 if "without" in k else C4) for k in order]
ax.barh(y, [v[0] for v in vals], xerr=[v[1] for v in vals], color=cols, height=0.6, capsize=2, error_kw=dict(lw=0.8))
ax.set_yticks(y); ax.set_yticklabels([zh[k] for k in order], fontsize=8.5); ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("站均 MAE 相对 M0 的变化 [m/s]（负值 = 更好，±1 SE）"); fig.tight_layout(); plt.show()
T.style.hide(axis="index").format(precision=3)
""")

md(r"""
**结论**
- **去掉 lnU×tree 交互项（M1）**：更简单也更准（−0.009 ± 0.002）。第 3 轮在大风时段看到的交互作用，在更长的数据里没有交叉验证的支持。
- **大气稳定度（M2）**：用 TrV 自己测的"路面温度 − 气温"作稳定度代理（路面比空气暖 = 不稳定，混合强，近地面风相对更大），稳定、显著地改善（−0.005 ± 0.001）。系数在 4 个时间段里都为正。
- **两尺度树木（M4）**：最大的单项改进（在 M2 基础上再 −0.016）。
- **合并全部站（M5）**：再 −0.012。
- **M6**（8 扇区、SMHI 足迹 100 m）与 M5 一样好但参数更少，**推荐作为新模型**：站均 MAE 0.740 → 0.698 m/s（−5.8%，−0.043 ± 0.006），中位数 0.650 → 0.631。
- 横风×树木（M3/M7）只有很小的改进（−0.002 到 −0.004），暂不加入。
- **粗糙度物理的作用**：在 M0 中去掉粗糙度换算会变差（+0.016 ± 0.008）；在 M6 中去掉，站均 MAE 反而略低，但中位数变差、偏差变大（+0.127 对 +0.097），差别在 1 SE 以内。
  直接配对比较，"M6 不做粗糙度换算 − M6" = −0.006 ± 0.007，不显著。也就是说，有了两尺度树木特征后，经验模型基本可以替代粗糙度换算。**仍建议保留物理换算**：它不需要拟合、能换算到任意高度（车辆 3.4–4 m），对新环境更可靠。
- **GBM 没有超过线性模型**（+0.007 ± 0.014）：在 133 个站、约 10 天的数据量下，线性模型已经用完了这些特征里的信息，换成机器学习没有收益。
- **剩下的主要误差**：交叉验证中对数偏差中位数约 +0.09（预测偏低约 9%），而在同一时段内拟合时几乎为 0。这说明误差里有随天气过程变化的区域性共同偏差，
  静态模型无法消除，需要第 6 部分的卡尔曼滤波用实时 TrV 观测来跟踪。
""")

md(r"""
## 5. 新模型 M6 的方程和系数

$$\ln\frac{U_{TrV}}{U_3} = b_0 + b_1\,\text{tree}_{100\,m} + b_2 \ln(x/h) + b_3 \ln U_{net} + b_4\,\Delta T + b_5\,\text{tree}_{1\,km}$$

$U_3$ 是物理换算 + 站点因子之后的预测（第 1–3 步不变），$\Delta T = T_{路面} - T_{气温}$（K）。风向模型保持第 6 轮的形式和设置。
系数用全部 236 小时拟合；"4 段范围"是 4 次交叉验证各自拟合的最小–最大值，用来看系数是否稳定。
""")

code(r"""
names = ["b0 截距", "b1 近处树木 (100 m)", "b2 ln(x/h)", "b3 ln U_net", "b4 ΔT (K⁻¹)", "b5 远处森林 (1 km)"]
order_idx = [0, 1, 2, 3, 4, 5]
pd.DataFrame(dict(参数=names, 系数=CO["full"], 四段最小=CO["folds"].min(0), 四段最大=CO["folds"].max(0))).style.hide(axis="index").format(precision=3)
""")

code(r"""
print(f"smearing 系数 S = {CO['smear']:.3f}；风向（与第 6 轮相同的模型）：d0 = {CO['d'][0]:.1f}°，d1 = {CO['d'][1]:.1f}°，四段范围 d0 {CO['dfolds'][:,0].min():.1f}…{CO['dfolds'][:,0].max():.1f}°，d1 {CO['dfolds'][:,1].min():.1f}…{CO['dfolds'][:,1].max():.1f}°")
print("举例（其他项不变时）：近处和远处都全是树（两项 = 1）相对两项 = 0，风速倍数 ≈ "
      f"{np.exp(CO['full'][1] + CO['full'][5]):.2f}；只有近处树木：{np.exp(CO['full'][1]):.2f}；只有远处森林：{np.exp(CO['full'][5]):.2f}")
print(f"路面比气温高 2 K（不稳定）对比低 2 K（稳定）：风速倍数 {np.exp(4 * CO['full'][4]):.2f}")
""")

md(r"""
所有系数的符号在 4 个时间段里都一致，量级稳定（ΔT 在 0.012–0.034，远处森林在 −0.18 到 −0.13）。

## 6. 二次检查：只用前 124 小时拟合，在后 112 小时检验
""")

code(r"""
sys.path.insert(0, "/home/claude/smhi_trv_wind")
from smhi_trv_wind import Settings
from smhi_trv_wind.sensitivity import Engine, Spec
cfg = Settings(max_pair_km=150.0)
Eng = Engine(cfg, pd.read_parquet("/home/claude/se/features_variants.parquet"), t_split="2026-11-19 00:00", t_end="2026-11-23 15:00")
M6 = CAND["specs"]["M6 = M5 + 8 sectors, L_smhi 100 m"]
r0, r6 = Eng.run(Spec()), Eng.run(M6)
rows = []
for per, name in (("A", "拟合期 11-13 → 11-18（124 h）"), ("B", "检验期 11-19 → 11-23（112 h）")):
    m, se = paired(r6[per]["mae"], r0[per]["mae"])
    rows.append({"时段": name, "最近 SMHI ×0.912（中位数）": round(np.nanmedian(r0[per]["mae_raw"]), 3),
                 "M0 站均 / 中位数": f"{np.nanmean(r0[per]['mae']):.3f} / {np.nanmedian(r0[per]['mae']):.3f}",
                 "M6 站均 / 中位数": f"{np.nanmean(r6[per]['mae']):.3f} / {np.nanmedian(r6[per]['mae']):.3f}",
                 "M6 − M0": f"{m:+.3f} ± {se:.3f}", "M6 优于最近站": f"{int((r6[per]['mae'] < r6[per]['mae_raw']).sum())}/133",
                 "风向 MAE 中位数 M0 / M6": f"{np.nanmedian(r0[per]['dir']):.1f}° / {np.nanmedian(r6[per]['dir']):.1f}°"})
pd.DataFrame(rows).style.hide(axis="index")
""")

md(r"""
在完全没参与拟合的后 112 小时，M6 的改进（约 −0.06 m/s）比交叉验证中更大，说明改进不是调参造成的假象。

## 7. 推荐参数表

| 参数 | 第 6 轮 | 推荐值 | 合理范围 | 依据 |
|---|---|---|---|---|
| 混合高度 z_b | 60 m | 60 m | 30–150 m | 不敏感；Wieringa 文献值 |
| 树木粗糙度 z0 = c·h | 0.1 | 0.1 | 0.05–0.1 | 0.125 以上变差 |
| SMHI 粗糙度足迹 L | 1000 m | 100 m | 25–400 m | 差别很小（≤ 0.004 m/s） |
| 路边站粗糙度足迹 L | 100 m | 100 m | 50–1000 m | 不敏感 |
| 特征扇区半角 | ±22.5° | ±22.5° | ±22.5–45° | ±11.25° 变差 |
| 站点因子扇区数 | 16 | 8 | 4–16 | 差别不显著，8 个每格样本更多 |
| 站点因子：站网风速门槛 | 3 m/s | 3 m/s | 1–4 m/s | 不敏感 |
| 近处树木半径 | 200 m | **100 m** | 50–100 m | 150 m 以上显著变差 |
| 远处森林范围 | 无 | **1 km** | 1 km 左右（扇区宽度不敏感） | 新增项，300–500 m 无效 |
| 拟合最低风速 | 2 m/s | 2 m/s | 1.5–2 m/s | <1.5 噪声大，>2.5 偏差大 |
| 风速合并半径 | 60 km | **全部站（≤ 150 km）** | ≥ 80 km | ≤ 40 km 明显变差 |
| 风速合并权重 | 1/d | **1/d^1.5** | 1/d – 1/d² | 等权（p = 0）变差 |
| 风向合并 | 60 km，1/d | 60 km，1/d | 20–150 km | 对结果影响在 ±1° 内 |
| 车辆路段长度 | 500 m（硕士论文） | **250 m** | 200–300 m | 特征相关长度约 250 m；NOR-XWIND 信噪比 |

## 8. 局限

- 只有约 10 天、11 月（落叶期、气温接近 0 °C）的数据，最大站网风速约 7 m/s；夏季有叶期、强风条件下的参数可能不同，需要 TrV 长期历史（VViS 档案）重做。
- 选值用的就是这 236 小时（虽然是交叉验证），M6 的改进幅度可能略有乐观；第 6 节的后段检验部分缓解了这个问题。最好的检验是今后几周继续追加的 TrV 数据，以及 RV40 车辆数据。
- 路段长度的建议来自站点特征的空间结构和挪威数据，还没有用瑞典车辆数据验证。
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "/home/claude/se/ModelA_round7_sensitivity_and_improvements.ipynb"); print("ok")
