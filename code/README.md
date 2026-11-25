# smhi_trv_wind：SMHI 气象站 → Trafikverket 路边站的风校正

把 SMHI 气象站 10 m 的风，经过高度换算、粗糙度换算、站点因子校正、路边站环境模型、多站合并，
转换成 Trafikverket（TrV）路边站 6 m 处的风速和风向，并逐步评估每一步的效果（第 6 轮分析）。

## 1. 目录结构

```
smhi_trv_wind/
├── README.md
├── pyproject.toml
├── smhi_trv_wind/                 Python 包
│   ├── config.py                  所有路径、时段、物理常数、超参数（标明 [phys]/[data]/[choice]）
│   ├── geo.py                     距离、方位角、角度折算、16 扇区
│   ├── data/
│   │   ├── smhi.py                SMHI 开放数据：下载站点和逐小时风速/风向，读成 [时间 × 站点] 矩阵
│   │   └── trafikverket.py        TrV Open API：下载站点和观测（只保留约 7 天，需定期追加），读成矩阵
│   ├── features/
│   │   ├── gis.py                 NMD 激光物体高度 + NMD 地表覆盖 → 每站 16 扇区特征
│   │   ├── variants.py            敏感性用的特征变体：不同半径/扇区宽度/足迹 L/沿路窗口
│   │   └── road.py                OSRM 道路方位（路边站所在道路的走向）
│   ├── model/
│   │   ├── physics.py             第 1–2 步：对数律高度换算、Wieringa 两步粗糙度换算
│   │   ├── site_factors.py        第 3 步：SMHI 站点因子 c_s(k)（双向固定效应），稳定性检验，地理解释
│   │   ├── exposure.py            第 4–5 步：路边站环境模型（风速）+ 道路引导（风向），留一站拟合
│   │   └── combine.py             第 6 步：60 km 内多站 1/d 加权合并
│   ├── evaluate.py                站对指标、校正阶梯、站点指标、SMHI 站价值
│   ├── sensitivity.py             第 7 轮：快速重算引擎（矩阵化）、时间分块 × 留一站交叉验证、推荐模型 RECOMMENDED
│   ├── pipeline.py                串起第 1–6 步：run() → Result，save() 写出结果
│   └── viz/
│       ├── station_map.py         交互地图 HTML
│       └── map_template.html
├── scripts/
│   ├── 01_fetch_smhi.py           下载 SMHI
│   ├── 02_fetch_trafikverket.py   下载 TrV 并追加到本地档案
│   ├── 03_build_features.py       计算站点 GIS 特征和道路方位
│   ├── 04_run_model.py            运行模型、输出指标和地图（第 6 轮模型）
│   ├── 05_sensitivity.py          单参数扫描：前段拟合、后段检验
│   ├── 06_sensitivity_cv.py       单参数扫描：时间分块 × 留一站交叉验证
│   ├── 07_candidate_models.py     模型结构改进 M0–M7、消融、GBM 上限
│   └── 08_two_scale_grid.py       近处 × 远处树木尺度网格
└── tests/test_model.py            合成数据单元测试（物理恒等式、参数还原）
```

## 2. 数据流

```
SMHI 开放数据 (10 m, 逐小时) ──┐
                               ├─ data/smhi.py ──────── 历史 2022-01 → 2026-11-13 ──┐
                               │                         检验期 11-13 → 11-22 ──────┤
TrV Open API (6 m, 5 min) ─────┴─ data/trafikverket.py ─ 检验期整点值 ──────────────┤
                                                                                    │
NMD 物体高度 (激光, 10 m) ──┐                                                       │
NMD 地表覆盖 (水体, 10 m) ──┼─ features/gis.py ─ 每站 × 16 扇区:                    │
OSRM / OpenStreetMap ───────┘  features/road.py   tree_near, x/h, z0(足迹), 道路方位 ─┤
                                                                                    ▼
                         pipeline.run():
                         历史 → physics.potential_wind → site_factors.fit  (c_s(k))
                         检验期 → 每对 (TrV j, SMHI s ≤ 80 km) 逐小时:
                             U0 原始 → U1 高度 → U2 粗糙度 → U3 去站点因子
                         → exposure.fit_loso   U4, φ4（被预测站不参与拟合）
                         → combine.combine     每个 TrV 站一个估计
                         → evaluate            阶梯、站点指标、SMHI 站价值
                         → viz.station_map     地图
```

## 3. 公式与代码位置

| 步骤 | 公式 | 参数怎么来 | 代码 |
|---|---|---|---|
| 1 高度 | U1 = U0 · ln(6/0.03)/ln(10/0.03) = 0.912·U0 | 物理常数 | `physics.height_factor` |
| 2 粗糙度 | U2 = U0 · P(z0_s,k) · F(z0_j,k)，经 60 m 混合高度 | 物理常数；z0 来自 NMD：树 0.1h，水 0.0002，其余 0.03；足迹 exp(−r/L) | `physics.to_potential/to_site`，`features/gis.roughness_features` |
| 3 站点因子 | ln U_pot,s(t) = a(t) + c_s(k) + ε，Σ_s c_s(k)=0；U3 = U2·e^(−c_s(k)) | SMHI 历史数据，交替最小二乘 | `site_factors.fit` |
| 4 风速模型 | ln(U_TrV/U3) = b0 + b1·tree + b2·ln(x/h) + (b3 + b4·tree)·ln U_net；U4 = U3·e^ŷ·S | TrV 数据，OLS，留一站；S 为 smearing | `exposure.ExposureModel` |
| 5 风向模型 | φ4 = φ_s + d0 + d1·sin(2α)·tree，α = φ_s − 道路方位 | TrV 数据，OLS，留一站 | `exposure.ExposureModel` |
| 6 合并 | ln Û = Σ w_i ln U4_i / Σ w_i，w = 1/d，60 km 内（至少 2 站）；风向加权矢量平均 | 规则（[choice]） | `combine.combine` |

k 是站网矢量平均风向所在的 16 扇区；tree、x/h、z0 都按 k 取对应扇区的值。

## 4. 运行

```bash
pip install -e .[dev]
export SWW_DATA=/path/to/data        # 数据根目录（见第 5 节）
export SWW_OUT=/path/to/output
export TRV_API_KEY=...               # 只在下载 TrV 时需要；不要写进代码或提交到 git

python scripts/01_fetch_smhi.py
python scripts/02_fetch_trafikverket.py       # 建议每 3 天运行一次，积累长序列
python scripts/03_build_features.py           # 需要先下载 NMD 栅格
SWW_ROUTE=rv40_osrm.json python scripts/04_run_model.py
python -m pytest -q tests
```

在 Python 里交互使用：

```python
from smhi_trv_wind import Settings, pipeline, evaluate
cfg = Settings(combine_km=40)          # 任何 [choice] 参数都可以这样改，做敏感性分析
res = pipeline.run(cfg)
evaluate.ladder(res.P); evaluate.summary(res.S); res.exposure.coefficients(); res.V
```

## 5. 数据目录（SWW_DATA）

```
smhi/stations_wind.csv                     01_fetch_smhi.py
smhi/{id}_{ws|wd|gust}_{corrected-archive|latest-months}.csv
trv/measurepoints.csv                      02_fetch_trafikverket.py
trv/obs_7d_120km.parquet                   （持续追加）
trv/road_bearing.csv                       03_build_features.py
features_16dir.parquet                     tree_near
features2_16dir.parquet                    x_edge, h_edge, xh_edge
features_water_16dir.parquet               z0w_fp_{100,400,1000}, dist_coast_km
nmd_ext/NMD2023_Tillaggsskikt_Objekthojd_objekttackning_v1_1/...tif   Naturvårdsverket 下载
nmd_base/{N,F}/..., nmd_O/...              NMD2018 基础层（各省 zip 解压）
```

数据来源：SMHI 开放数据（CC BY 4.0）；Trafikverket Open API（免费注册取 key）；Naturvårdsverket NMD2018/NMD2023（开放）；
OpenStreetMap / OSRM（道路方位）。数据文件和 API key 不在这个代码包里。

## 6. 输出（SWW_OUT）

| 文件 | 内容 |
|---|---|
| pairs_hourly.parquet | 每对站每小时：U0–U4、φ_s、φ4、特征 |
| pair_metrics.csv | 每对站：各步 MAE、偏差、相关、风向误差 |
| combined_hourly.parquet | 每个 TrV 站每小时的合并估计 |
| station_metrics.csv | 每个 TrV 站：模型 vs 最近 SMHI |
| smhi_station_value.csv | SMHI 站价值（两种定义） |
| site_factors_*.csv | 站点因子表 [站 × 扇区]（原始风 / 潜在风） |
| ladder.csv, model.json | 校正阶梯、模型系数、汇总 |
| SMHI_TrV_station_map.html | 交互地图 |

## 7. 复现检查（2026-11-22 数据）

用本包重新计算，与第 6 轮的结果一致：
- 813 对、133 个 TrV 站、217 小时；站点因子训练 25,680 小时；站均因子离散 0.469 → 0.366。
- 校正阶梯 MAE：1.753 → 1.560 → 1.372 → 1.300 → 0.742 m/s；偏差 −57% → −14%；平均误差 +1.51 → +0.02 m/s。
- 系数：b0 −1.133、tree +0.229、ln(x/h) +0.111、lnU +0.362、lnU·tree −0.345；d0 −13.6°、d1 −23.9°。
- 合并（60 km，至少 2 站）：MAE 中位数 0.63 m/s，风向 17.2°，121/133 个站优于最近站。
- GIS 特征：抽查 4 个站（2 个 TrV、2 个 SMHI），tree_near、x/h、z0、离海距离与原表完全一致。

与之前报告的两处差别：
1. 高度系数用精确值 0.912（之前取 0.91），所以第二步 MAE 是 1.56（之前 1.55），最近站基线 1.536（之前 1.537）。
2. SMHI 站价值改用"控制路边站"的定义（`quality_eff`）：在相同的 TrV 站上比较各 SMHI 站。之前的定义只控制距离
   （`quality_distance_only`），会把"所服务的 TrV 站本身容易预测"也算成站点价值。新排名：Landvetter、Torup、Halmstad、
   Rångedala、Tomtabacken、Axamo 最好；Naven、Kroppefjäll、Ullared、Såtenäs、Hällum 最差。Hagshult 从第 2 位降到中间。

## 8. 第 7 轮：敏感性分析与推荐模型（2026-11-23）

数据延长到 2026-11-23 15:00（236 小时）。评分用时间分块（4 段）× 留一站交叉验证，选值用 1-SE 规则。

```python
from smhi_trv_wind.sensitivity import Engine, Spec, cv, RECOMMENDED
E = Engine(Settings(max_pair_km=150), pd.read_parquet(".../features_variants.parquet"), t_split="2026-11-19 00:00", t_end="2026-11-23 15:00")
cv(E, RECOMMENDED)["mae"].mean()            # 0.698 m/s（第 6 轮模型 0.740）
```

推荐模型 M6 与第 6 轮的区别：近处树木半径 200 → 100 m；新增 1 km 远处森林覆盖率；新增稳定度 ΔT = T_路面 − T_气温；
去掉 lnU×tree 交互项；风速合并改用全部站、权重 1/d^1.5；站点因子 8 个扇区；SMHI 足迹 100 m。风向部分不变。
`pipeline.py` 仍是第 6 轮模型（为了复现）；M6 目前只在 `sensitivity.Engine` 中实现。

## 9. 已知局限

- 只有 9 天检验数据；超参数（60 km、阈值、200 m、L）部分是在同一时段比较后选的，论文中应在训练期固定、到独立时段检验。
- TrV API 只保留约 7 天，长期历史需要 VViS 档案。
- 还没有车辆数据；TrV 不是道路风真值。
