# 尼罗罗非鱼 (Oreochromis niloticus) Biolith 占有模型分析

# VeryBiolith
本项目使用 [Biolith](https://github.com/timmh/biolith) 包对尼罗罗非鱼的出现数据进行完整的贝叶斯占有模型分析。

## 绘制 Figure S3 风格鱼类受威胁等级地图

该仓库提供了一个可复用脚本 `scripts/plot_threat_maps.py`，用于根据水文分区（如 HydroBASINS）与各分区的鱼类受威胁等级统计（VU、EN、CR），绘制与论文 Figure S3 类似的三联图。

### 数据准备
- 水文分区矢量数据（Shapefile 或 GeoPackage），例如 HydroBASINS 中国区域。需包含分区唯一 ID（默认 `HYBAS_ID`）。
- 河流网络（可选），例如 HydroRIVERS（用于叠加河流白色线条）。
- 物种统计表 CSV：至少包含列 `HYBAS_ID`（或你自定义的分区 ID 列）、`VU`、`EN`、`CR`，分别为每个分区的受威胁物种数。

示例 CSV（列名仅示例）：
```
HYBAS_ID,VU,EN,CR
123456,4,1,0
123457,7,2,1
...
```

### 安装依赖
建议使用 Python 3.9+，并安装：
```
pip install geopandas matplotlib pandas numpy matplotlib-scalebar
```

### 运行脚本
在仓库根目录执行：
```
python scripts/plot_threat_maps.py --hydro <path/to/hydrobasins.shp> \
  --species <path/to/fish_threat_by_basin.csv> \
  --rivers <path/to/hydrorivers.shp> \
  --id_col HYBAS_ID \
  --output outputs/figure_s3.png
```

脚本会：
- 读取并按分区 ID 连接分区与统计表；缺失值按 0 处理。
- 重投影到 `EPSG:3857`，便于快速绘图。
- 分别在三行子图绘制 `VU`、`EN`、`CR`，使用 `RdYlBu_r` 渐变色并叠加河流线。
- 自动添加“Species Richness”水平颜色条与简易北箭头。
- 将输出保存到 `outputs/figure_s3.png`。

### 常见定制
- 更换配色：在 `plot_category(..., cmap="RdYlBu_r")` 修改为 `Spectral_r` 或自定义 cmap。
- 设定统一上限：把 `vmax` 固定为论文中的最大值，避免不同子图刻度不一致。
- 改投影：如需等面积投影，可替换为 Albers（部分系统需安装额外投影库）。
- 添加底图或注记：可以用 `contextily` 加瓦片底图，或在图上标注关键区域。

### 数据来源参考
- HydroBASINS / HydroSHEDS: https://www.hydrosheds.org/
- HydroRIVERS: https://www.hydrosheds.org/products/hydrorivers

> 若你已有处理好的数据文件，直接按上面命令运行即可生成与论文风格接近的三联地图。

## 项目概述

本项目实现了 Biolith 开源项目的完整使用案例，包括：
- 多源数据整合 (GBIF, FishNet2, iDigBio)
- 数据清理和标准化
- 空间网格化处理
- 贝叶斯占有模型拟合
- 结果可视化和报告生成

## 数据来源

- **GBIF**: 全球生物多样性信息设施 (88,000+ 条记录)
- **FishNet2**: 鱼类标本数据库 (700+ 条记录)
- **iDigBio**: 综合数字化生物收藏 (600+ 条记录)

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：
- `biolith>=0.0.9` - 核心建模包
- `pandas`, `numpy` - 数据处理
- `matplotlib`, `seaborn` - 可视化
- `jax`, `numpyro` - 贝叶斯推断
- `jupyter` - 交互式分析

### 2. 检查 JAX 安装

如果你有 GPU 并想加速计算，可以安装 GPU 版本的 JAX：

```bash
# CUDA 12
pip install -U "jax[cuda12]"

# CUDA 11
pip install -U "jax[cuda11]"
```

## 使用方法

### 方式一：运行 Python 脚本（推荐用于快速分析）

```bash
python run_analysis.py
```

这将自动执行完整的分析流程并生成结果文件。

### 方式二：使用 Jupyter Notebook（推荐用于交互式分析）

```bash
jupyter notebook "尼罗罗非鱼占有模型分析.ipynb"
```

Notebook 包含详细的说明和可视化，适合学习和探索。

## 分析流程

1. **数据加载和清理** 
   - 读取三个数据源的 CSV 文件
   - 标准化字段名称和格式
   - 处理缺失值和异常数据

2. **探索性数据分析**
   - 绘制全球分布地图
   - 分析时间趋势
   - 统计描述性指标

3. **空间网格化**
   - 创建 30×30 的空间网格
   - 将观测数据分配到网格单元
   - 汇总站点级别信息

4. **占有模型准备**
   - 构建观测矩阵 (站点 × 访问)
   - 添加环境协变量（纬度、经度）
   - 标准化协变量

5. **Biolith 模型拟合**
   - 使用 NumPyro 进行 MCMC 采样
   - Warmup: 1000 次迭代
   - Samples: 2000 次迭代

6. **结果分析**
   - 提取占有概率 (ψ) 和检测概率 (p)
   - 计算可信区间
   - 分析协变量效应

7. **可视化和报告**
   - 后验分布图
   - 空间占有概率图
   - MCMC 诊断图
   - 生成分析报告

## 输出文件

运行分析后会生成以下文件：

### 图表文件
- `distribution_map.png` - 全球分布和密度热力图
- `model_results.png` - 模型参数后验分布
- `spatial_occupancy.png` - 空间占有概率预测
- `occupancy_results.png` - 简化结果图（脚本版本）

### 数据文件
- `cleaned_occurrence_data.csv` - 清理后的原始数据
- `site_summary.csv` - 站点级别汇总统计
- `analysis_report.txt` - 文本格式分析报告

## 模型说明

### 占有模型 (Occupancy Model)

占有模型是一个层次贝叶斯模型，用于估计物种在给定区域的真实占有状态，同时考虑检测不完美的问题。

**模型结构：**

1. **占有过程** (生态过程)
   ```
   z_i ~ Bernoulli(ψ_i)
   logit(ψ_i) = β₀ + β₁×latitude + β₂×longitude
   ```
   - `z_i`: 站点 i 的真实占有状态 (0 或 1)
   - `ψ_i`: 站点 i 的占有概率

2. **检测过程** (观测过程)
   ```
   y_ij ~ Bernoulli(z_i × p_ij)
   logit(p_ij) = α₀ + α₁×latitude + α₂×longitude
   ```
   - `y_ij`: 站点 i 第 j 次访问的观测结果
   - `p_ij`: 给定物种存在时的检测概率

### 关键参数

- **ψ (占有概率)**: 物种在站点真实存在的概率
- **p (检测概率)**: 在物种存在的情况下，单次调查检测到它的概率
- **β (占有协变量系数)**: 环境变量对占有的影响
- **α (检测协变量系数)**: 环境变量对检测的影响

## 结果解读示例

```
占有概率 (ψ): 0.857 [0.823, 0.889]
检测概率 (p): 0.423 [0.398, 0.449]
```

**解释：**
- 在研究区域内，尼罗罗非鱼平均在约 85.7% 的站点存在
- 在物种存在的站点，单次调查检测到它的概率约为 42.3%
- 这意味着需要多次调查才能可靠地确认物种的缺失

## 项目结构

```
VeryBiolith/
├── data/
│   └── Oreochromis niloticus OCC/
│       ├── gbif_occurrence.csv
│       ├── fishnet2_occurrence.csv
│       └── idigbio_occurrence.csv
├── 尼罗罗非鱼占有模型分析.ipynb  # Jupyter Notebook
├── run_analysis.py               # Python 脚本
├── requirements.txt              # 依赖包列表
├── README.md                     # 项目说明
└── LICENSE                       # 许可证
```

## 关于 Biolith

[Biolith](https://github.com/timmh/biolith) 是一个用 Python 编写的贝叶斯生态建模包，具有以下特点：

- ✅ **易于修改**: 模型容易理解和实现，无需推导似然函数
- ⚡ **快速**: 支持 GPU 加速，模型拟合速度快
- 🐍 **Python 生态**: 完全用 Python 编写，易于集成
- 🔬 **现代工具**: 基于 NumPyro 和 JAX，采用现代贝叶斯推断方法

### 引用

如果你在研究中使用了 Biolith，请引用：

```bibtex
@software{biolith,
  author = {Haucke, Timm},
  title = {Biolith: Bayesian Ecological Modeling in Python},
  url = {https://github.com/timmh/biolith},
  year = {2025}
}
```

## 常见问题

### Q: 模型运行很慢怎么办？

A: 
1. 减少网格分辨率（修改 `lat_bins` 和 `lon_bins`）
2. 减少 MCMC 迭代次数
3. 安装 GPU 版本的 JAX

### Q: 如何添加更多协变量？

A: 在 Notebook 的第 18 个单元格中修改 `X_occ` 和 `X_det` 矩阵，添加额外的列。

### Q: 如何解释协变量系数？

A: 系数在 logit 尺度上。正系数表示该变量增加时占有/检测概率增加；负系数则相反。

## 贡献

欢迎提出问题和改进建议！

## 许可证

MIT License

## 致谢

- Biolith 项目：https://github.com/timmh/biolith
- 数据来源：GBIF, FishNet2, iDigBio

---

**作者**: VeryBiolith 项目组  
**日期**: 2025年10月  
**版本**: 1.0
