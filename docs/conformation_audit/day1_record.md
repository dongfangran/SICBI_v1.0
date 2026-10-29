# 构象生成流程核查

## 1. 核查信息

- 日期：2026-10-26
- 负责人：张孟懿
- 仓库：https://github.com/dongfangran/SICBI_v1.0
- 分支：`audit/conformation-process`
- 核查仓库代码版本：`84cc196a3b3bacac2d4bbbe0cc76af0637295619`
- 本地仓库目录：`E:\paper\code\v1\SICBI_v1.0`
- 集群项目目录：`/public/home/wanglin3/zhangmy/SICBI`
- 本次操作：
  - 阅读仓库说明与代码。
  - 只读查看集群脚本、历史日志和文件目录。
  - 使用 NumPy 读取 10GS 的代表帧索引、簇帧数和正式状态概率表。
  - 未重新运行 MD、构象处理、dMaSIF 或模型训练。

## 2. 构象生成流程

### 2.1 处理规则

| 步骤 | 代码中的处理规则 | 后续可研究的问题 |
| --- | --- | --- |
| 口袋定义 | 以读取轨迹第一帧的配体重原子几何中心为中心，默认半径为 **12 Å**。蛋白残基只要有一个重原子在该范围内，就保留完整残基；所有帧沿用固定口袋。配体重原子参与后续联合 RMSD 计算。 | 口袋截取范围及参考帧的选择如何影响动态表征与预测？ |
| 轨迹对齐 | 优先使用口袋蛋白主链的 **N、CA、C 原子**，将各帧对齐到读取轨迹的第一帧。若选中的这些原子总数不足 3 个，则回退到口袋蛋白重原子。 | 不同对齐原子集合是否影响聚类结果和预测？ |
| 构象聚类 | 对齐后，以**口袋蛋白重原子＋配体重原子**的联合 RMSD 衡量差异，采用**平均连接层次聚类**，预设分成 **10 个候选簇**。计算聚类距离时使用 `nofit`，不再次拟合坐标。 | 聚类依据、簇数及自适应构象数量如何影响冗余与预测？ |
| 代表帧选择 | 对每个簇，计算簇内各帧在上述联合原子集合上的两两**坐标平方欧氏距离**，选择距离总和最小的实际帧作为代表帧，保存其零基索引。 | 不同代表帧选择准则是否影响构象表征与预测？ |
| 表面特征生成 | 对各代表帧的**完整蛋白**使用固定的预训练 dMaSIF 提取表面点及 **16 维特征**，配体不作为 dMaSIF 输入。随后保留距固定参考配体中心不超过口袋半径的表面点。最终 `input.npy` 每行包含“状态 ID＋16 维特征”，不保存表面点坐标。 | 表面采样的随机性是否影响预测稳定性？特征能否区分相关构象？ |
| 状态合并与占比 | 通过空间近邻匹配后的表面特征平均余弦距离比较状态，该距离具有方向性。默认匹配半径为 **2 Å**，最低有效匹配比例为 **0.5**。按簇帧数从多到少处理；若候选状态到已保留状态的距离小于 `cutoff`，则合并帧数。保留状态重新编号，占比＝归属帧数／总帧数。 | 合并策略、阈值及状态权重如何影响构象覆盖和预测？ |



### 2.2 仓库代码定位

下表路径均相对于仓库根目录。

| 核查内容 | 文件 | 函数 |
| --- | --- | --- |
| 口袋定义、对齐原子 | `data/train_data/data_process/stat.py` | `read_complex` |
| 轨迹对齐、构象聚类 | `data/train_data/data_process/stat.py` | `cluster_frames` |
| 代表帧选择 | `data/train_data/data_process/stat.py` | `medoid_frame` |
| dMaSIF 配置与权重加载 | `data/train_data/data_process/stat.py` | `load_dmasif` |
| 表面特征生成与截取 | `data/train_data/data_process/stat.py` | `protein_patch` |
| 状态间表面距离 | `data/train_data/data_process/stat.py` | `state_distance` |
| 状态合并、编号与占比 | `data/train_data/data_process/protein_data.py` | `finalize` |

## 3. 集群处理脚本与参数核查

### 3.1 候选构象与表面特征生成入口

查看的作业脚本：

`/public/home/wanglin3/zhangmy/SICBI/SICBI_proteindata/SICBI_stat_1.pbs`

脚本指定：

- Conda 环境：`dMaSIF_new`
- 处理程序：`stat_1.py`
- cpptraj：`/public/home/wanglin3/software/miniconda3/envs/amber_env/bin/cpptraj`
- 使用 `cd "$PBS_O_WORKDIR"` 进入作业提交目录。

当前 `stat_1.py` 的默认参数：

| 参数 | 默认值 |
| --- | --- |
| MD 轨迹 | `data/MD.hdf5` |
| 拓扑等配套文件压缩包 | `data/parameter_restart_files_MD.tar.gz` |
| dMaSIF 权重 | `models/dMaSIF_search_3layer_12A_16dim` |
| 本分片缓存 | `stat_cache/shard_00000_03000` |
| 口袋半径 | `12 Å` |
| 近邻匹配半径 | `2 Å` |
| 最低有效匹配比例 | `0.5` |
| 预处理随机种子 | `0` |

说明：

- 上述文件默认路径相对于作业运行目录解析。
- 当前 PBS 命令没有显式覆盖口袋半径、匹配半径、最低匹配比例和随机种子。
- 这些参数依据当前脚本与 PBS 内容确定；尚未找到完整历史参数快照。
- 10GS 实际缓存位于最后一个分片，因此不能用 `stat_1.py` 单独证明它的全部历史处理参数。

### 3.2 状态合并入口

查看到两套当前作业配置：

| 作业脚本 | 调用程序 | 显式指定阈值 |
| --- | --- | --- |
| `SICBI_aggregate_000.pbs` | `protein_data_000.py` | `--cutoff 0` |
| `SICBI_aggregate_003.pbs` | `protein_data_003.py` | `--cutoff 0.03` |

以上文件位于：

`/public/home/wanglin3/zhangmy/SICBI/SICBI_proteindata`

两个程序的状态合并逻辑一致，默认输出目录分别为 `protein_output_000` 和 `protein_output_003`。

## 4. 历史训练记录与正式输入定位

### 4.1 与历史结果一致的训练日志

找到以下日志：

`/public/home/wanglin3/zhangmy/SICBI/SICBI_new/model_1_000/SICBI_training_1_000_seed.o11004044`

| 项目 | 日志记录 |
| --- | --- |
| 最佳验证损失 | `0.670749` |
| 最佳轮次 | 第 19 轮 |
| 停止轮次 | 第 25 轮 |
| 早停耐心值 | 连续 6 轮未改善 |
| 训练耗时 | `1592` 秒 |
| 退出状态 | `0` |

这些记录与已有模型的历史训练记录一致。

日志记录的最佳权重路径：

`/public/home/wanglin3/zhangmy/SICBI/SICBI_new/model_1_000/outputs/best_model_seed.pt`

尚未核对该文件与提交版本权重的文件内容是否完全一致。

### 4.2 对应训练脚本

`/public/home/wanglin3/zhangmy/SICBI/SICBI_new/model_1_000/train_seed.py`

当前脚本中的配置：

| 项目 | 配置 |
| --- | --- |
| 数据清单 | `ROOT.parent / "data_000" / "manifest.csv"` |
| Batch size | `4` |
| 最大训练轮数 | `100` |
| 初始学习率 | `1e-4` |
| 最低学习率 | `1e-6` |
| Warmup 比例 | `0.10` |
| Weight decay | `1e-5` |
| 早停耐心值 | `6` |
| 随机种子 | 从 `SICBI_SEED` 读取，未设置时默认 `42` |

### 4.3 正式输入路径

当前训练脚本指定的数据清单：

`/public/home/wanglin3/zhangmy/SICBI/SICBI_new/data_000/manifest.csv`

该模型目录的 `pipeline.py`：

- 以 manifest 所在目录为起点解析输入路径。
- 允许每个样本包含 **1–10 个状态**。
- 不要求所有样本必须包含 10 个状态。

示例样本 10GS 的输入文件：

- 特征：`/public/home/wanglin3/zhangmy/SICBI/SICBI_new/data_000/proteins/10GS/input.npy`
- 状态占比：`/public/home/wanglin3/zhangmy/SICBI/SICBI_new/data_000/proteins/10GS/state_probabilities.npy`

两个文件均已确认存在。

注意：项目根目录还有一份要求 60 个状态的 `pipeline.py`，与本次定位的 `model_1_000` 版本不同，本次核查以对应模型目录中的代码为准。

## 5. 10GS 单样本核验

### 5.1 候选构象缓存

缓存目录：

`/public/home/wanglin3/zhangmy/SICBI/SICBI_proteindata/stat_cache/shard_12000_final/10GS`

已找到：

| 文件 | 内容 |
| --- | --- |
| `candidate_input.npy` | 候选构象的表面特征 |
| `populations.npy` | 各候选簇包含的帧数 |
| `representative_frames.npy` | 各候选构象的代表帧索引 |
| `state_distance.npy` | 候选状态之间的表面特征距离 |

### 5.2 代表帧与占比

以下按候选缓存顺序列出：

| 候选序号 | 代表帧索引，从 0 开始 | 簇内帧数 | 簇帧数／总帧数 |
| --- | --- | --- | --- |
| 1 | 56 | 50 | 0.50 |
| 2 | 8 | 26 | 0.26 |
| 3 | 38 | 9 | 0.09 |
| 4 | 53 | 4 | 0.04 |
| 5 | 83 | 4 | 0.04 |
| 6 | 28 | 2 | 0.02 |
| 7 | 36 | 2 | 0.02 |
| 8 | 0 | 1 | 0.01 |
| 9 | 1 | 1 | 0.01 |
| 10 | 86 | 1 | 0.01 |

正式状态概率表：

| 最终状态 ID | 占比 |
| --- | --- |
| 1 | 0.50 |
| 2 | 0.26 |
| 3 | 0.09 |
| 4 | 0.04 |
| 5 | 0.04 |
| 6 | 0.02 |
| 7 | 0.02 |
| 8 | 0.01 |
| 9 | 0.01 |
| 10 | 0.01 |

### 5.3 核验结果与解释范围

- 候选簇帧数合计为 **100**。
- 代表帧索引共 **10** 个，互不重复，且均处于 `0–99` 范围内。
- 正式概率表包含 **10** 个状态。
- 正式占比与候选簇帧数除以总帧数的结果按当前顺序一致。
- 正式占比总和为 `0.99999994`，属于正常浮点精度误差。
- 结果与未发生状态合并的情况一致，但不能单独证明使用了 `cutoff=0`。
- 正式状态特征与候选特征尚未逐状态核对，因此尚未完成“代表帧—候选特征—最终状态 ID”的完整对应核验。
- 帧数合计 100 仅说明此次聚类记录覆盖 100 帧，不能据此确定原始模拟总帧数、时长或保存间隔。
- 尚未读取逐帧聚类归属，不能确认各代表帧确实属于对应簇，或重新验证代表帧选择计算。

### 5.4 初步观察

- 前 3 个候选簇累计占比为 **85%**。
- 最后 3 个候选簇各占 **1%**。
- 该现象可为后续构象数量与选择策略实验提供依据。
- 当前尚不能判断低占比构象是否冗余或是否具有预测价值。

## 6. 待确认事项

- [ ] 核对 10GS 的正式特征与候选特征是否逐状态一致。
- [ ] 建立候选 ID、代表帧索引与最终状态 ID 的明确对应表。
- [ ] 核查 10GS 所属分片的处理脚本、参数及运行日志。
- [ ] 追踪候选缓存生成正式 `data_000` 输入的过程，确认实际合并阈值。
- [ ] 确认轨迹第一读取帧对应的模拟时刻、帧间隔及是否经过抽帧。
- [ ] 确认是否保留逐帧聚类归属、代表帧结构及表面点坐标。
- [ ] 后续将输入检查扩展到整个数据集，统计缺失、异常、状态数和表面点数。
