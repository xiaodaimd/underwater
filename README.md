# SeaDiff Underwater Enhancement

本项目整理 SeaDiff 水下图像增强的最终训练与推理代码，包含三个阶段：

1. 400k 教师：从零训练 SeaDiff 扩散教师；
2. E3 动态物理教师：冻结 Beta 网络，只微调去噪网络，把物理先验随扩散时间步逐步注入；
3. 单步 Student：在冻结的 E3 骨干上加入九个残差 Adapter，用双层优化搜索全局门控，选出 Top-3 后固定并蒸馏微调，最终一次 U-Net 前向完成增强。

仓库只保留最终训练链路上的代码、配置和数据清单，不包含历史实验、旧脚本、日志、权重或指标结果。

代码阅读与数据流说明见 **[动态物理先验与渐进蒸馏：代码及数据流](docs/CODE_AND_DATA_FLOW_ZH.md)**，包含函数入口、张量维度、梯度方向、阶段教师交接、训练/部署差别及当前实现边界。核心源码内同步补有中文注释。

## 方法概览

### 400k 教师

教师由 `DocDiff` 中的两个子网络组成：

- `Denoise_UNet`：12 通道输入，3 通道输出，预测 `x0`；
- `Beta_UNet`：3 通道水下图输入，预测物理参数图。

输入顺序固定为 `[x_t, U, H, J]`：

- `x_t`：第 `t` 步噪声状态；
- `U`：水下图；
- `H`：RGB-uv 直方图条件；
- `J`：由深度与 Beta 网络计算的物理先验。

训练使用线性 beta 日程（T=1000，1e-6 到 0.02）和 x0 预测。损失为均方误差加 0.1 倍 VGG19 感知损失（relu1_2、relu2_2、relu3_3，等权逐层 MSE）。Beta 网络启用 `beta_norm` 数值稳定化。教师从 `scripts/make_teacher_init.py` 生成的确定性共享初始化开始训练 400,000 iterations。

### E3 动态物理教师

E3 不改变网络结构、扩散过程或损失，只改变条件槽位：

```text
H_slot = 0
J_t    = U + w(t) * (J - U)
w(t)   = 0.1 + 0.9 * (1 - alpha_bar_t)^2
```

每个时间步的输入为 `[x_t, U, 0, J_t]`。E3 从 400k 教师的 denoiser 与 Beta 权重初始化，冻结 Beta，只训练 denoiser 10,000 iterations。`w(t)` 是物理先验的输入插值强度，不是网络内部注意力权重。

注意时间方向：反向采样 `999 → 0` 时，`w(t)` 从约 1 降到 0.1，`J_t` 逐渐靠近 `U`。静态 `J` 只由水下图及其深度/Beta/环境光生成，不根据当前 `x_t` 重新估计。

### Student Adapter 蒸馏

`OneStepSeaDiffStudent` 深拷贝 E3 denoiser 作为全程冻结的骨干，在九个位置插入残差 Adapter：

```text
enc1, enc2, enc3, enc4, mid, dec4, dec3, dec2, dec1
```

每个 Adapter 为 `1x1 降维 -> SiLU -> 1x1 升维`，bottleneck 为 `C/4`，升维权重零初始化。前向为：

```text
h_out = h + gate * adapter(h)
```

九个位置各有一个独立 sigmoid 门控。训练分三段：

- 预热 5 epochs：门控固定为 0.5，只训练 Adapter；
- 搜索 55 epochs：内层更新 Adapter，外层每 4 个内层 batch 用 BiLaLoRA 式 (7) 超梯度更新门控；
- 微调 40 epochs：按第 60 epoch 的门控值选 Top-3，固定选中位置与门控值，只用 600 对完整训练集微调选中 Adapter。

蒸馏目标由教师 DDIM 链重建，学生损失为：

```text
L_total = L_original + 0.5 * L_distill
L_distill = mean MSE(student_x0, stopgrad(teacher_x0_target))
```

学生最终只做一次 U-Net 前向。

这里的“一次前向”指部署时的 U-Net 调用；训练仍包含教师链、多次学生前向和梯度计算。搜索中的 `1000 → 500 → … → 1` 是网格点数逐级减少；每个样本从当前学生网格随机选择区间，训练 `x_t` 由 GT 加噪得到，部署则从纯噪声开始。

## 目录结构

```text
configs/
  teacher_400k.yaml          400k 教师训练与推理配置
  teacher_e3.yaml            E3 微调与推理配置
  student.yaml               Student 搜索/微调共用配置
  data/                      UIEB 600/130 与 inner480/dev120 清单
scripts/
  make_teacher_init.py       重建确定性共享初始化（含 SHA 校验）
  prepare_conditions.py      从水下图生成 V1 深度与 RGB-uv 直方图
  train_teacher_400k.py      400k 教师训练入口
  train_teacher_e3.py        E3 教师微调入口
  train_student_search.py    预热 + 双层搜索 + Top-3 选择
  train_student_finetune.py  固定 Top-3 的 40 轮微调
  infer_teacher.py           400k / E3 教师推理
  infer_student.py           单步 Student 推理与可选指标
  export_student.py          导出只含 Top-3 Adapter 的部署权重
src/seadiff_distill/
  models/                    U-Net、Adapter、门控、权重加载
  diffusion/                 线性扩散日程、DDPM/DDIM 采样器
  data/                      清单读取、图像/条件打包、物理先验
  losses/                    教师损失、感知损失、蒸馏损失
  teacher/                   400k 与 E3 训练器、动态条件
  student/                   双层优化、教师链、Top-3、训练器
  evaluation/                稳定初始噪声、PSNR/SSIM/L1
  config.py                  YAML/JSON 读取与路径解析
  determinism.py             RNG 与确定性设置
third_party/SeaDiff/         上游 Depth Anything V1 与 RGBuvHistBlock
```

## 环境安装

建议 Python 3.10+、PyTorch 2.x、单卡 CUDA。安装 PyTorch 时请按官方说明选择与本机 CUDA 匹配的版本，然后安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

`opencv-python-headless` 用于 Depth Anything 预处理；`scikit-image` 只在计算 SSIM 时需要。

## 数据准备

训练清单为 CSV，固定五列：

```text
sample_id,underwater,reference,depth,histogram
```

所有路径都是相对于 `data.data_root` 的 POSIX 路径。`depth` 为 FP32 `[336,336]`、逐图 min-max 到 `[0,1]`；`histogram` 为 FP32 `[3,336,336]`、总和为 1。

仓库内的清单对应如下数据布局（`data_root` 默认为 `data/`）：

```text
data/
  datasets/UIEB官方数据集/UIEB_600train/...
  datasets/UIEB官方数据集/UIEB_130test/...
  conditions/uieb_train600/{depth,histogram}/*.npy
  conditions/uieb_test130_256/{depth,histogram}/*.npy
```

如果使用自己的目录，请修改清单中的路径，或用 `--data-root` 指向新的根目录。条件文件可由清单中的 `depth`/`histogram` 目标路径生成：

```bash
python scripts/prepare_conditions.py \
  --manifest configs/data/uieb_train600.csv \
  --data-root data \
  --depth-weights weights/depth_anything_v1_small.pth \
  --device cuda:0
```

深度模型固定为 Depth Anything **V1 Small**，官方权重 SHA256 为 `59afe57cfd9f4284deaf5b753d954723d7136ae842fba3e068ba03537ca1e60e`。不要换成 V2；两者的实现与权重不兼容。

## 训练

所有命令都在仓库根目录执行。配置中的相对路径按仓库根目录解析。

### 1. 400k 教师

先生成共享初始化：

```bash
python scripts/make_teacher_init.py
```

该脚本会重建两次并检查逐位一致，输出 `weights/shared_init/teacher_beta_norm_seed3407.pt`，SHA256 必须为 `f6c228b944a548edaf98a7194e294d767d99f9a068242f4572b83699e0dc00ec`。

```bash
python scripts/train_teacher_400k.py \
  --config configs/teacher_400k.yaml \
  --device cuda:0
```

最终权重写入：

```text
outputs/teacher_400k/weights/model_denoiser_400000.pth
outputs/teacher_400k/weights/model_beta_predictor_400000.pth
```

断点恢复使用 `--resume outputs/teacher_400k/teacher_last.pt`。

### 2. E3 动态物理教师

`configs/teacher_e3.yaml` 默认从上面的 400k 最终权重初始化，并校验其 SHA256。

```bash
python scripts/train_teacher_e3.py \
  --config configs/teacher_e3.yaml \
  --device cuda:0
```

最终权重写入：

```text
outputs/teacher_e3/weights/model_denoiser_10000.pth
outputs/teacher_e3/weights/model_beta_frozen_10000.pth
```

### 3. Student 搜索与微调

搜索阶段：

```bash
python scripts/train_student_search.py \
  --config configs/student.yaml \
  --device cuda:0
```

该命令完成 5 轮预热、55 轮双层搜索和 Top-3 选择，输出：

```text
outputs/student_adapter/checkpoints/selected_top3.pt
outputs/student_adapter/selection_details.json
```

固定选择微调：

```bash
python scripts/train_student_finetune.py \
  --config configs/student.yaml \
  --device cuda:0
```

默认从 `selected_top3.pt` 恢复，训练 40 轮，输出 `outputs/student_adapter/checkpoints/finetune_epoch100.pt`。搜索与微调共用同一份 `configs/student.yaml`，不要在两次运行之间修改配置，否则断点哈希校验会失败。

导出部署权重：

```bash
python scripts/export_student.py \
  --config configs/student.yaml
```

输出 `outputs/student_adapter/deployment/top3_student.pt`，并验证导出模型与训练 checkpoint 学生输出一致。

## 推理

### 教师

E3 动态物理教师（默认配置）：

```bash
python scripts/infer_teacher.py \
  --config configs/teacher_e3.yaml \
  --manifest configs/data/uieb_test130.csv \
  --data-root data \
  --sampler ddim --steps 50 \
  --output-dir outputs/infer_teacher_e3 \
  --device cuda:0
```

400k 静态条件教师：

```bash
python scripts/infer_teacher.py \
  --config configs/teacher_400k.yaml \
  --manifest configs/data/uieb_test130.csv \
  --data-root data \
  --sampler ddim --steps 50 \
  --output-dir outputs/infer_teacher_400k \
  --device cuda:0
```

E3 推理每个时间步重建 `[x_t, U, 0, J_t]`；400k 推理使用固定的 `[x_t, U, H, J]`。初始噪声由 `sample_id` 和 `--seed` 确定性生成。

### Student

```bash
python scripts/infer_student.py \
  --config configs/student.yaml \
  --deployment outputs/student_adapter/deployment/top3_student.pt \
  --manifest configs/data/uieb_test130.csv \
  --data-root data \
  --output-dir outputs/infer_student \
  --device cuda:0
```

也可以直接加载训练 checkpoint：

```bash
python scripts/infer_student.py \
  --config configs/student.yaml \
  --checkpoint outputs/student_adapter/checkpoints/finetune_epoch100.pt \
  --manifest configs/data/uieb_test130.csv \
  --data-root data \
  --output-dir outputs/infer_student_ckpt \
  --device cuda:0
```

在含 `reference` 的清单上可输出指标：

```bash
python scripts/infer_student.py \
  --config configs/student.yaml \
  --deployment outputs/student_adapter/deployment/top3_student.pt \
  --manifest configs/data/uieb_test130.csv \
  --data-root data \
  --output-dir outputs/infer_student \
  --metrics-json outputs/student_metrics.json \
  --ssim \
  --device cuda:0
```

Student 每次推理恰好执行一次 U-Net 前向，输入为 `[x_999, U, 0, J_999]`，输出经过 `clamp(0,1)` 保存。

## 权重与加载方式

| 权重 | 作用 | 加载方式 |
| --- | --- | --- |
| `weights/shared_init/teacher_beta_norm_seed3407.pt` | 400k 教师共享初始化 | `TeacherTrainer` 校验文件 SHA 后加载 state_dict |
| `outputs/teacher_400k/weights/model_denoiser_400000.pth` | 400k denoiser | `load_seadiff_teacher` 严格校验 SHA 与 FP32 |
| `outputs/teacher_400k/weights/model_beta_predictor_400000.pth` | 400k Beta 网络 | 同上 |
| `outputs/teacher_e3/weights/model_denoiser_10000.pth` | E3 denoiser | Student 骨干与 E3 推理共用 |
| `outputs/teacher_e3/weights/model_beta_frozen_10000.pth` | E3 冻结 Beta | 与 400k Beta 内容一致 |
| `outputs/student_adapter/checkpoints/finetune_epoch100.pt` | 完整训练 checkpoint | `load_student_training_checkpoint` |
| `outputs/student_adapter/deployment/top3_student.pt` | 只含 Top-3 Adapter 的部署权重 | `load_deployment_student` |

教师权重加载器只接受原始 FP32 state_dict，不接受外层 checkpoint 字典，并逐文件核对 SHA256。部署权重加载器只构造选中的 Adapter，骨干仍由 E3 denoiser 提供。

## 复现约束

- 模型计算、损失函数和超参数保持最终训练版本，不在脚本中改写。
- 教师、学生和指标计算全程 FP32。
- Adapter 搜索阶段内层与外层使用固定学习率、固定更新频率和固定超参数，不做额外调参。
- Top-3 只在第 60 epoch 由门控值确定一次；微调阶段门控不再更新。
- 测试集不参与训练、搜索或阈值选择。
