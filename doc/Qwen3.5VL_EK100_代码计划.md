# Qwen3.5VL 冻结版 EK100 代码计划

## 1. 目标与范围

本计划的目标是在当前仓库中新增一套独立的 EK100 训练链路，采用：

- 输入 `32` 帧视频；
- 冻结 `Qwen3.5-VL`；
- 使用图像对应 `hidden states` 作为特征；
- 训练 `3-query attentive probe`（`verb/noun/action` 三头）；
- 不引入未来 latent predictor（先做稳定 baseline）。

这是一条“先跑通、再增强”的路线，便于快速得到可复现实验结果。

---

## 2. 目录与文件规划

新增目录：

- `evals/action_anticipation_qwen/`

建议文件：

- `evals/action_anticipation_qwen/eval.py`
- `evals/action_anticipation_qwen/dataloader.py`
- `evals/action_anticipation_qwen/epickitchens.py`
- `evals/action_anticipation_qwen/models.py`
- `evals/action_anticipation_qwen/utils.py`

配置文件：

- `configs/eval_qwen/ek100.yaml`
- `configs/inference_qwen/ek100.yaml`

说明：与现有 `action_anticipation_frozen` 并行，不改原流程，避免影响现有可复现实验。

---

## 3. 关键设计原则

1. **复用 EK100 标注与采样逻辑**  
   复用现有 `epickitchens.py` 的标签映射逻辑（`verb/noun/action`），确保类别空间一致。

2. **改数据输出格式，不改标签定义**  
   采样后输出 `images(list)` 而非 JEPA2 tensor 输入格式。

3. **Qwen 冻结，仅训练 probe**  
   所有 Qwen 参数 `requires_grad=False`，只更新 probe 参数。

4. **接口先统一后扩展**  
   先统一 `tokens=[B,N,D] -> probe` 的接口，后续再加 predictor 或 LoRA。

---

## 4. 数据管线设计

## 4.1 输入与采样

- 与现有 EK100 保持一致：
  - `frames_per_clip=32`
  - `frames_per_second=8`
  - 相同 anticipation 采样策略（用于标签时间对齐）

## 4.2 dataset 输出字段（训练/验证统一）

- `images`: 长度为 32 的帧列表（PIL 或 uint8 tensor）
- `verb`: 原始 verb id
- `noun`: 原始 noun id
- `anticipation_time`: float（保留字段，便于后续扩展）

## 4.3 collate 阶段

- 调用 Qwen3.5-VL processor 打包 batch；
- 输出 `input_ids / pixel_values / image_grid_thw / attention_mask ...`；
- 同时输出 `verb/noun/action` 标签（action 由 `(verb,noun)` 映射）。

---

## 5. 模型与前向设计

## 5.1 FrozenQwenBackbone

职责：

- 封装 Qwen3.5-VL 的前向；
- 返回指定层 hidden states（建议配置项：`vlm_feature_layer`）；
- 从序列中提取图像 token 对应特征，组织成 `[B,N,D]`。

要求：

- 固定一种图像 token 提取规则，全流程保持一致；
- 不混用多种提取路径（避免结果漂移）。

## 5.2 QwenAttentiveClassifier

沿用当前 EK100 probe 设计：

- `AttentivePooler(num_queries=3, depth=num_probe_blocks, num_heads=...)`
- 三个分类头：
  - verb linear
  - noun linear
  - action linear

输出：

- `{"verb": logits, "noun": logits, "action": logits}`

---

## 6. 训练与验证流程

## 6.1 训练策略

- 冻结 Qwen；
- 仅优化 probe；
- 损失：
  - `L_total = L_verb + L_noun + L_action`
  - 支持 CE / focal（与现有配置对齐）

## 6.2 验证指标

沿用现有评估风格：

- `accuracy`
- `ClassMeanRecall(k=5)`（核心看 `action R@5`）

## 6.3 checkpoint

- 保存 probe 权重（可选保存 backbone config 信息）；
- 支持 `val_only=true` 直接加载评估。

---

## 7. 配置项建议（最小集合）

`experiment.classifier`：

- `num_probe_blocks`
- `num_heads`

`experiment.data`：

- `dataset=EK100`
- `base_path`
- `dataset_train`
- `dataset_val`
- `frames_per_clip=32`
- `frames_per_second=8`

`model_kwargs`：

- `base_model_id`（Qwen3.5-VL 路径或HF id）
- `vlm_feature_layer`（默认 -1）
- `freeze_backbone=true`

`optimization`：

- `batch_size`
- `num_epochs`
- `use_bfloat16`
- `multihead_kwargs`

---

## 8. 里程碑与验收标准

## M1：前向打通

- 能完成：`32帧 -> Qwen -> [B,N,D] -> probe`
- 验收：无 shape 错误，单 batch 可 forward。

## M2：训练闭环

- 能跑 1 epoch 训练和验证；
- 验收：loss 正常下降、日志输出完整、checkpoint 可保存。

## M3：基线结果

- 完成标准配置训练；
- 验收：可复现 `action/verb/noun` 指标，得到 Qwen 冻结 baseline。

---

## 9. 风险与规避

1. **图像 token 提取不一致**  
   规避：固定单一路径，写成函数并加 shape assert。

2. **Qwen 版本不一致（2.5/3.5混用）**  
   规避：配置中强制声明 model id，并在启动时打印版本信息。

3. **processor/tokenizer 不匹配**  
   规避：统一从同一路径加载 model+processor。

4. **类别映射偏移**  
   规避：严格复用现有 EK100 标签映射逻辑与过滤逻辑。

---

## 10. 下一步实现顺序（执行建议）

1. 建目录与骨架文件；
2. 先复制并改造 `epickitchens.py + dataloader.py`；
3. 实现 `FrozenQwenBackbone` 与 `QwenAttentiveClassifier`；
4. 接入 `eval.py` 训练循环；
5. 增加 `configs/eval_qwen/ek100.yaml`；
6. 先本地小 batch 跑通，再开完整训练。

