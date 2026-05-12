# V-JEPA 2.1 单帧表征 + DiT 动作头的 LIBERO 训练计划

## 1. 目标与结论

本计划的目标是在当前仓库中实现一条新的、尽量简单且可复现的机器人训练链路：

- 使用 `V-JEPA 2.1` 作为视觉 encoder；
- 只输入 **单帧图像**；
- 不使用语言分支；
- 使用 `starVLA` 中现成的 `DiT` 动作头；
- 直接复用 `LeRobot` / `LIBERO` dataloader；
- 训练目标为 `LIBERO` 上的连续动作预测。

这是一个明确的 `MVP` 路线。它的核心价值不是“最复杂”，而是：

1. 先验证 `V-JEPA 2.1` 的视觉表征是否适合机器人动作预测；
2. 把变量数量压到最低，避免同时引入语言建模、视频时序建模、GR00T flow-matching 等额外不确定性；
3. 尽快跑通训练、验证、推理闭环，为后续扩展到多帧输入或更复杂动作头打基础。

---

## 2. 最终方案（一句话）

**把 `LIBERO` 的单帧观测过 `V-JEPA 2.1 encoder`，得到视觉 token 序列；再通过一个轻量投影层把 token 维度对齐到 `DiT` 动作头需要的条件维度；最后预测未来 `8` 步 `7D` 动作。**

---

## 3. 当前设计边界

本阶段我们明确做和不做的事情：

### 3.1 做的事情

- 单帧图像输入；
- `V-JEPA 2.1` image mode 编码；
- 保留 `LIBERO` 的动作 chunk 监督；
- 尽量保留 `state` 输入；
- 复用 `starVLA` 的训练器和 dataloader；
- 新增一个独立 framework，不破坏现有 `QwenGR00T` / `QwenOFT` 流程。

### 3.2 暂时不做的事情

- 不接语言分支；
- 不使用 Qwen / VLM；
- 不做多帧 clip 输入；
- 不做 JEPA predictor / AC predictor；
- 不做 GR00T flow-matching 头；
- 不做跨任务共训里的 VLM 数据部分。

---

## 4. 为什么这个方案成立

### 4.1 V-JEPA 2.1 能处理单张图像

`app/vjepa_2_1/models/vision_transformer.py` 的 `forward()` 明确支持 `x.ndim == 4` 的图像输入，因此单帧路线在模型定义层面是成立的。

### 4.2 DiT 动作头只需要条件 token，不强依赖语言

`starVLA/model/modules/action_model/DiTActionHeader.py` 的 `ActionModel.forward()` 输入是：

- `gt_action: [B, T, action_dim]`
- `condition: [B, L, D]`

也就是说，只要我们能提供一段条件 token 序列，就可以训练这个头。条件 token 不一定必须来自 VLM，也可以来自 `V-JEPA 2.1`。

### 4.3 LeRobot / LIBERO 已经提供了我们需要的监督

当前 `starVLA` 的 `LIBERO` dataloader 已经能产出：

- `image`
- `action`
- `lang`
- 可选 `state`

因此训练监督和数据组织方式都可以直接复用。

---

## 5. 整体架构设计

新增一个新的 framework，建议命名：

- `JEPA_DiT`

建议放置位置：

- `starVLA/starVLA/model/framework/WM4A/JEPADiT.py`

说明：

- 这里不建议塞进 `VLM4A/`，因为本方案不再依赖 VLM；
- 放进 `WM4A/` 更合理，它本质上是“用视觉表征作为动作建模条件”的变体；
- 也可以新建 `Representation4A/`，但第一版没必要扩大框架结构。

整体数据流如下：

1. `LeRobot/LIBERO dataloader` 输出单条样本：
   - `image`
   - `action`
   - `state`（建议开启）
   - `lang`（先读取但不使用）
2. framework 从样本中取出第一视角图像，组成 batch；
3. 图像经 `V-JEPA 2.1 encoder` 编码为 `[B, N, D_jepa]`；
4. 通过 `LayerNorm + Linear` 先对齐通道维度；
5. 再通过一个 **沿 token 维度压缩的 MLP reducer**，把 `JEPA` patch token 压成 `DiT` 所需的固定数量条件 token；
6. 条件 token 输入 `DiTActionHeader`；
7. 动作头预测未来动作 chunk；
8. 计算 diffusion noise prediction loss。

---

## 6. 关键设计原则

### 6.1 先冻结 encoder，先训头

第一版强烈建议：

- 默认冻结 `V-JEPA 2.1 encoder`
- 只训练 projector + DiT action head

原因：

- 降低显存压力；
- 更容易验证表征是否本身有效；
- 训练更稳定；
- 便于和“随机初始化视觉 backbone”做对比。

后续若 baseline 有效，再尝试：

- 解冻最后若干 block；
- 或全量 finetune。

### 6.2 保留 state，去掉 language

虽然本计划不做语言分支，但我建议：

- `lang` 不参与 forward；
- `state` 尽量保留。

原因：

- `LIBERO` 是带 proprioception 的控制任务；
- 视觉单帧本身可能无法完整表达机械臂当前姿态；
- 保留 state 往往比纯视觉稳定。

### 6.3 图像分辨率做成可配置

当前 `starVLA` 的 dataloader 打包 sample 时会把图像 resize 到 `224x224`。  
但 `V-JEPA 2.1` 的公开 checkpoint 常见是 `384` 分辨率系列。

因此建议：

- 不要把分辨率写死在 dataloader；
- 在 framework 内统一 resize；
- 首选尝试 `384x384`；
- 若显存压力过大，再测试 `224x224`。

---

## 7. 文件与代码改动规划

## 7.1 新增文件

- `starVLA/starVLA/model/framework/WM4A/JEPADiT.py`
- `doc/VJEPA2.1_LIBERO_单帧DiT训练计划.md`

## 7.2 可能新增的辅助文件

- `starVLA/starVLA/model/modules/world_model/VJEPA2_1.py`

说明：

如果希望把 `V-JEPA 2.1` 封装得更干净，可以单独写一个 wrapper。  
如果想先快一点，也可以直接在 `JEPADiT.py` 中内联初始化。

## 7.3 需要修改的文件

- `starVLA/starVLA/model/framework/WM4A/__init__.py`
  - 注册新 framework
- `starVLA/starVLA/config/training/starvla_cotrain_libero.yaml`
  - 新增或复制出 `JEPADiT` 对应配置
- `starVLA/examples/LIBERO/train_files/starvla_cotrain_libero.yaml`
  - 增加一个示例训练配置
- `starVLA/starVLA/dataloader/gr00t_lerobot/datasets.py`
  - 最好把图像 resize 改成可配置，避免写死 `224`

---

## 8. 新 framework 的接口设计

## 8.1 建议配置结构

```yaml
framework:
  name: JEPADiT
  jepa:
    pretrained_ckpt: /share/project/lvjing/models/vjepa2/...
    model_name: vjepa2_1_vit_large_384
    image_size: 384
    freeze_encoder: true
    token_pool: none
    use_cls: false
  projector:
    type: linear
    input_dim: 1024
    output_dim: 768
    use_layernorm: true
  action_model:
    action_model_type: DiT-B
    action_hidden_dim: 768
    action_dim: 7
    state_dim: 8
    future_action_window_size: 7
    action_horizon: 8
    past_action_window_size: 0
datasets:
  vla_data:
    dataset_py: lerobot_datasets
    include_state: true
```
```

说明：

- `action_hidden_dim` 必须和 projector 输出维度一致；
- `state_dim` 要根据最终 dataloader 输出核实；
- 如果 `state` 最终是 8 维，就不要沿用旧配置里的 7。

## 8.2 `forward(examples)` 的职责

输入：

- `examples: List[dict]`

每个样本包含：

- `image`
- `action`
- `state`（可选但建议启用）
- `lang`（忽略）

输出：

- `{"action_loss": loss}`

## 8.3 `predict_action(examples)` 的职责

输入：

- 单条或多条样本

输出：

- `{"normalized_actions": np.ndarray[B, T, action_dim]}`

说明：

- 这部分保持和现有 starVLA framework 对齐；
- 这样可以直接接入现有 evaluation server / evaluation script。

---

## 9. 视觉特征设计

## 9.1 输入形式

虽然 dataloader 输出的是 PIL list，但我们最终只取单帧。

建议规则：

- 先只用主视角第一张图；
- 如果样本有多视角，第一版只保留 `primary_image`；
- wrist image 暂不接入。

## 9.2 encoder 输出

`V-JEPA 2.1` 单帧输入后，输出为 patch token 序列：

- 形状近似为 `[B, N, D_jepa]`

第一版不做复杂 token 选择，建议：

- 直接保留全部 patch tokens；
- 不做平均池化；
- 不做 query pooling；
- 把整段 token 序列作为动作头的条件输入。

这样最符合 DiT 条件建模的使用方式。

## 9.3 projector 设计

建议第一版使用两段式 projector：

- `LayerNorm(D_jepa) -> Linear(D_jepa, D_dit)`
- `token-MLP: Linear(N_patch, H) -> GELU -> Linear(H, N_cond)`

如果效果不稳定，再尝试：

- `Linear -> GELU -> Linear`

这里的关键点是：

- 不使用简单 average pooling；
- 而是让模型学习“哪些 patch token 应该被保留、怎么压成固定数量的条件 token”。

但第一版仍然不建议把 reducer 做得太深。

---

## 10. 动作头设计

本方案明确使用 `starVLA` 现有 `DiTActionHeader`。

训练方式：

1. 从 dataloader 读取动作 chunk；
2. 取最后 `future_action_window_size + 1` 步作为监督；
3. 用 JEPA condition token 做 diffusion noise prediction；
4. 用头内部的 MSE diffusion loss 训练。

说明：

- 这部分尽量不改 `DiTActionHeader` 本身；
- 若有必要，只在 framework 层面对接输入 shape。

---

## 11. 数据侧改动建议

## 11.1 继续复用 `lerobot_datasets`

这是当前方案的一个大优点，尽量不破坏。

## 11.2 开启 `include_state`

建议在 config 中显式设置：

- `include_state: true`

并检查最终 sample 中 `state` 维度。

## 11.3 图像 resize 改成可配置

当前 `datasets.py` 在 `_pack_sample()` 里写死：

- `Image.fromarray(image).resize((224,224))`

建议改为：

- 从 `data_cfg.default_image_resolution` 或 framework config 读取目标尺寸；
- 若未配置，再 fallback 到 `224`。

---

## 12. 分阶段执行计划

## Phase 1：框架骨架搭建

目标：

- 建立 `JEPADiT` framework；
- 能被 `FRAMEWORK_REGISTRY` 正常找到；
- 能初始化 `V-JEPA 2.1 + projector + DiT head`。

验收：

- 单次 import 成功；
- 模型构造成功；
- 打印参数量和模块结构正常。

## Phase 2：前向打通

目标：

- 跑通 `examples -> image -> JEPA -> projector -> DiT loss`。

验收：

- 单 batch forward 成功；
- `action_loss` 为有限值；
- 无 shape mismatch。

## Phase 3：训练闭环

目标：

- 接入现有 `train_starvla.py` / `train_starvla_cotrain.py`；
- 至少能完整训练若干 step；
- 能保存 checkpoint。

验收：

- loss 有下降趋势；
- checkpoint 可正常保存与恢复；
- mixed precision 下训练稳定。

## Phase 4：推理闭环

目标：

- 实现 `predict_action()`；
- 输出 `normalized_actions`；
- 可以对接 `LIBERO` eval interface。

验收：

- 单条样本可返回 `[B, T, 7]`；
- eval 脚本能调用而不报接口错误。

## Phase 5：基线实验

建议至少跑以下三组：

1. `freeze encoder + train projector + DiT`
2. `freeze encoder + no state`
3. `partial unfreeze encoder last k blocks`

验收：

- 至少获得一条稳定可重复的 baseline；
- 能比较 `state` 是否必要；
- 能判断继续解冻 encoder 是否值得。

---

## 13. 风险点与规避

### 风险 1：图像尺寸与 checkpoint 不匹配

问题：

- dataloader 默认 `224`
- `V-JEPA 2.1` checkpoint 常见是 `384`

规避：

- 分辨率做成 config；
- 第一版优先试 `384`；
- 同时记录显存占用和吞吐。

### 风险 2：state 维度与配置不一致

问题：

- 旧配置里常写 `state_dim: 7`
- `LIBERO` dataloader 可能实际输出 `8` 维

规避：

- 在 framework 初始化时打印实际 batch state shape；
- 用 assert 保证 config 和数据一致。

### 风险 3：只用单帧导致任务判别不足

问题：

- 某些操作只靠单帧视觉可能不够

规避：

- 第一版先接受这个边界；
- 如果 baseline 太差，再扩展为 2 帧或短 clip。

### 风险 4：不使用语言会限制多任务区分能力

问题：

- LIBERO 是 instruction-conditioned benchmark

规避：

- 第一版先验证纯视觉 + state 是否够用；
- 如果任务混淆严重，再补文本分支。

---

## 14. 建议的最小实验顺序

1. 先实现 `JEPADiT`，冻结 encoder；
2. 用 `LIBERO goal` 或较小子集跑通；
3. 确认单 batch 和短训练稳定；
4. 再扩展到 `libero_all`；
5. 最后再考虑是否加入语言或多帧。

---

## 15. 验收标准

本计划完成的最低标准不是分数，而是工程闭环：

1. 能从 `LeRobot/LIBERO` 读取样本；
2. 能用 `V-JEPA 2.1` 编码单帧图像；
3. 能把 token 喂给 `DiT` 头并训练；
4. 能输出合法动作 chunk；
5. 能接入 `LIBERO` eval 脚本。

当这五项都完成后，再进入效果优化阶段。

---

## 16. 实现优先级（实际开工顺序）

1. 新建 `JEPADiT.py`
2. 接 `V-JEPA 2.1` checkpoint 加载
3. 增加 projector
4. 复用 `DiTActionHeader`
5. 调整 config
6. 改 dataloader 图像 resize 为可配置
7. 跑单 batch forward
8. 跑短训练
9. 跑 LIBERO eval

---

## 17. 备注

这是一个“先验证 JEPA 表征是否能直接服务动作预测”的计划，不是最终形态。

如果这条线成立，后续自然可以扩展到：

- 加语言；
- 加多帧；
- 换 GR00T / OFT / MLP 头；
- 做 encoder 部分解冻；
- 做跨数据集共训。

但第一版不要同时做这些事。
