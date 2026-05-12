# V-JEPA 2.1 + Galbot G1 三视角训练实施计划

## 目标

在 Galbot G1 小数据集 `/share/project/zhangningboo/galbot_g1_dataset/0_original/4_diewan_new` 上训练当前 `JEPADiT` 框架。

目标模型结构保持当前主线：

```text
三路 RGB 图像
  -> 共享 frozen V-JEPA 2.1 vitG 384 image encoder
  -> token 维拼接
  -> QwenGR00T 同构 GR00T_ActionHeader
  -> 8-step action chunk
```

约束：

- 使用 V-JEPA 2.1 的 2D image encoder 路径。
- 不使用 3D/video encoder。
- 三个视角分别编码，不把三张图拼成一张大图。
- JEPA encoder 共享参数并冻结。
- 第一版先做最小可训练版本，不引入 view embedding 或 token pooling。

## 当前 Galbot 数据结论

有效数据不在顶层 `meta/*.bin`，而在每个轨迹包的 processed 目录：

```text
/share/project/zhangningboo/galbot_g1_dataset/0_original/4_diewan_new/{2015,2016,2017,2019}/proc/CoRobot/Galbot_G1_stack_bowl_1_20xx
```

每个 processed dataset 都是 LeRobot 风格：

```text
data/chunk-000/episode_xxxxxx.parquet
videos/chunk-000/<video_key>/episode_xxxxxx.mp4
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/episodes_stats.jsonl
```

四个包一致：

```text
Galbot_G1_stack_bowl_1_2015: 50 episodes, 21585 frames
Galbot_G1_stack_bowl_1_2016: 50 episodes, 22567 frames
Galbot_G1_stack_bowl_1_2017: 50 episodes, 21004 frames
Galbot_G1_stack_bowl_1_2019: 50 episodes, 25070 frames
```

总量：

```text
200 episodes
90226 frames
```

低维字段：

```text
action: shape [38]
observation.state: shape [38]
```

可用视频字段：

```text
observation.images.image_head_right
observation.images.image_head_left
observation.images.image_arm_right
observation.images.image_arm_left
```

任务文本：

```text
Galbot_G1_stack_bowl_1
```

注意点：

- `meta/info.json` 中 `codebase_version` 是 `v2.1`。
- 当前 StarVLA dataloader 显式支持 `v2.0` 和 `v3.0`，但该数据文件布局实际兼容 `v2.0`。
- 目录中目前缺少 `meta/modality.json`。
- 当前 LIBERO 配置是 7 维 action/state，不能直接用于 Galbot。

## 视角选择

按当前需求使用三个视角：

```text
双手:
  observation.images.image_arm_left
  observation.images.image_arm_right

左眼:
  observation.images.image_head_left
```

StarVLA 内部建议命名：

```text
video.arm_left
video.arm_right
video.head_left
```

第一版 `JEPADiT` 只需要把 dataloader 输出整理成：

```python
example["image"] = [arm_left, arm_right, head_left]
```

## 模型改造方案

当前 `JEPADiT` 只取第一张图：

```python
batch_images = [example["image"][0] if isinstance(example["image"], list) else example["image"] for example in examples]
vl_embs = self._extract_jepa_tokens(batch_images)
```

需要改成支持多视角：

```text
example["image"] = [view0, view1, view2]
```

实现建议：

1. 新增配置项：

```yaml
framework:
  jepa:
    image_views: 3
    multi_view_fusion: concat_tokens
```

2. 改造 `_extract_jepa_tokens()`：

```text
输入: List[List[image]]
对每个 view 单独预处理
共享同一个 jepa_encoder forward
输出 token 按 dim=1 拼接
```

目标 tensor shape：

```text
单视角:
  [B, N, 1664]

三视角:
  [B, 3N, 1664]
```

重要点：

- `cross_attention_dim` 仍然是 `1664`。
- 拼接发生在 token 维，不是 channel 维。
- 不需要改成 `3 * 1664`。
- JEPA encoder 参数仍然共享且冻结。

3. 训练和推理都使用同一套多视角逻辑。

4. 第一版不加 view embedding。

后续可选优化：

- 每个视角加 learnable view embedding。
- 每路 token 做 pooling 或 token compressor，减少 cross-attention token 数。
- 对不同视角做随机 dropout，提高鲁棒性。

## 数据适配方案

建议新建 Galbot example：

```text
starVLA/examples/Galbot/train_files/
```

需要新增：

```text
data_registry/data_config.py
jepadit_train_galbot.yaml
run_jepadit_train_galbot.sh
modality.json
```

### 数据 root

建议不要直接改原始数据。建立训练用 root，使用软链接指向 processed dataset：

```text
starVLA/playground/Datasets/GALBOT_G1_DIEWAN/
  Galbot_G1_stack_bowl_1_2015 -> .../2015/proc/CoRobot/Galbot_G1_stack_bowl_1_2015
  Galbot_G1_stack_bowl_1_2016 -> .../2016/proc/CoRobot/Galbot_G1_stack_bowl_1_2016
  Galbot_G1_stack_bowl_1_2017 -> .../2017/proc/CoRobot/Galbot_G1_stack_bowl_1_2017
  Galbot_G1_stack_bowl_1_2019 -> .../2019/proc/CoRobot/Galbot_G1_stack_bowl_1_2019
```

每个 dataset 目录下补：

```text
meta/modality.json
```

### modality.json

第一版建议映射完整 38 维 action/state：

```json
{
  "state": {
    "full": {
      "start": 0,
      "end": 38,
      "original_key": "observation.state"
    }
  },
  "action": {
    "full": {
      "start": 0,
      "end": 38,
      "original_key": "action"
    }
  },
  "video": {
    "arm_left": {
      "original_key": "observation.images.image_arm_left"
    },
    "arm_right": {
      "original_key": "observation.images.image_arm_right"
    },
    "head_left": {
      "original_key": "observation.images.image_head_left"
    }
  },
  "annotation": {
    "human.action.task_description": {
      "original_key": "task_index"
    }
  }
}
```

需要确认 `LeRobotModalityMetadata` 是否接受 state/action 条目上的 `original_key`。如果不接受，就按现有 schema 改成 nested key 的标准形式。

### data_config.py

新增 `GalbotG1DataConfig`：

```python
class GalbotG1DataConfig:
    video_keys = [
        "video.arm_left",
        "video.arm_right",
        "video.head_left",
    ]
    state_keys = ["state.full"]
    action_keys = ["action.full"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]
```

transform 第一版：

```text
action.full: min_max
state.full: min_max
```

如果 state 不参与模型输入，可以先不启用 state transform，保持 `include_state: false`。但为了统计和未来推理，建议 registry 保留 state。

mixture：

```python
DATASET_NAMED_MIXTURES = {
    "galbot_stack_bowl": [
        ("Galbot_G1_stack_bowl_1_2015", 1.0, "galbot_g1"),
        ("Galbot_G1_stack_bowl_1_2016", 1.0, "galbot_g1"),
        ("Galbot_G1_stack_bowl_1_2017", 1.0, "galbot_g1"),
        ("Galbot_G1_stack_bowl_1_2019", 1.0, "galbot_g1"),
    ],
}
```

embodiment：

```python
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "galbot_g1": EmbodimentTag.NEW_EMBODIMENT
}
```

如果 `EmbodimentTag.NEW_EMBODIMENT` 不存在，沿用当前 dataloader fallback 行为，不在 map 中注册该 key。

## 训练配置修改

基于当前：

```text
starVLA/examples/LIBERO/train_files/jepadit_train_libero.yaml
```

新建：

```text
starVLA/examples/Galbot/train_files/jepadit_train_galbot.yaml
```

核心改动：

```yaml
run_id: jepadit_galbot_g1_stack_bowl_3view

framework:
  name: JEPADiT
  jepa:
    model_name: vjepa2_1_vit_gigantic_384
    checkpoint_path: /share/project/lvjing/models/vjepa2_1/vjepa2_1_vitG_384.pt
    checkpoint_key: encoder
    img_size: 384
    freeze_encoder: true
    image_views: 3
    multi_view_fusion: concat_tokens
  action_model:
    action_model_type: DiT-B
    action_dim: 38
    state_dim: 38
    future_action_window_size: 7
    action_horizon: 8
    past_action_window_size: 0
    repeated_diffusion_steps: 8
    hidden_size: 1024
    diffusion_model_cfg:
      cross_attention_dim: 1664
      num_layers: 16
      output_dim: 1024

datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: playground/Datasets/GALBOT_G1_DIEWAN
    data_mix: galbot_stack_bowl
    action_type: delta_qpos
    sequential_step_sampling: false
    include_state: false
    default_image_resolution: [3, 384, 384]
    per_device_batch_size: 4
    load_all_data_for_training: true
    obs: ["image_0", "image_1", "image_2"]
    video_backend: torchvision_av
```

注意：

- `action_dim/state_dim` 必须从 `7` 改成 `38`。
- `cross_attention_dim` 仍是 `1664`。
- `per_device_batch_size` 先用 `4` 或 `8`，因为三视角会显著增加 JEPA forward 成本。
- `include_state: false` 可保持第一版简单；如果后续发现 action head 需要 proprio，再打开 state。

## LeRobot v2.1 兼容点

当前 `LeRobotSingleDataset` 默认使用：

```python
self._lerobot_version = self.data_cfg.get("lerobot_version", "v2.0")
```

所以即使 `info.json` 写着 `v2.1`，只要 config 不指定 `lerobot_version`，dataloader 会按 `v2.0` 文件布局读取。

第一版建议：

- 不改原始 `info.json`。
- 在 `data_cfg` 中不传 `lerobot_version`，让它按默认 `v2.0` 走。
- 如果后续完整性检查依赖 `codebase_version`，再在 Galbot config 显式设置：

```python
data_cfg = {
    "lerobot_version": "v2.0"
}
```

## 验证计划

### 1. 数据链接检查

确认四个软链接存在：

```bash
find starVLA/playground/Datasets/GALBOT_G1_DIEWAN -maxdepth 2 -type f | head
```

确认每个目录有：

```text
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/modality.json
data/chunk-000/*.parquet
videos/chunk-000/*/*.mp4
```

### 2. 单样本 dataloader smoke

目标：

```text
sample["image"] 长度为 3
sample["action"].shape == [8, 38]
如果 include_state=true: sample["state"].shape 最后一维为 38
language 能正常解析为 Galbot_G1_stack_bowl_1
```

### 3. JEPADiT forward smoke

目标：

```text
vl_embs 单视角: [B, N, 1664]
vl_embs 三视角: [B, 3N, 1664]
action_target: [B, 8, 38]
action_loss 正常返回 finite scalar
```

### 4. 短步数训练

先跑：

```text
max_train_steps: 2
eval_interval: 1
save_interval: 1
per_device_batch_size: 1 或 2
```

确认：

- dataloader 多进程不卡。
- 视频解码正常。
- `JEPADiT` 三视角前向正常。
- checkpoint 能保存。

### 5. 小规模训练

建议第一轮：

```text
max_train_steps: 1000
per_device_batch_size: 4
save_interval: 500
eval_interval: 100
```

观察：

- `action_loss` 是否下降。
- 是否出现 38 维 action normalization 异常。
- GPU 显存和吞吐是否可接受。

## 风险点

1. `modality.json` schema 可能需要按 StarVLA 的 `LeRobotModalityMetadata` 精确调整。

2. 三视角 token 数是单视角三倍，cross-attention 计算量增加。第一版可接受，后续可能需要 token pooling。

3. action 是 38 维全身控制，包含双臂、夹爪、腿、头、底盘和 odom。是否全部作为训练 action 需要确认。第一版先全量训练，后续可裁剪到实际控制维度。

4. `action_type: delta_qpos` 对 38 维全量 action 未必语义完全合适。当前 action 和 state 都是 follower/leader joint/state，初版可沿用现有 delta 逻辑，但需要检查 gripper 和 chassis/odom 维度是否应保持 absolute。

5. 任务文本只有一个，语言条件不会提供多任务区分。小数据集训练主要验证动作模仿能力。

## 推荐实施顺序

1. 新建 Galbot example 目录和 registry。
2. 建软链接数据 root。
3. 补四个 dataset 的 `meta/modality.json`。
4. 改 `JEPADiT` 支持多视角 token concat。
5. 新建 `jepadit_train_galbot.yaml` 和启动脚本。
6. 跑 dataloader smoke。
7. 跑 forward smoke。
8. 跑 2-step 分布式训练 smoke。
9. 跑 1k-step 小训练。
