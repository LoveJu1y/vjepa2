# Galbot OpenPI 数据适配说明

本文只讲数据，不讲模型。

目标是把 5 个 Galbot 任务统一成 starVLA 可训练的数据格式，并复用 OpenPI 的统计量与动作语义。

## 1. 我们到底在复用什么

我们不是直接拿 OpenPI 的权重训练，而是复用 OpenPI 的两类东西：

1. 数据命名和统计格式
2. `norm_stats.json` 里的归一化边界

OpenPI 侧的关键统计文件都长这样：

```text
.../openpi/assets/pi05_galbot_mmm_*/.../norm_stats.json
```

这个文件的核心字段是：

```json
{
  "norm_stats": {
    "state": { "mean": ..., "std": ..., "q01": ..., "q99": ... },
    "actions": { "mean": ..., "std": ..., "q01": ..., "q99": ... }
  }
}
```

我们在 starVLA 里做的事情，是把 Galbot 数据整理成和 OpenPI 一致的 arms-only 形式，然后直接复用这些 `q01/q99`。

## 2. 5 个任务的数据源

当前 5 个任务的原始数据目录是：

```text
/share/project/zhangningboo/galbot_g1_dataset/0_original/4_diewan_0502
/share/project/zhangningboo/galbot_g1_dataset/0_original/5_book_0430
/share/project/zhangningboo/galbot_g1_dataset/0_original/4_stamp_0503
/share/project/zhangningboo/galbot_g1_dataset/0_original/2_chouzhi_0506
/share/project/zhangningboo/galbot_g1_dataset/0_original/10_sugar_0507
```

每个任务下面又分成 4 个 split，比如：

- `diewan`: `2015 / 2016 / 2017 / 2019`
- `book`: `1990 / 1991 / 1992 / 1993`
- `stamp`: `2027 / 2028 / 2029 / 2031`
- `chouzhi`: `2053 / 2054 / 2055 / 2056`
- `sugar`: `2063 / 2064 / 2065 / 2066`

每个 split 的真实 processed 数据都在：

```text
<split>/proc/CoRobot/<processed_dataset_name>/
```

里面一般有：

- `data/`
- `videos/`
- `meta/info.json`
- `meta/tasks.jsonl`
- `meta/episodes.jsonl`
- `meta/episodes_stats.jsonl`

## 3. starVLA 里怎么包装这些数据

为了让 starVLA 直接读，我们给每个任务建了一个 wrapper 根目录：

```text
starVLA/playground/Datasets/GALBOT_G1_XXXX_YYYY
```

例如：

- `GALBOT_G1_BOOK_0430`
- `GALBOT_G1_STAMP_0503`
- `GALBOT_G1_CHOUZHI_0506`
- `GALBOT_G1_SUGAR_0507`
- `GALBOT_G1_DIEWAN_0502`

wrapper 的做法是：

1. `data/` 软链接到原始 processed 数据的 `data/`
2. `videos/` 软链接到原始 processed 数据的 `videos/`
3. `meta/` 复制原始 `info.json / tasks.jsonl / episodes.jsonl / episodes_stats.jsonl`
4. 新增 `meta/modality.json`

这样 starVLA 不需要知道原始目录长什么样，只看 wrapper 就行。

## 4. `modality.json` 怎么写

我们给 5 个任务统一使用 arms-only 的 modality 映射：

- `video.arm_left`
- `video.arm_right`
- `video.head_left`
- `state.arms`
- `action.arms_future`

对应的典型映射是：

```json
{
  "state": {
    "arms": {
      "start": 0,
      "end": 16,
      "original_key": "observation.state"
    }
  },
  "action": {
    "arms_future": {
      "start": 0,
      "end": 16,
      "original_key": "action"
    }
  }
}
```

这意味着：

- 原始 parquet 里的 `observation.state` / `action` 先被读出来
- 再切出 arms 16 维
- 再交给后面的 Galbot arms 预处理

## 5. 原始 16 维 arms 是怎么变成训练数据的

我们当前的 Galbot arms 预处理在：

- [galbot_arms.py](/share/project/lvjing/vjepa2/starVLA/starVLA/dataloader/gr00t_lerobot/galbot_arms.py:1)

核心步骤只有三步：

### 5.1 重排左右臂顺序

原始 16 维 arms 不是 OpenPI 顺序。

我们会先按 OpenPI 的顺序重排成：

```text
left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)
```

### 5.2 夹爪单位换算

原始 gripper 是毫米量级，我们统一除以 `1000.0`，转成米：

```text
gripper_m = gripper_raw / 1000.0
```

### 5.3 action 变成“关节 delta + 夹爪绝对值”

训练时我们不用纯绝对动作，而是：

- 关节维度：相对当前 state 做 delta
- 夹爪维度：保持绝对值

也就是：

```text
action[joint] = future_state[joint] - current_state[joint]
action[gripper] = future_state[gripper]
```

这一步的代码在：

- `galbot_prepare_arms_action(...)`

## 6. state 是怎么用的

当前这条链路里，`state` 不是丢掉的，而是要用的。

它有两种用途：

1. 训练时输入给 action head
2. action 预处理时做 delta 基准

所以现在的 arms-only Galbot 数据，训练侧会保留：

- `state.arms`
- `action.arms_future`

并且在 QwenGR00T 这类需要 state 的框架里，`include_state` 必须保持开启。

## 7. OpenPI 的 stats 怎么复用

OpenPI 的 stats 不是拿来“看一看”，而是直接作为归一化边界。

我们在 starVLA 里做了两层支持：

1. 直接读取 OpenPI 的 `norm_stats.json`
2. 转成 starVLA 自己的数据统计格式

代码位置：

- [datasets.py](/share/project/lvjing/vjepa2/starVLA/starVLA/dataloader/gr00t_lerobot/datasets.py:175)

对应规则是：

- `OpenPI norm_stats.state` -> `state.arms`
- `OpenPI norm_stats.actions` -> `action.arms_future`
- `q01 / q99` 作为真实归一化边界

如果某个维度是静态常数维（例如 `q01 == q99`），当前实现会按静态维处理，不去强行缩放坏掉它。

## 8. 训练样本长什么样

在 starVLA 训练时，一个 sample 的典型结构是：

```python
{
  "image": [PIL.Image, PIL.Image, PIL.Image],
  "lang": str,
  "language": str,
  "state": np.ndarray,   # 可选，Galbot QwenGR00T 里会用
  "action": np.ndarray,  # shape = [30, 16]
}
```

其中：

- `image` 是三视角
- `state` 是当前时刻 arms state
- `action` 是 30 步的 future action chunk

## 9. 五个任务对应关系

### diewan

- 原始路径：`/share/project/zhangningboo/galbot_g1_dataset/0_original/4_diewan_0502`
- wrapper：`GALBOT_G1_DIEWAN_0502`
- OpenPI stats：`.../pi05_galbot_mmm_diewan/4_diewan_0502/norm_stats.json`

### book

- 原始路径：`/share/project/zhangningboo/galbot_g1_dataset/0_original/5_book_0430`
- wrapper：`GALBOT_G1_BOOK_0430`
- OpenPI stats：`.../pi05_galbot_mmm_book/mmm_1990_1991_1992_1993_val_loss_filtered_processed/norm_stats.json`

### stamp

- 原始路径：`/share/project/zhangningboo/galbot_g1_dataset/0_original/4_stamp_0503`
- wrapper：`GALBOT_G1_STAMP_0503`
- OpenPI stats：`.../pi05_galbot_mmm_stamp_new/mmm_2027_2028_2029_2031_val_loss_filtered_processed/norm_stats.json`

### chouzhi

- 原始路径：`/share/project/zhangningboo/galbot_g1_dataset/0_original/2_chouzhi_0506`
- wrapper：`GALBOT_G1_CHOUZHI_0506`
- OpenPI stats：`.../pi05_galbot_mmm_chouzhi/2_chouzhi_0506_val_loss_filtered_processed/norm_stats.json`

### sugar

- 原始路径：`/share/project/zhangningboo/galbot_g1_dataset/0_original/10_sugar_0507`
- wrapper：`GALBOT_G1_SUGAR_0507`
- OpenPI stats：`.../pi05_galbot_mmm_sugar/10_sugar_0507_val_loss_filtered_processed/norm_stats.json`

## 10. 你最该记住的一句话

我们现在对 Galbot 这 5 个任务的处理逻辑，本质上是：

```text
raw parquet -> wrapper -> arms 16维 -> 重排 / 单位换算 / delta -> OpenPI q01/q99 stats -> 训练
```

如果后面要换模型，数据链路本身不用重做，前提是保持这个 arms-only 语义不变。
