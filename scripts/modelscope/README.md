# ModelScope Checkpoint Scripts

These scripts upload and download this checkpoint folder:

```text
/share/project/lvjing/vjepa2/starVLA/playground/Checkpoints/jepadit_galbot_g1_stack_bowl_3view_arms_delta_20k_bs16_chunk30
```

The ModelScope token is embedded in the scripts as requested. Both scripts activate
the conda `base` environment when `conda` is available, then install `modelscope`
if the package is missing.

Upload:

```bash
scripts/modelscope/upload_g1_stack_bowl_checkpoint_to_modelscope.sh <modelscope_repo_id>
```

Example:

```bash
scripts/modelscope/upload_g1_stack_bowl_checkpoint_to_modelscope.sh your_name/jepadit_galbot_g1_stack_bowl
```

Download:

```bash
scripts/modelscope/download_g1_stack_bowl_checkpoint_from_modelscope.sh <modelscope_repo_id>
```

By default, download writes under:

```text
/share/project/lvjing/vjepa2/starVLA/playground/Checkpoints
```

You can override the local directory:

```bash
scripts/modelscope/download_g1_stack_bowl_checkpoint_from_modelscope.sh your_name/jepadit_galbot_g1_stack_bowl /tmp/Checkpoints
```
