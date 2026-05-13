import dataclasses
import json
import logging
import math
import os
import pathlib

import imageio
import numpy as np
import torch
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _patch_torch_load_for_legacy_init_states() -> None:
    if getattr(torch.load, "_libero_plus_patched", False):
        return

    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    patched_torch_load._libero_plus_patched = True
    torch.load = patched_torch_load


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224, 224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    log_path: str = "experiments/libero/logs"
    max_videos_to_save: int = -1  # Save at most N successful rollout videos; negative means save all successful

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    post_process_action: bool = True

    job_name: str = "test"
    start_idx: int = 0
    end_idx: int = -1
    task_orders: str = ""  # Comma-separated 0-based task ids to evaluate explicitly


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # LIBERO-plus init-state files are legacy torch pickles; torch>=2.6 now defaults
    # to weights_only=True and rejects them unless we explicitly opt out.
    _patch_torch_load_for_legacy_init_states()

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.log_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    if args.task_orders:
        task_orders = []
        for raw in args.task_orders.split(","):
            raw = raw.strip()
            if not raw:
                continue
            task_id = int(raw)
            if task_id < 0 or task_id >= num_tasks_in_suite:
                raise ValueError(f"Invalid task id {task_id} for suite with {num_tasks_in_suite} tasks")
            task_orders.append(task_id)
        if not task_orders:
            raise ValueError("task_orders was provided but no valid task ids were parsed")
        start_idx = min(task_orders)
        end_idx = max(task_orders) + 1
        logging.info(f"Evaluating explicit task ids {task_orders} out of {num_tasks_in_suite}")
    else:
        start_idx = max(args.start_idx, 0)
        end_idx = num_tasks_in_suite if args.end_idx < 0 else min(args.end_idx, num_tasks_in_suite)
        if start_idx >= end_idx:
            raise ValueError(
                f"Invalid task range [{args.start_idx}, {args.end_idx}) for suite with {num_tasks_in_suite} tasks"
            )
        task_orders = list(range(start_idx, end_idx))
        logging.info(f"Evaluating tasks in range [{start_idx}, {end_idx}) out of {num_tasks_in_suite}")

    client_model = ModelClient(
        policy_ckpt_path=args.pretrained_path,  # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )

    disturb_res = {}
    LIBERO_HOME = os.environ.get("LIBERO_HOME", "path_to_LIBERO-plus_home")
    with open(os.path.join(LIBERO_HOME, "libero/libero/benchmark/task_classification.json")) as f:
        TASK_MAPPING = json.load(f)[args.task_suite_name]
    ID2CATEGORY = {}
    for item in TASK_MAPPING:
        category = item["category"]
        item_name = item["name"]
        ID2CATEGORY[item["id"]] = (category, item_name)
        if category not in disturb_res:
            disturb_res[category] = {
                "suite_total_count": 0,
                "evaluated_count": 0,
                "success_count": 0,
                "evaluated_task_ids": [],
                "evaluated_task_names": [],
            }
        disturb_res[category]["suite_total_count"] += 1

    # Start evaluation
    saved_videos = 0
    total_episodes, total_successes = 0, 0
    per_task_success_rate = {}
    evaluated_task_ids = []
    evaluated_task_names = []
    for task_id in tqdm.tqdm(task_orders):

        task_key = task_id + 1
        category, task_name = ID2CATEGORY[task_key]
        disturb_res[category]["evaluated_count"] += 1
        disturb_res[category]["evaluated_task_ids"].append(task_key)
        disturb_res[category]["evaluated_task_names"].append(task_name)
        evaluated_task_ids.append(task_key)
        evaluated_task_names.append(task_name)

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):

            logging.info(f"\nTask: {task_description}")

            # Reset environment
            client_model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            full_actions = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0

            # full_actions = np.load("./debug/action.npy")

            while t < max_steps + args.num_steps_wait:

                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = {  #
                    "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # align key with model API --> two images provided here --> check training
                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "lang": observation["instruction"][0],
                }

                response = client_model.step(example=example_dict, step=step)

                # #
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(
                        f"Unexpected action sizes: "
                        f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                        f"Falling back to LIBERO_DUMMY_ACTION."
                    )
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)

                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    disturb_res[category]["success_count"] += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            should_save_video = bool(done) and (args.max_videos_to_save < 0 or saved_videos < args.max_videos_to_save)
            if should_save_video:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"rollout_{ID2CATEGORY[task_id + 1][1]}_episode{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=25,
                )
                saved_videos += 1

            full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_episode{episode_idx}_{suffix}.npy", full_actions)

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        current_task_success_rate = float(task_successes) / float(task_episodes)
        per_task_success_rate[task_name] = current_task_success_rate
        logging.info(f"Current task success rate: {current_task_success_rate}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    suite_category_success_rate = {}
    evaluated_category_success_rate = {}
    for category, stats in disturb_res.items():
        suite_total_count = stats["suite_total_count"]
        evaluated_count = stats["evaluated_count"]
        suite_rate = float(stats["success_count"]) / float(suite_total_count) if suite_total_count > 0 else 0.0
        evaluated_rate = float(stats["success_count"]) / float(evaluated_count) if evaluated_count > 0 else 0.0
        stats["suite_success_rate"] = suite_rate
        stats["evaluated_success_rate"] = evaluated_rate
        suite_category_success_rate[category] = suite_rate
        evaluated_category_success_rate[category] = evaluated_rate

    category_metrics_path = pathlib.Path(args.log_path) / f"{args.task_suite_name}.json"
    with open(category_metrics_path, "w", encoding="utf-8") as f:
        json.dump(disturb_res, f, indent=2, ensure_ascii=False)

    total_success_rate = float(total_successes) / float(total_episodes)
    metrics = {
        "task_suite_name": args.task_suite_name,
        "num_trials_per_task": args.num_trials_per_task,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "task_orders": task_orders,
        "num_tasks_in_suite": num_tasks_in_suite,
        "evaluated_tasks": len(task_orders),
        "evaluated_task_ids": evaluated_task_ids,
        "evaluated_task_names": evaluated_task_names,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_success_rate,
        "per_task_success_rate": per_task_success_rate,
        "per_category_suite_success_rate": suite_category_success_rate,
        "per_category_evaluated_success_rate": evaluated_category_success_rate,
        "per_category_evaluated_count": {
            category: stats["evaluated_count"] for category, stats in disturb_res.items()
        },
        "per_category_success_count": {
            category: stats["success_count"] for category, stats in disturb_res.items()
        },
    }
    metrics_path = pathlib.Path(args.video_out_path) / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logging.info(f"Category metrics saved to {category_metrics_path}")
    logging.info(f"Metrics saved to {metrics_path}")
    logging.info(f"Total success rate: {total_success_rate}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
