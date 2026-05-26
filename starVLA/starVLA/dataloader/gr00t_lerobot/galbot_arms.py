"""OpenPI-compatible Galbot arms preprocessing helpers."""

import numpy as np

GALBOT_ARMS_RAW_TO_OPENPI = np.array([8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7])
GALBOT_ARMS_JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14])
GALBOT_ARMS_GRIPPER_INDICES = np.array([7, 15])
GALBOT_GRIPPER_SCALE = 1.0


def galbot_reorder_arms_to_openpi(values: np.ndarray) -> np.ndarray:
    """Convert raw flat Galbot arms to OpenPI order: left arm/gripper, right arm/gripper."""

    return values[..., GALBOT_ARMS_RAW_TO_OPENPI]


def galbot_grippers_to_meters(values: np.ndarray) -> np.ndarray:
    """Apply the configured Galbot gripper scale."""

    values = values.copy()
    values[..., GALBOT_ARMS_GRIPPER_INDICES] = values[..., GALBOT_ARMS_GRIPPER_INDICES] / GALBOT_GRIPPER_SCALE
    return values


def galbot_prepare_arms_state(values: np.ndarray) -> np.ndarray:
    """Prepare current state in the same layout and gripper scaling as training stats."""

    return galbot_grippers_to_meters(galbot_reorder_arms_to_openpi(values))


def galbot_prepare_arms_action(future_values: np.ndarray, current_state: np.ndarray) -> np.ndarray:
    """Prepare action chunks: joint deltas relative to current state, grippers absolute."""

    state = galbot_prepare_arms_state(current_state)
    action = galbot_grippers_to_meters(galbot_reorder_arms_to_openpi(future_values))
    action = action.copy()
    action[..., GALBOT_ARMS_JOINT_INDICES] = action[..., GALBOT_ARMS_JOINT_INDICES] - state[..., GALBOT_ARMS_JOINT_INDICES]
    return action
