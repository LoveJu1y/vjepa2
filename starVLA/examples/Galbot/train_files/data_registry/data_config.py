"""Galbot G1 benchmark data config, embodiment tag, and mixtures."""

from typing import Any

import numpy as np
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.galbot_arms import (
    galbot_prepare_arms_action,
    galbot_prepare_arms_state,
)
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform, ModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class GalbotArmsDeltaTransform(ModalityTransform):
    """Match OpenPI Galbot arms-only action layout and semantics."""

    apply_to: list[str] = ["state.arms", "action.arms_future"]

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if "state.arms" not in data or "action.arms_future" not in data:
            return data

        state = np.asarray(data["state.arms"])
        action = np.asarray(data["action.arms_future"])

        anchor = state[0] if state.ndim == 2 else state

        data["state.arms"] = galbot_prepare_arms_state(state)
        data["action.arms_future"] = galbot_prepare_arms_action(action, anchor)
        return data


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
    action_indices = list(range(30))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={"state.full": "min_max"},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={"action.full": "min_max"},
            ),
        ])


class GalbotG1ArmsDeltaDataConfig:
    video_keys = [
        "video.arm_left",
        "video.arm_right",
        "video.head_left",
    ]
    state_keys = ["state.arms"]
    action_keys = ["action.arms_future"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(1, 31))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            GalbotArmsDeltaTransform(),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={"state.arms": "q99"},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={"action.arms_future": "q99"},
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "galbot_g1": GalbotG1DataConfig(),
    "galbot_g1_arms_delta": GalbotG1ArmsDeltaDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "galbot_g1": EmbodimentTag.NEW_EMBODIMENT,
    "galbot_g1_arms_delta": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    "galbot_stack_bowl": [
        ("Galbot_G1_stack_bowl_1_2015", 1.0, "galbot_g1"),
        ("Galbot_G1_stack_bowl_1_2016", 1.0, "galbot_g1"),
        ("Galbot_G1_stack_bowl_1_2017", 1.0, "galbot_g1"),
        ("Galbot_G1_stack_bowl_1_2019", 1.0, "galbot_g1"),
    ],
    "galbot_stack_bowl_arms_delta": [
        ("Galbot_G1_stack_bowl_1_2015", 1.0, "galbot_g1_arms_delta"),
        ("Galbot_G1_stack_bowl_1_2016", 1.0, "galbot_g1_arms_delta"),
        ("Galbot_G1_stack_bowl_1_2017", 1.0, "galbot_g1_arms_delta"),
        ("Galbot_G1_stack_bowl_1_2019", 1.0, "galbot_g1_arms_delta"),
    ],
    "galbot_book_0430_arms_delta": [
        (
            "Galbot_G1_Push_the_book_to_the_edge_of_the_table_then_grab_it_and_place_it_on_the_bookshelf_2_1990",
            1.0,
            "galbot_g1_arms_delta",
        ),
        (
            "Galbot_G1_Push_the_book_to_the_edge_of_the_table_then_grab_it_and_place_it_on_the_bookshelf_2_1991",
            1.0,
            "galbot_g1_arms_delta",
        ),
        (
            "Galbot_G1_Push_the_book_to_the_edge_of_the_table_then_grab_it_and_place_it_on_the_bookshelf_2_1992",
            1.0,
            "galbot_g1_arms_delta",
        ),
        (
            "Galbot_G1_Push_the_book_to_the_edge_of_the_table_then_grab_it_and_place_it_on_the_bookshelf_2_1993",
            1.0,
            "galbot_g1_arms_delta",
        ),
    ],
    "galbot_stamp_0503_arms_delta": [
        (
            "Galbot_G1_Stamp_the_document_on_the_table_ then_return_the_stamp_to_its_original_position_3_2027",
            1.0,
            "galbot_g1_arms_delta",
        ),
        (
            "Galbot_G1_Stamp_the_document_on_the_table_ then_return_the_stamp_to_its_original_position_3_2028",
            1.0,
            "galbot_g1_arms_delta",
        ),
        (
            "Galbot_G1_Stamp_the_document_on_the_table_ then_return_the_stamp_to_its_original_position_3_2029",
            1.0,
            "galbot_g1_arms_delta",
        ),
        (
            "Galbot_G1_Stamp_the_document_on_the_table_ then_return_the_stamp_to_its_original_position_3_2031",
            1.0,
            "galbot_g1_arms_delta",
        ),
    ],
    "galbot_chouzhi_0506_arms_delta": [
        (
            "Galbot_G1_Hold_the_tissue_box_with_one_hand_and_pull_out_a_tissue_with_the_other_3_2053",
            1.0,
            "galbot_g1_arms_delta",
        ),
        (
            "Galbot_G1_Hold_the_tissue_box_with_one_hand_and_pull_out_a_tissue_with_the_other_3_2054",
            1.0,
            "galbot_g1_arms_delta",
        ),
        (
            "Galbot_G1_Hold_the_tissue_box_with_one_hand_and_pull_out_a_tissue_with_the_other_3_2055",
            1.0,
            "galbot_g1_arms_delta",
        ),
        (
            "Galbot_G1_Hold_the_tissue_box_with_one_hand_and_pull_out_a_tissue_with_the_other_3_2056",
            1.0,
            "galbot_g1_arms_delta",
        ),
    ],
    "galbot_sugar_0507_arms_delta": [
        (
            "Galbot_G1_Scoop_sugar_from_the_sugar_jar_with_a_spoon_pour_it_into_a_mug_with_water_and_stir_2_2063",
            1.0,
            "galbot_g1_arms_delta",
        ),
        (
            "Galbot_G1_Scoop_sugar_from_the_sugar_jar_with_a_spoon_pour_it_into_a_mug_with_water_and_stir_2_2064",
            1.0,
            "galbot_g1_arms_delta",
        ),
        (
            "Galbot_G1_Scoop_sugar_from_the_sugar_jar_with_a_spoon_pour_it_into_a_mug_with_water_and_stir_2_2065",
            1.0,
            "galbot_g1_arms_delta",
        ),
        (
            "Galbot_G1_Scoop_sugar_from_the_sugar_jar_with_a_spoon_pour_it_into_a_mug_with_water_and_stir_2_2066",
            1.0,
            "galbot_g1_arms_delta",
        ),
    ],
}
