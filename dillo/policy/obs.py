"""Observation key conventions used by the ACT policy."""

OBS_MODALITY = {
    "rgb": ["agentview_rgb", "eye_in_hand_rgb"],
    "depth": [],
    "low_dim": ["gripper_states", "joint_states"],
}

OBS_KEY_MAPPING = {
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
    "gripper_states": "robot0_gripper_qpos",
    "joint_states": "robot0_joint_pos",
}

OBS_KEYS = [key for keys in OBS_MODALITY.values() for key in keys]
