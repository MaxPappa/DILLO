"""Prompt used to annotate DILLO action chunks."""

CHUNK_PROMPT_VIDEO_AND_OBS = """
In the following you are presented a short video clip consisting of frames from
the start and end of an action chunk executed by a Panda robot arm in a tabletop
environment.  You are also given the end-effector position and gripper state
before and after the chunk.

The task is: {task_instruction}.

Observation format:
- End-Effector Position: [x, y, z] coordinates of the gripper.
- Gripper Openness: scalar openness of the gripper.
- EEF Movement Direction and Gripper State Change summarize the transition.

Describe how the robot's behavior changed and how it relates to the object
interaction and task goal. Focus on whether the gripper moved toward or away
from the relevant object, whether it appears to reach, grasp, push, lift, place,
release, or avoid interaction, whether objects visibly moved, and how the
gripper openness changed.

Use natural directions instead of axis names:
- left/right for x movement
- backward/forward for y movement
- down/up for z movement

If the arm moves away from the object without completing the task, say that the
behavior may be incorrect or unhelpful. Do not claim a grasp, lift, or placement
unless it is visible in the frames.

The observations for this chunk are:
{frame_observations}

Return one concise paragraph describing the transition.
"""
