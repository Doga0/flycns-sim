"""Compose the FlyGym basic example into a native MuJoCo model."""

from dataclasses import dataclass

import mujoco
from flygym.anatomy import (
    ActuatedDOFPreset,
    AxisOrder,
    ContactBodiesPreset,
    JointPreset,
    Skeleton,
)
from flygym.compose import FlatGroundWorld, KinematicPosePreset, NeuroMechFly
from flygym.utils.math import Rotation3D


@dataclass
class FlySimulation:
    model: mujoco.MjModel
    data: mujoco.MjData
    camera: str


def build_fly() -> FlySimulation:
    fly = NeuroMechFly(name="fly")
    skeleton = Skeleton(
        joint_preset=JointPreset.ALL_BIOLOGICAL,
        axis_order=AxisOrder.ROLL_PITCH_YAW,
    )
    neutral_pose = KinematicPosePreset.NEUTRAL
    fly.add_joints(skeleton, neutral_pose=neutral_pose)
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(actuated_dofs, actuator_type="position", neutral_input=neutral_pose, kp=50)
    fly.colorize()
    camera = fly.add_tracking_camera()
    world = FlatGroundWorld()
    world.add_fly(
        fly,
        [0, 0, 0.7],
        Rotation3D(format="quat", values=[1, 0, 0, 0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
    )
    model, data = world.compile()
    neutral_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "neutral")
    if neutral_id < 0:
        raise RuntimeError("FlyGym model has no neutral keyframe")
    mujoco.mj_resetDataKeyframe(model, data, neutral_id)
    mujoco.mj_forward(model, data)
    return FlySimulation(model, data, camera.name)
