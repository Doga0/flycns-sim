"""Narrow MuJoCo/FlyGym body-state boundary used by sensory encoders."""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class BodyState:
    t_ms: float
    joint_name: str
    joint_position_rad: float
    joint_velocity_rad_s: float
    actuator_target_rad: float | None

    def to_dict(self):
        return asdict(self)


class JointStateReader:
    """Read one named hinge in one place, rather than exposing qpos addresses."""

    def __init__(self, model: mujoco.MjModel, joint_name: str, actuator_name: str | None = None):
        self.model = model
        self.joint_name = joint_name
        self.joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if self.joint_id < 0 or model.jnt_type[self.joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Unknown/non-hinge joint: {joint_name}")
        self.qpos_address = int(model.jnt_qposadr[self.joint_id])
        self.qvel_address = int(model.jnt_dofadr[self.joint_id])
        self.actuator_index = None
        if actuator_name is not None:
            self.actuator_index = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if (
                self.actuator_index < 0
                or model.actuator_trnid[self.actuator_index, 0] != self.joint_id
            ):
                raise ValueError(f"Actuator does not directly target {joint_name}: {actuator_name}")

    def read(self, data: mujoco.MjData) -> BodyState:
        values = (data.time * 1000, data.qpos[self.qpos_address], data.qvel[self.qvel_address])
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite body sensory state")
        target = None if self.actuator_index is None else float(data.ctrl[self.actuator_index])
        if target is not None and not np.isfinite(target):
            raise FloatingPointError("Nonfinite body actuator target")
        return BodyState(
            float(values[0]), self.joint_name, float(values[1]), float(values[2]), target
        )
