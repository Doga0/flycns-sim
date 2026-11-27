"""Deterministic external joint torque; independent of neural and body state."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class JointTorque:
    kind: str
    joint: str
    amplitude_nm: float
    start_ms: float
    ramp_duration_ms: float = 0.0
    hold_duration_ms: float = 0.0
    release_duration_ms: float = 0.0

    @classmethod
    def from_config(cls, config, model):
        if config is None:
            return None
        if config.get("type") not in {"step_torque", "ramp_hold_release"}:
            raise ValueError("Unknown perturbation type")
        fields = {
            "kind": config["type"],
            "joint": config["joint"],
            "amplitude_nm": config["amplitude_nm"],
            "start_ms": config["start_ms"],
            "ramp_duration_ms": config.get("ramp_duration_ms", 0.0),
            "hold_duration_ms": config.get("hold_duration_ms", 0.0),
            "release_duration_ms": config.get("release_duration_ms", 0.0),
        }
        result = cls(**fields)
        numbers = (
            result.amplitude_nm,
            result.start_ms,
            result.ramp_duration_ms,
            result.hold_duration_ms,
            result.release_duration_ms,
        )
        if not np.isfinite(numbers).all() or result.start_ms < 0 or result.amplitude_nm == 0:
            raise ValueError("Invalid external torque parameters")
        if result.kind == "step_torque":
            if (
                result.hold_duration_ms <= 0
                or result.ramp_duration_ms
                or result.release_duration_ms
            ):
                raise ValueError("Step torque needs positive hold only")
        elif min(result.ramp_duration_ms, result.hold_duration_ms, result.release_duration_ms) <= 0:
            raise ValueError("Ramp torque needs positive ramp, hold and release")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, result.joint)
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError("External torque requires a named hinge joint")
        return result

    @property
    def stop_ms(self):
        return (
            self.start_ms + self.ramp_duration_ms + self.hold_duration_ms + self.release_duration_ms
        )

    def torque_at(self, t_ms):
        """Half-open schedule: the same time has the same torque in every condition."""
        if not np.isfinite(t_ms) or t_ms < self.start_ms or t_ms >= self.stop_ms:
            return 0.0
        elapsed = t_ms - self.start_ms
        if elapsed < self.ramp_duration_ms:
            return self.amplitude_nm * elapsed / self.ramp_duration_ms
        elapsed -= self.ramp_duration_ms
        if elapsed < self.hold_duration_ms:
            return self.amplitude_nm
        elapsed -= self.hold_duration_ms
        return self.amplitude_nm * (1 - elapsed / self.release_duration_ms)

    def dof(self, model):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.joint)
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError("External torque joint differs from compiled body")
        return int(model.jnt_dofadr[joint_id])
