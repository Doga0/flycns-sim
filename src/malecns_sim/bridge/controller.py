"""Apply validated position targets to explicitly mapped MuJoCo actuators."""

import numpy as np

from malecns_sim.bridge.decoder import MotorCommand


class BodyController:
    def __init__(self, mapping, model):
        self.channels = {c["body"]["actuator_name"]: c for c in mapping.channels}
        self.previous = {name: c["body"]["neutral_position"] for name, c in self.channels.items()}
        self.last_time = {name: 0.0 for name in self.channels}
        self.dt = mapping.control_dt_ms
        self.model = model
        for name, c in self.channels.items():
            i = c["body"]["actuator_index"]
            if model.actuator(i).name != name:
                raise ValueError("Controller model differs from mapping")
            # Explicit engineering bounds, also enforced inside MuJoCo. The source body
            # has no enabled joint/control range; do not label these anatomical limits.
            model.actuator_ctrlrange[i] = [c["minimum"], c["maximum"]]
            model.actuator_ctrllimited[i] = True

    def apply(self, command: MotorCommand, mj_data):
        name = command.actuator_name
        if name not in self.channels:
            raise ValueError(f"Unknown/unmapped actuator: {name}")
        numeric = [v for k, v in command.to_dict().items() if k != "actuator_name"]
        if not np.isfinite(numeric).all() or not np.isfinite(mj_data.ctrl).all():
            raise ValueError("Nonfinite body command or controls")
        c = self.channels[name]
        if not np.isclose(command.t_ms, self.last_time[name] + self.dt, rtol=0, atol=1e-8):
            raise ValueError("Out-of-order command or missing control step")
        if not np.isclose(mj_data.time * 1000, command.t_ms, rtol=0, atol=1e-7):
            raise ValueError("Command must be applied on its physics clock boundary")
        if (
            not c["minimum"] <= command.target <= c["maximum"]
            or abs(command.target - self.previous[name]) > c["decoder"]["max_delta_rad"] + 1e-12
        ):
            raise ValueError("Unsafe command range or slew rate")
        mj_data.ctrl[c["body"]["actuator_index"]] = command.target
        self.previous[name] = command.target
        self.last_time[name] = command.t_ms
