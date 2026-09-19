"""Inventory configured FlyGym/MuJoCo channels without mapping them to MaleCNS."""

import argparse
from pathlib import Path

import mujoco

from malecns_sim.cns.graph.store import write_json
from malecns_sim.simulation.fly import build_fly


def _name(model, kind, index):
    return mujoco.mj_id2name(model, kind, int(index))


def inventory_model(model: mujoco.MjModel) -> dict:
    actuators = []
    actuator_joints = set()
    for index in range(model.nu):
        target = int(model.actuator_trnid[index, 0])
        joint = _name(model, mujoco.mjtObj.mjOBJ_JOINT, target)
        actuator_joints.add(target)
        actuators.append(
            {
                "actuator_index": index,
                "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index),
                "actuator_type": "position",
                "joint": joint,
                "joint_id": target,
                "control_range": model.actuator_ctrlrange[index].tolist(),
                "force_range": model.actuator_forcerange[index].tolist(),
            }
        )
    joint_state = [
        {
            "joint_id": index,
            "joint": _name(model, mujoco.mjtObj.mjOBJ_JOINT, index),
            "qpos_address": int(model.jnt_qposadr[index]),
            "qvel_address": int(model.jnt_dofadr[index]),
            "position_source": "MjData.qpos",
            "velocity_source": "MjData.qvel",
        }
        for index in sorted(actuator_joints)
    ]
    sensors = []
    for index in range(model.nsensor):
        obj_type = mujoco.mjtObj(int(model.sensor_objtype[index]))
        obj_id = int(model.sensor_objid[index])
        sensors.append(
            {
                "sensor_index": index,
                "name": _name(model, mujoco.mjtObj.mjOBJ_SENSOR, index),
                "sensor_type": mujoco.mjtSensor(int(model.sensor_type[index])).name,
                "object_type": obj_type.name,
                "object_name": _name(model, obj_type, obj_id) if obj_id >= 0 else None,
                "data_address": int(model.sensor_adr[index]),
                "dimension": int(model.sensor_dim[index]),
                "data_source": "MjData.sensordata",
            }
        )
    sites = [
        {"site_id": index, "name": _name(model, mujoco.mjtObj.mjOBJ_SITE, index)}
        for index in range(model.nsite)
    ]
    return {
        "schema": 1,
        "body": "FlyGym NeuroMechFly",
        "actuators": actuators,
        "readable_state_channels": {
            "joint_position": joint_state,
            "joint_velocity": joint_state,
            "actuator_force": [
                {
                    "actuator_index": row["actuator_index"],
                    "actuator": row["name"],
                    "source": "MjData.actuator_force",
                }
                for row in actuators
            ],
            "configured_sensors": sensors,
            "sites": sites,
            "runtime_contacts": {
                "source": "MjData.contact[0:MjData.ncon]",
                "description": "Runtime collision contacts; not a fixed scalar sensor channel.",
            },
        },
        "counts": {
            "actuators": len(actuators),
            "actuated_joint_positions": len(joint_state),
            "actuated_joint_velocities": len(joint_state),
            "actuator_force_channels": len(actuators),
            "configured_sensors": len(sensors),
            "sites": len(sites),
        },
        "mapping": {
            "male_cns_to_actuator": None,
            "body_sensor_to_male_cns": None,
            "note": "Inventory only. No neural/body encoding or decoding is defined in v0.4.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/body-io.json"))
    args = parser.parse_args()
    inventory = inventory_model(build_fly().model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, inventory)
    print("ACTUATORS\n" + "-" * 40)
    for actuator in inventory["actuators"]:
        print(f"{actuator['actuator_index']:>2}  {actuator['name']}")
    print("\nSENSORS\n" + "-" * 40)
    print(f"Joint positions: {inventory['counts']['actuated_joint_positions']}")
    print(f"Joint velocities: {inventory['counts']['actuated_joint_velocities']}")
    print(f"Actuator forces: {inventory['counts']['actuator_force_channels']}")
    for sensor in inventory["readable_state_channels"]["configured_sensors"]:
        print(f"{sensor['sensor_index']:>2}  {sensor['name']} ({sensor['sensor_type']})")
    print(f"Sites currently configured: {inventory['counts']['sites']}")
    print(f"\nFull body I/O inventory: {args.output}")


if __name__ == "__main__":
    main()
