from malecns_sim.simulation.body_io import inventory_model
from malecns_sim.simulation.fly import build_fly


def test_body_io_is_inventory_without_neural_mapping():
    report = inventory_model(build_fly().model)
    assert report["counts"]["actuators"] == 42
    assert report["counts"]["actuated_joint_positions"] == 42
    assert report["counts"]["actuator_force_channels"] == 42
    assert report["counts"]["configured_sensors"] == 6
    assert report["mapping"]["male_cns_to_actuator"] is None
    assert report["mapping"]["body_sensor_to_male_cns"] is None
