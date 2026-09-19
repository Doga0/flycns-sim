"""Versioned exact-value rules for deriving interface fields from released metadata."""

from pathlib import Path

import pandas as pd
import yaml

from malecns_sim.io.roles import NeuronRole


def load_io_policy(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[3] / "configs/io/malecns_io_v1.yaml"
        if not path.is_file():
            path = Path(__file__).parent / "profiles/malecns_io_v1.yaml"
    policy = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or policy.get("schema") != 1:
        raise ValueError("I/O policy must be a schema-1 YAML mapping")
    expected = {role.value for role in NeuronRole}
    if set(policy.get("role_mapping", {})) != expected:
        raise ValueError("I/O policy must define every NeuronRole exactly once")
    values = [value for group in policy["role_mapping"].values() for value in group]
    if len(values) != len(set(values)) or any(not isinstance(value, str) for value in values):
        raise ValueError("Superclass values must be unique strings across role rules")
    return policy


def _assigned(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype("string").str.strip().ne("")


def derive_catalogue(nodes: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Return normalized fields while retaining the raw annotation columns verbatim."""
    required = {
        "node_index",
        "body_id",
        policy["source_field"],
        "class",
        "subclass",
        "type",
        "somaSide",
        "rootSide",
        "entryNerve",
        "exitNerve",
        "somaNeuromere",
    }
    missing = required - set(nodes)
    if missing:
        raise ValueError(f"Graph node metadata lacks I/O fields: {sorted(missing)}")
    result = nodes.copy()
    inverse = {value: role for role, values in policy["role_mapping"].items() for value in values}
    source = result[policy["source_field"]]
    result.insert(2, "role", source.map(inverse).fillna(NeuronRole.OTHER.value))
    result.insert(3, "role_source_field", policy["source_field"])
    result.insert(4, "role_source_value", source)
    result.insert(
        5,
        "role_policy_status",
        source.map(inverse).notna().map({True: "mapped", False: "unmapped"}),
    )

    side = pd.Series(pd.NA, index=result.index, dtype="string")
    side_field = pd.Series(pd.NA, index=result.index, dtype="string")
    for field in policy["side_source_priority"]:
        use = side.isna() & _assigned(result[field])
        side.loc[use] = result.loc[use, field].astype("string")
        side_field.loc[use] = field
    insert_at = 6
    for name, values in (
        ("side", side),
        ("side_source_field", side_field),
        ("side_source_value", side.copy()),
    ):
        result.insert(insert_at, name, values)
        insert_at += 1

    nerve = pd.Series(pd.NA, index=result.index, dtype="string")
    nerve_field = pd.Series(pd.NA, index=result.index, dtype="string")
    for role, field in policy["nerve_source_by_role"].items():
        use = result["role"].eq(role) & _assigned(result[field])
        nerve.loc[use] = result.loc[use, field].astype("string")
        nerve_field.loc[use] = field
    for name, values in (
        ("nerve", nerve),
        ("nerve_source_field", nerve_field),
        ("nerve_source_value", nerve.copy()),
    ):
        result.insert(insert_at, name, values)
        insert_at += 1

    body_field = policy["body_region_source_field"]
    body_region = result[body_field].astype("string")
    result.insert(insert_at, "body_region", body_region)
    result.insert(insert_at + 1, "body_region_source_field", body_field)
    result.insert(insert_at + 2, "body_region_source_value", body_region.copy())
    insert_at += 3

    sensory = result["role"].eq(NeuronRole.SENSORY.value)
    systems = result["class"].map(policy["sensory_system_from_class"]).where(sensory)
    system_field = pd.Series(pd.NA, index=result.index, dtype="string")
    system_field.loc[systems.notna()] = "class"
    result.insert(insert_at, "sensory_system", systems.astype("string"))
    result.insert(insert_at + 1, "sensory_system_source_field", system_field)
    result.insert(insert_at + 2, "sensory_system_source_value", result["class"].where(sensory))
    return result
