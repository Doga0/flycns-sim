"""Declarative node policies. No Python expressions or implicit scientific filters."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_policy(profile: str | Path = "published") -> dict:
    path = Path(profile)
    if not path.is_file():
        if str(profile) not in {"published", "traced", "all-annotated"}:
            raise FileNotFoundError(f"Unknown graph profile or missing YAML: {profile}")
        root = Path(__file__).resolve().parents[4]
        path = root / "configs" / "graph" / f"{profile}.yaml"
        if not path.is_file():
            path = Path(__file__).parent / "profiles" / f"{profile}.yaml"
    policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    allowed = {
        "schema",
        "id",
        "description",
        "reference_neurons",
        "reference_membership_verified",
        "reference_url",
        "selection",
        "edge_policy",
        "self_edges_preserved",
        "edge_threshold",
    }
    if not isinstance(policy, dict) or set(policy) - allowed:
        raise ValueError("Invalid policy object or unknown policy fields")
    if policy.get("schema") != 1 or not isinstance(policy.get("id"), str):
        raise ValueError("Policy requires schema: 1 and a string id")
    if not policy["id"] or any(
        c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in policy["id"]
    ):
        raise ValueError("Policy id must be a safe lowercase directory name")
    if not isinstance(policy.get("reference_neurons"), int) or policy["reference_neurons"] <= 0:
        raise ValueError("Policy requires a positive reference_neurons count")
    if not isinstance(policy.get("reference_membership_verified"), bool):
        raise ValueError("Policy must state reference_membership_verified")
    if (
        policy.get("edge_policy") != "all-released"
        or policy.get("self_edges_preserved") is not True
        or "edge_threshold" not in policy
        or policy["edge_threshold"] is not None
    ):
        raise ValueError("v0.2 supports all-released edges, preserved self-edges, no threshold")
    if "selection" not in policy:
        raise ValueError("Policy requires an explicit selection")
    return policy


def select_nodes(frame: pd.DataFrame, rule: dict) -> np.ndarray:
    if not isinstance(rule, dict):
        raise ValueError("Selection rule must be a mapping")
    for operator in ("all", "any", "not"):
        if operator not in rule:
            continue
        if set(rule) != {operator}:
            raise ValueError("Logical selection rules cannot have extra fields")
        if operator == "not":
            return ~select_nodes(frame, rule[operator])
        if not isinstance(rule[operator], list):
            raise ValueError(f"{operator} expects a list")
        result = np.full(len(frame), operator == "all", dtype=bool)
        for child in rule[operator]:
            if operator == "all":
                result &= select_nodes(frame, child)
            else:
                result |= select_nodes(frame, child)
        return result
    if set(rule) not in ({"column", "assigned"}, {"column", "in"}):
        raise ValueError("Leaf rule requires column and exactly one of assigned/in")
    if rule["column"] not in frame:
        raise ValueError(f"Missing policy column: {rule['column']}")
    values = frame[rule["column"]]
    if "in" in rule:
        if not isinstance(rule["in"], list):
            raise ValueError("in expects a list")
        return values.isin(rule["in"]).to_numpy(dtype=bool)
    if not isinstance(rule["assigned"], bool):
        raise ValueError("assigned expects a boolean")
    assigned = (values.notna() & values.astype("string").str.strip().ne("")).fillna(False)
    return assigned.to_numpy(dtype=bool) == rule["assigned"]


def exact_ids(values, name: str = "body_id") -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iu" or np.any(array < 0):
        raise ValueError(f"{name} must contain nonnegative integers, never floats or nulls")
    return array.astype("<u8", copy=False)
