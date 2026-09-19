"""Transmitter signs are a model policy, never a mutation of anatomical contacts."""

from collections import Counter

import numpy as np

from malecns_sim.neural.config import LIFConfig


def transmitter_signs(labels, config: LIFConfig) -> tuple[np.ndarray, dict]:
    normalized = [None if value is None else str(value).strip().lower() for value in labels]
    mapping = {key.strip().lower(): value for key, value in config.transmitter_sign.items()}
    unknown = Counter(value for value in normalized if value not in mapping)
    if unknown and config.unknown_nt_policy == "error":
        raise ValueError(f"Unmapped neurotransmitters: {dict(unknown)}")
    fallback = 1 if config.unknown_nt_policy == "excitatory" else 0
    signs = np.array([mapping.get(value, fallback) for value in normalized], dtype=np.int8)
    report = {
        "policy": config.unknown_nt_policy,
        "mapped_nodes": len(signs) - sum(unknown.values()),
        "unmapped_nodes": sum(unknown.values()),
        "unmapped_labels": [
            {"label": label, "nodes": count}
            for label, count in sorted(
                unknown.items(), key=lambda item: (item[0] is not None, item[0] or "")
            )
        ],
        "positive_nodes": int((signs > 0).sum()),
        "negative_nodes": int((signs < 0).sum()),
        "zero_sign_nodes": int((signs == 0).sum()),
        "reason": (
            "Unmapped labels, including histamine under this six-transmitter policy, "
            "are not assigned a biological sign. Default zero disables their outgoing "
            "influence while retaining the nodes and contacts in the anatomical graph."
        ),
    }
    return signs, report
