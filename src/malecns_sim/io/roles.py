"""Normalized, exclusive primary roles used by the I/O catalogue."""

from enum import StrEnum


class NeuronRole(StrEnum):
    SENSORY = "sensory"
    MOTOR = "motor"
    DESCENDING = "descending"
    ASCENDING = "ascending"
    INTERNEURON = "interneuron"
    ENDOCRINE = "endocrine"
    OTHER = "other"
