from __future__ import annotations

from collections.abc import Iterable

from .metrics import relative_degradation


def degradation_rows(
    rows: Iterable[dict],
    metric: str,
    reference_condition: str = "clean",
    higher_is_better: bool = True,
) -> list[dict]:
    values = list(rows)
    references = [row for row in values if row["condition"] == reference_condition]
    if len(references) != 1:
        raise ValueError("exactly one reference condition is required")
    clean = float(references[0][metric])
    return [
        {
            **row,
            f"{metric}_relative_degradation": relative_degradation(
                clean, float(row[metric]), higher_is_better=higher_is_better
            ),
        }
        for row in values
    ]
