from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import Any


@dataclass
class PackedCausalCollator:
    """
    Converts packed rows of length context_length + 1 into:

        input_ids = row[:-1]
        labels    = row[1:]

    No padding is needed because all rows are already fixed-length.
    """

    context_length: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        rows = []

        for feature in features:
            row = feature["input_ids"]

            if isinstance(row, torch.Tensor):
                row = row.to(dtype=torch.long)
            else:
                row = torch.tensor(row, dtype=torch.long)

            expected_length = self.context_length + 1
            if row.ndim != 1 or row.shape[0] != expected_length:
                raise ValueError(
                    f"Expected input_ids shape ({expected_length},), "
                    f"got {tuple(row.shape)}"
                )

            rows.append(row)

        packed = torch.stack(rows, dim=0)

        return {
            "input_ids": packed[:, :-1].contiguous(),
            "labels": packed[:, 1:].contiguous(),
            "attention_mask": torch.ones(
                packed.shape[0],
                self.context_length,
                dtype=torch.long,
            ),
        }