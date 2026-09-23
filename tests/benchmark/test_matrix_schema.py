from __future__ import annotations

import pytest
from pydantic import ValidationError

from bg6022.benchmark.matrix import ScientificObject


@pytest.mark.parametrize(
    ("charge", "multiplicity"),
    [(1, 1), (0, 2), (-1, 2)],
)
def test_scientific_matrix_v1_rejects_non_neutral_singlets(charge, multiplicity) -> None:
    with pytest.raises(ValidationError, match="neutral closed-shell singlets"):
        ScientificObject.model_validate(
            {
                "id": "water",
                "label": "H2O",
                "geometry": "geometries/water.xyz",
                "charge": charge,
                "multiplicity": multiplicity,
            },
            strict=True,
        )
