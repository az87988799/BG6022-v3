from __future__ import annotations

import pytest

from bg6022.tools.molecule import parse_xyz_bytes, validate_electronic_state

WATER = b"3\nwater\nO 0 0 0\nH 0 1 0\nH 0 -1 0\n"


def test_valid_xyz_and_closed_shell_water() -> None:
    geometry = parse_xyz_bytes(WATER)
    assert geometry.symbols == ("O", "H", "H")
    validate_electronic_state(geometry, charge=0, multiplicity=1)


@pytest.mark.parametrize(
    "content",
    [
        b"2\nwater\nO 0 0 0\nH 0 1 0\nH 0 -1 0\n",
        b"1\nwater\nXx 0 0 0\n",
        b"1\nwater\nH nan 0 0\n",
        b"1\nwater\nH 0 0\n",
    ],
)
def test_invalid_xyz_is_rejected(content: bytes) -> None:
    with pytest.raises(ValueError):
        parse_xyz_bytes(content)


def test_impossible_charge_and_multiplicity_are_rejected() -> None:
    geometry = parse_xyz_bytes(b"1\nhydrogen\nH 0 0 0\n")
    with pytest.raises(ValueError, match="parity"):
        validate_electronic_state(geometry, charge=0, multiplicity=1)
