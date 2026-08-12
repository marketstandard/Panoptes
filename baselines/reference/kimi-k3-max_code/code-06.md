```python
import pytest

from solution import clamp


@pytest.mark.parametrize(
    "value, low, high, expected",
    [
        (5, 0, 10, 5),            # normal: value inside range
        (-3, -10, 10, -3),        # negative value inside range
        (0, 0, 10, 0),            # boundary: exactly at low
        (10, 0, 10, 10),          # boundary: exactly at high
        (-1, 0, 10, 0),           # below range clamps to low
        (11, 0, 10, 10),          # above range clamps to high
        (5.5, 0.0, 10.0, 5.5),    # floats pass through
        (3, 3, 3, 3),             # boundary: low == high
        (2, 3, 3, 3),             # below degenerate range
        (4, 3, 3, 3),             # above degenerate range
    ],
)
def test_clamp_returns_expected(value, low, high, expected):
    assert clamp(value, low, high) == expected


@pytest.mark.parametrize("low, high", [(10, 0), (1, -1), (0.5, 0.4)])
def test_clamp_rejects_inverted_bounds(low, high):
    with pytest.raises(ValueError, match="low must not exceed high"):
        clamp(1, low, high)
```
