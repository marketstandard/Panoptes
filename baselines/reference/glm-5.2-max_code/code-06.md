```python
import pytest

from clamp import clamp


@pytest.mark.parametrize(
    "value, low, high, expected",
    [
        # Normal behavior: value sits inside the range.
        (5, 0, 10, 5),
        (5.5, 0.0, 10.0, 5.5),
        # Lower boundary: value below low is clamped up.
        (-1, 0, 10, 0),
        (0, 0, 10, 0),       # exactly at low
        # Upper boundary: value above high is clamped down.
        (11, 0, 10, 10),
        (10, 0, 10, 10),     # exactly at high
        # Degenerate range: low == high collapses to a single value.
        (7, 4, 4, 4),
        (4, 4, 4, 4),
        # Negative ranges.
        (-5, -10, -1, -5),
        (-20, -10, -1, -10),
    ],
)
def test_clamp_normal_and_boundaries(value, low, high, expected):
    assert clamp(value, low, high) == expected


def test_clamp_returns_float_for_float_inputs():
    result = clamp(2.5, 0.0, 5.0)
    assert isinstance(result, float)
    assert result == 2.5


def test_clamp_raises_when_low_exceeds_high():
    with pytest.raises(ValueError, match="low must not exceed high"):
        clamp(5, 10, 0)


def test_clamp_error_message_exact():
    with pytest.raises(ValueError) as exc_info:
        clamp(0, 1, 0)
    assert str(exc_info.value) == "low must not exceed high"
```
