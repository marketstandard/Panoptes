```python
import pytest

from clamp_module import clamp


@pytest.mark.parametrize(
    "value, low, high, expected",
    [
        (5.0, 0.0, 10.0, 5.0),
        (-3.5, -10.0, 10.0, -3.5),
        (2.5, 2.5, 2.5, 2.5),
    ],
)
def test_clamp_returns_value_inside_range(value, low, high, expected):
    assert clamp(value, low, high) == expected


@pytest.mark.parametrize(
    "value, low, high, expected",
    [
        (0.0, 0.0, 10.0, 0.0),
        (10.0, 0.0, 10.0, 10.0),
        (-1.0, 0.0, 10.0, 0.0),
        (11.0, 0.0, 10.0, 10.0),
        (0.0, 0.0, 0.0, 0.0),
    ],
)
def test_clamp_boundaries(value, low, high, expected):
    assert clamp(value, low, high) == expected


@pytest.mark.parametrize(
    "value, low, high",
    [
        (1.0, 10.0, 0.0),
        (0.0, 1.0, 0.999),
    ],
)
def test_clamp_raises_when_low_exceeds_high(value, low, high):
    with pytest.raises(ValueError, match="low must not exceed high"):
        clamp(value, low, high)
```
