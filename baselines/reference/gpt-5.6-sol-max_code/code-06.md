```python
import pytest

from your_module import clamp


@pytest.mark.parametrize(
    ("value", "low", "high", "expected"),
    [
        (5.0, 0.0, 10.0, 5.0),    # inside the range
        (-2.0, 0.0, 10.0, 0.0),   # below the range
        (12.0, 0.0, 10.0, 10.0),  # above the range
        (0.0, 0.0, 10.0, 0.0),    # lower boundary
        (10.0, 0.0, 10.0, 10.0),  # upper boundary
        (5.0, 5.0, 5.0, 5.0),     # equal boundaries
    ],
)
def test_clamp(value, low, high, expected):
    assert clamp(value, low, high) == expected


def test_clamp_rejects_inverted_range():
    with pytest.raises(ValueError, match="low must not exceed high"):
        clamp(5.0, 10.0, 0.0)
```