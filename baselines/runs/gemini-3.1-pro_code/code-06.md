```python
import pytest

def clamp(value: float, low: float, high: float) -> float:
    """Clamp value into the inclusive range [low, high]."""
    if low > high:
        raise ValueError("low must not exceed high")
    return max(low, min(high, value))

@pytest.mark.parametrize("value, low, high, expected", [
    # Normal behavior
    (5.0, 0.0, 10.0, 5.0),
    (-2.5, -5.0, 5.0, -2.5),
    (15.0, 0.0, 10.0, 10.0),
    (-10.0, -5.0, 5.0, -5.0),
    
    # Boundary values
    (0.0, 0.0, 10.0, 0.0),
    (10.0, 0.0, 10.0, 10.0),
    (5.0, 5.0, 5.0, 5.0),
    
    # Float precision cases
    (10.0001, 0.0, 10.0, 10.0),
    (-0.0001, 0.0, 10.0, 0.0),
])
def test_clamp_valid_ranges(value, low, high, expected):
    assert clamp(value, low, high) == expected

def test_clamp_invalid_range():
    with pytest.raises(ValueError, match="low must not exceed high"):
        clamp(5.0, 10.0, 0.0)
```