```python
import math

import pytest

from clamp_module import clamp  # adjust to wherever clamp lives


class TestNormalBehavior:
    @pytest.mark.parametrize(
        ("value", "low", "high", "expected"),
        [
            (5.0, 0.0, 10.0, 5.0),      # inside the range, untouched
            (-3.0, 0.0, 10.0, 0.0),     # below the range, pulled up to low
            (42.0, 0.0, 10.0, 10.0),    # above the range, pulled down to high
            (-5.0, -10.0, -1.0, -5.0),  # entirely negative range
            (0.0, -1.0, 1.0, 0.0),      # range straddling zero
            (7, 0, 10, 7),              # ints work as well as floats
        ],
    )
    def test_clamps_to_expected_value(self, value, low, high, expected):
        assert clamp(value, low, high) == expected


class TestBoundaries:
    @pytest.mark.parametrize("value", [0.0, 10.0])
    def test_endpoints_are_inclusive(self, value):
        assert clamp(value, 0.0, 10.0) == value

    def test_degenerate_range_returns_that_single_point(self):
        assert clamp(5.0, 3.0, 3.0) == 3.0
        assert clamp(3.0, 3.0, 3.0) == 3.0
        assert clamp(1.0, 3.0, 3.0) == 3.0

    def test_just_inside_and_just_outside_the_edge(self):
        eps = 1e-9
        assert clamp(0.0 - eps, 0.0, 10.0) == 0.0
        assert clamp(0.0 + eps, 0.0, 10.0) == pytest.approx(eps)
        assert clamp(10.0 + eps, 0.0, 10.0) == 10.0

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(math.inf, 10.0), (-math.inf, 0.0)],
    )
    def test_infinities_are_clamped(self, value, expected):
        assert clamp(value, 0.0, 10.0) == expected

    def test_infinite_bounds_pass_the_value_through(self):
        assert clamp(1234.5, -math.inf, math.inf) == 1234.5


class TestErrorCase:
    def test_low_greater_than_high_raises(self):
        with pytest.raises(ValueError, match="low must not exceed high"):
            clamp(5.0, 10.0, 0.0)

    def test_validation_happens_before_any_clamping(self):
        # Even a value that would be "fine" under either bound must still raise.
        with pytest.raises(ValueError):
            clamp(7.0, 10.0, 0.0)

    def test_equal_bounds_do_not_raise(self):
        assert clamp(7.0, 3.0, 3.0) == 3.0


class TestKnownSharpEdges:
    """Documents how the current implementation behaves; adjust if the contract changes."""

    def test_nan_value_falls_through_to_high(self):
        # min(high, nan) is nan, then max(low, nan) is nan under CPython's
        # short-circuiting comparisons -- so NaN in means NaN out.
        assert math.isnan(clamp(math.nan, 0.0, 10.0))

    def test_negative_zero_compares_equal_to_zero(self):
        assert clamp(-0.0, 0.0, 10.0) == 0.0
```

What each group is doing:

- **Normal behavior** is parametrized because the six cases differ only in data — a table reads far better than six near-identical test bodies, and a failure names the exact row.
- **Boundaries** covers the inclusive endpoints stated in the docstring, the degenerate `low == high` range, values an epsilon to either side of an edge, and infinities. `low == high` sits deliberately in this group rather than the error group, since the guard is `low > high`.
- **Error case** asserts both the exception type and its message via `match`, so a future refactor that raises `ValueError` for some unrelated reason won't silently keep the test green.
- **Sharp edges** pins down NaN handling. This isn't in the docstring's contract, and if you decide NaN should raise instead, that test is where the decision gets recorded.
