# Panoptes baseline prompts — code

Version 1.0.0. The machine-readable source of truth is `prompts.manifest.json`; this document is the human-friendly copy-paste view. Keep the two in sync (a test in `baselines/tests/` checks it).

## Run protocol — read before running

1. Start a **fresh session** for every prompt: no chat history, memory, or prior context.
2. Use **product defaults**: no system prompt, no custom instructions, default temperature.
3. **Single turn**: send the prompt verbatim, record the complete first reply. No follow-ups, no regeneration.
4. Disable browsing, code execution, retrieval, and plugins where the interface allows.
5. Save the **raw, unedited reply** as `<prompt-id>.md` (for example `code-01.md`) inside the run folder.
6. Record the exact **model version string** shown by the product, the interface you used (`chat-ui`, `api`, or `agent-chat`), and the date.

Fastest path: run `python baselines/baseline.py scaffold --model <model-slug> --kind code` to create a ready-to-fill run folder, paste each prompt below into the model, save each reply, then `python baselines/baseline.py finalize --run baselines/runs/<model-slug>_code`.

---

## code-01 — Log level counter

- Language: Python
- Tags: `lang:python`, `task:utility`
- Output file: `code-01.md`

```text
Write a Python function `parse_log_levels(text: str) -> dict[str, int]` that returns a count of how many lines in a log file begin with `INFO`, `WARN`, or `ERROR` (case-sensitive). Lines beginning with anything else are ignored. Include a one-line docstring.
```

## code-02 — LRU cache

- Language: Python
- Tags: `lang:python`, `task:data-structure`
- Output file: `code-02.md`

```text
Implement an LRU cache in Python as a class `LRUCache` with `get(key)` and `put(key, value)`, both O(1) on average. The constructor takes a positive integer `capacity`; `get` returns -1 for missing keys. After the code, explain in two sentences why your design meets the complexity requirement.
```

## code-03 — Countdown timer component

- Language: TypeScript (React)
- Tags: `lang:typescript`, `task:ui-component`
- Output file: `code-03.md`

```text
Write a React component in TypeScript named `CountdownTimer`. It accepts a prop `seconds` and displays the remaining time formatted as `mm:ss`, updating once per second until it reaches `00:00`. Include the props interface and clean up any timers.
```

## code-04 — Largest files CLI

- Language: Python
- Tags: `lang:python`, `task:cli`
- Output file: `code-04.md`

```text
Write a Python command-line script that takes a directory path as its only argument and prints the five largest regular files directly inside that directory (not recursive), largest first, with sizes shown in megabytes to two decimals. Handle the case where the directory does not exist.
```

## code-05 — Second-largest bug fix

- Language: Python
- Tags: `lang:python`, `task:bug-fix`
- Output file: `code-05.md`

````text
The Python function below should return the second-largest distinct value in a list of numbers, or None if no such value exists. It contains a bug. In one or two sentences explain the bug, then provide the corrected function.

```python
def second_largest(values):
    values = sorted(values)
    if len(values) < 2:
        return None
    return values[-2]
```
````

## code-06 — Clamp test suite

- Language: Python (pytest)
- Tags: `lang:python`, `task:test-writing`
- Output file: `code-06.md`

````text
Write a pytest test suite for the Python function below. Cover normal behavior, boundary values, and the error case. Use parametrization where it improves clarity.

```python
def clamp(value: float, low: float, high: float) -> float:
    """Clamp value into the inclusive range [low, high]."""
    if low > high:
        raise ValueError("low must not exceed high")
    return max(low, min(high, value))
```
````

## code-07 — Customer order summary query

- Language: SQL
- Tags: `lang:sql`, `task:query`
- Output file: `code-07.md`

```text
Given a table `orders(id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL, total_cents INTEGER NOT NULL, created_at TEXT NOT NULL)`, write a SQL query that returns one row per customer with columns `customer_id`, `order_count`, and `total_dollars` (total spend in dollars, two decimal places). Include only customers with more than three orders, and order by total spend descending.
```

## code-08 — TTL cache class

- Language: Python
- Tags: `lang:python`, `task:class-design`
- Output file: `code-08.md`

```text
Design and implement a Python class `TTLCache` using only the standard library. It must support `set(key, value, ttl_seconds)`, `get(key)`, and `delete(key)`. Expired entries must behave exactly as if they were never set. Include a short usage example in a comment at the bottom.
```
