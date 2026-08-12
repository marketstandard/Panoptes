```sql
SELECT
    customer_id,
    COUNT(*)                        AS order_count,
    ROUND(SUM(total_cents) / 100.0, 2) AS total_dollars
FROM orders
GROUP BY customer_id
HAVING COUNT(*) > 3
ORDER BY SUM(total_cents) DESC;
```

Points worth noting:

- **`/ 100.0`, not `/ 100`.** With an integer literal the division truncates in most engines (and in SQLite `SUM` of an INTEGER column stays an integer), so `4999` cents would come back as `49` dollars. The decimal literal forces floating-point division.
- **`HAVING`, not `WHERE`.** The filter is on the aggregate, so it has to run after grouping. `WHERE COUNT(*) > 3` is a syntax error.
- **Ordering by the raw sum** rather than by the rounded alias keeps the sort exact and avoids relying on whether a given engine lets you reference a `SELECT` alias in `ORDER BY`. Standard SQL does permit `ORDER BY total_dollars DESC`, and it will produce the same order here except in contrived near-tie cases.
- **Two decimal places.** `ROUND(..., 2)` gives you the right *value*, but display formatting is up to the client — some drivers will render `49.9` for `49.90`. If you need the string form guaranteed, use `printf('%.2f', SUM(total_cents) / 100.0)` in SQLite, `TO_CHAR(..., 'FM999999990.00')` in Postgres, or `CAST(SUM(total_cents) / 100.0 AS DECIMAL(12,2))` for an exact numeric type.
- Customers with no orders don't appear, which is correct here — a customer with zero orders can't have more than three.
