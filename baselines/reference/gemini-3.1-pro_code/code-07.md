```sql
SELECT 
    customer_id, 
    COUNT(id) AS order_count, 
    ROUND(SUM(total_cents) / 100.0, 2) AS total_dollars
FROM orders
GROUP BY customer_id
HAVING COUNT(id) > 3
ORDER BY total_dollars DESC;
```