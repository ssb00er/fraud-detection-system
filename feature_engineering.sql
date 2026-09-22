-- ============================================================
-- 1. Hour-of-day derived from tx_time (seconds elapsed since first transaction)
-- ============================================================
SELECT
    id,
    tx_time,
    MOD(FLOOR(tx_time / 3600)::int, 24) AS hour_of_day,
    amount,
    class
FROM transactions
LIMIT 20;

-- ============================================================
-- 2. Amount percentile bucket using window functions
-- Flags whether a transaction's amount is in the top 1% of all amounts
-- (a classic simple "is this an outlier amount" fraud signal)
-- ============================================================
WITH amount_percentiles AS (
    SELECT
        id,
        amount,
        class,
        PERCENT_RANK() OVER (ORDER BY amount) AS amount_percentile
    FROM transactions
)
SELECT
    id,
    amount,
    class,
    amount_percentile,
    CASE WHEN amount_percentile >= 0.99 THEN 1 ELSE 0 END AS is_high_amount_outlier
FROM amount_percentiles
ORDER BY amount DESC
LIMIT 20;

-- ============================================================
-- 3. Rolling transaction count and average amount using a time-ordered window
-- Since there's no user_id in this dataset, this simulates "recent activity"
-- by using overall transaction order as a proxy — in a real dataset with
-- user_id, you would PARTITION BY user_id here.
-- ============================================================
SELECT
    id,
    tx_time,
    amount,
    class,
    COUNT(*) OVER (
        ORDER BY tx_time
        RANGE BETWEEN INTERVAL '3600' PRECEDING AND CURRENT ROW
    ) AS tx_count_last_hour,  -- Note: RANGE with INTERVAL requires a timestamp type;
                              -- see the numeric-safe version below if tx_time is NUMERIC.
    AVG(amount) OVER (
        ORDER BY tx_time
        ROWS BETWEEN 50 PRECEDING AND CURRENT ROW
    ) AS avg_amount_last_50_tx
FROM transactions
ORDER BY tx_time
LIMIT 20;

-- If the query above errors because tx_time is NUMERIC (not a timestamp),
-- use this numeric-safe version instead for the "last hour" style window:
SELECT
    id,
    tx_time,
    amount,
    class,
    COUNT(*) OVER (
        ORDER BY tx_time
        RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
    ) AS tx_count_last_hour_numeric,
    AVG(amount) OVER (
        ORDER BY tx_time
        ROWS BETWEEN 50 PRECEDING AND CURRENT ROW
    ) AS avg_amount_last_50_tx
FROM transactions
ORDER BY tx_time
LIMIT 20;

-- ============================================================
-- 4. Class balance summary (sanity check, aggregation)
-- ============================================================
SELECT
    class,
    COUNT(*) AS tx_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 4) AS pct_of_total,
    ROUND(AVG(amount), 2) AS avg_amount,
    ROUND(MIN(amount), 2) AS min_amount,
    ROUND(MAX(amount), 2) AS max_amount
FROM transactions
GROUP BY class;

-- ============================================================
-- 5. Materializing engineered features into a new table
-- Once you've decided which features are useful from the exploration above,
-- create a features table your model training script will read from.
-- This keeps raw data and engineered features cleanly separated.
-- ============================================================
DROP TABLE IF EXISTS transaction_features;
 
CREATE TABLE transaction_features AS
SELECT
    id,
    tx_time,
    amount,
    MOD(FLOOR(tx_time / 3600)::int, 24) AS hour_of_day,
    PERCENT_RANK() OVER (ORDER BY amount) AS amount_percentile,
    AVG(amount) OVER (
        ORDER BY tx_time
        ROWS BETWEEN 50 PRECEDING AND CURRENT ROW
    ) AS avg_amount_last_50_tx,
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
    v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
    v21, v22, v23, v24, v25, v26, v27, v28,
    class
FROM transactions;

-- Verify:
-- SELECT COUNT(*) FROM transaction_features;