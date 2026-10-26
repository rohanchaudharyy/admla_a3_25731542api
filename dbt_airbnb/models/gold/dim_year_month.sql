{{ config(schema='gold', materialized='table') }}

SELECT
    md5(year_month) AS year_month_sk,
    year_month,
    year_num,
    month_num,
    min(scraped_date) AS month_start_date,
    max(scraped_date) AS month_end_date
FROM {{ ref('listings_clean') }}
GROUP BY year_month, year_num, month_num
