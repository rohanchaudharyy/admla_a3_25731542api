{{ config(schema='gold', materialized='table') }}

SELECT DISTINCT
    md5(scraped_date::text) AS scraped_date_sk,
    scraped_date,
    extract(year FROM scraped_date)::int AS year_num,
    extract(month FROM scraped_date)::int AS month_num,
    extract(dow FROM scraped_date)::int AS day_of_week,
    to_char(scraped_date, 'YYYY-MM') AS year_month
FROM {{ ref('listings_clean') }}
