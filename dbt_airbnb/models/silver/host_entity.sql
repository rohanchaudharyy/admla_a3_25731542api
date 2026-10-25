{{ config(schema='silver', materialized='table') }}

/*
  Latest known host attributes per scrape window.
  updated_at = scraped_date so timestamp SCD2 snapshots track month-to-month changes
  (superhost flag, neighbourhood) when dbt is run after each Airflow monthly load.
*/

SELECT DISTINCT ON (host_id)
    host_id,
    host_name,
    host_since,
    host_is_superhost,
    host_neighbourhood,
    scraped_date AS updated_at
FROM {{ ref('listings_clean') }}
WHERE host_id IS NOT NULL
ORDER BY host_id, scraped_date DESC
