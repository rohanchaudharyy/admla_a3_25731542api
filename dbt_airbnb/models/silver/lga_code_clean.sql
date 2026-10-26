{{ config(schema='silver', materialized='table') }}

SELECT
    trim(lga_code)::text AS lga_code,
    trim(lga_name) AS lga_name,
    lower(trim(lga_name)) AS lga_name_norm
FROM {{ source('bronze', 'raw_nsw_lga_code') }}
WHERE nullif(trim(lga_code), '') IS NOT NULL
  AND nullif(trim(lga_name), '') IS NOT NULL
