{{ config(schema='silver', materialized='table') }}

SELECT
    trim(lga_code)::text AS lga_code,
    trim(lga_name) AS lga_name,
    lower(trim(lga_name)) AS lga_name_norm
FROM {{ source('bronze', 'lga_code_raw') }}
WHERE lga_code IS NOT NULL
  AND lga_name IS NOT NULL
