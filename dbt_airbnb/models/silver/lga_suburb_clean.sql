{{ config(schema='silver', materialized='table') }}

SELECT DISTINCT
    trim(s.lga_name) AS lga_name,
    lower(trim(s.lga_name)) AS lga_name_norm,
    trim(s.suburb_name) AS suburb_name,
    lower(trim(s.suburb_name)) AS suburb_name_norm,
    c.lga_code
FROM {{ source('bronze', 'raw_nsw_lga_suburb') }} s
LEFT JOIN {{ ref('lga_code_clean') }} c
    ON lower(trim(s.lga_name)) = c.lga_name_norm
WHERE nullif(trim(s.suburb_name), '') IS NOT NULL
