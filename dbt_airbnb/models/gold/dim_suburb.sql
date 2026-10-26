{{ config(schema='gold', materialized='table') }}

SELECT
    md5(lower(suburb_name) || '|' || coalesce(lga_code, '')) AS suburb_sk,
    suburb_name,
    lga_name,
    lga_code
FROM {{ ref('lga_suburb_clean') }}
