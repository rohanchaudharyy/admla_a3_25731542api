{{ config(schema='gold', materialized='table') }}

SELECT
    md5(lga_code) AS lga_sk,
    lga_code,
    lga_name,
    lga_name_norm
FROM {{ ref('lga_code_clean') }}
