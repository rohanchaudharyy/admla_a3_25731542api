{{ config(schema='gold', materialized='table') }}

SELECT
    md5(coalesce(accommodates::text, 'unknown')) AS accommodates_sk,
    accommodates,
    CASE
        WHEN accommodates IS NULL THEN 'Unknown'
        WHEN accommodates <= 2 THEN '1-2 guests'
        WHEN accommodates <= 4 THEN '3-4 guests'
        WHEN accommodates <= 6 THEN '5-6 guests'
        ELSE '7+ guests'
    END AS accommodates_band
FROM (
    SELECT DISTINCT accommodates
    FROM {{ ref('listings_clean') }}
) x
