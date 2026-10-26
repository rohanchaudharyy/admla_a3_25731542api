{{ config(schema='gold', materialized='table') }}

SELECT
    md5(lower(coalesce(property_type, 'unknown'))) AS property_type_sk,
    property_type
FROM (
    SELECT DISTINCT property_type
    FROM {{ ref('listings_clean') }}
) x
