{{ config(schema='gold', materialized='table') }}

SELECT
    md5(lower(coalesce(room_type, 'unknown'))) AS room_type_sk,
    room_type
FROM (
    SELECT DISTINCT room_type
    FROM {{ ref('listings_clean') }}
) x
