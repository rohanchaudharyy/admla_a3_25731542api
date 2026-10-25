{{ config(schema='gold', materialized='table') }}

SELECT
    md5(listing_id::text || '|' || coalesce(dbt_valid_from::text, '')) AS property_sk,
    listing_id,
    property_type,
    room_type,
    accommodates,
    dbt_valid_from,
    dbt_valid_to,
    dbt_updated_at
FROM {{ ref('property_snapshot') }}
