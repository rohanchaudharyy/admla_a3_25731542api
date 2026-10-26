{{ config(schema='gold', materialized='table') }}

SELECT
    md5(listing_id::text || '|' || coalesce(dbt_valid_from::text, '')) AS listing_sk,
    listing_id,
    host_id,
    listing_neighbourhood,
    property_type,
    room_type,
    accommodates,
    dbt_valid_from,
    dbt_valid_to,
    dbt_updated_at
FROM {{ ref('listing_snapshot') }}
