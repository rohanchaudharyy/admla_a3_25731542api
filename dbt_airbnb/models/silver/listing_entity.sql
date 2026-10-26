{{ config(schema='silver', materialized='table') }}

/*
  Listing identity attributes that evolve across scrapes.
  Snapshot with timestamp strategy on updated_at = scraped_date.
*/

SELECT DISTINCT ON (listing_id)
    listing_id,
    host_id,
    listing_neighbourhood,
    property_type,
    room_type,
    accommodates,
    scraped_date AS updated_at
FROM {{ ref('listings_clean') }}
WHERE listing_id IS NOT NULL
ORDER BY listing_id, scraped_date DESC
