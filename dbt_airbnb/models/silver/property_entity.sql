{{ config(schema='silver', materialized='table') }}

/*
  Property characteristics bound to a listing (dimension-bound entity).
  Snapshotted so Gold property dims retain SCD2 history.
*/

SELECT DISTINCT ON (listing_id)
    listing_id,
    property_type,
    room_type,
    accommodates,
    scraped_date AS updated_at
FROM {{ ref('listings_clean') }}
WHERE listing_id IS NOT NULL
ORDER BY listing_id, scraped_date DESC
