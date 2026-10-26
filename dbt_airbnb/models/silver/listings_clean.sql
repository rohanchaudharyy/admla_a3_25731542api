{{
  config(
    schema='silver',
    materialized='incremental',
    unique_key=['listing_id', 'scraped_date']
  )
}}

/*
  Clean bronze.raw_listings (all TEXT from airbnb_census_elt_bronze_load).
  Casts, flags, stays / estimated revenue.
  Active listing := has_availability = 't'
  Stays (active) := 30 - availability_30
*/

WITH src AS (
    SELECT
        nullif(trim(listing_id), '')::bigint AS listing_id,
        nullif(trim(scrape_id), '')::bigint AS scrape_id,
        CASE
            WHEN nullif(trim(scraped_date), '') IS NULL THEN NULL
            WHEN trim(scraped_date) ~ '^\d{4}-\d{2}-\d{2}'
                THEN trim(scraped_date)::date
            WHEN trim(scraped_date) ~ '^\d{1,2}/\d{1,2}/\d{4}'
                THEN to_date(trim(scraped_date), 'DD/MM/YYYY')
            ELSE trim(scraped_date)::date
        END AS scraped_date,
        nullif(trim(host_id), '')::bigint AS host_id,
        nullif(trim(host_name), '') AS host_name,
        CASE
            WHEN nullif(trim(host_since), '') IS NULL THEN NULL
            WHEN trim(host_since) ~ '^\d{4}-\d{2}-\d{2}'
                THEN trim(host_since)::date
            WHEN trim(host_since) ~ '^\d{1,2}/\d{1,2}/\d{4}'
                THEN to_date(trim(host_since), 'DD/MM/YYYY')
            ELSE trim(host_since)::date
        END AS host_since,
        lower(trim(host_is_superhost)) AS host_is_superhost_raw,
        nullif(trim(host_neighbourhood), '') AS host_neighbourhood,
        nullif(trim(listing_neighbourhood), '') AS listing_neighbourhood,
        nullif(trim(property_type), '') AS property_type,
        nullif(trim(room_type), '') AS room_type,
        nullif(trim(accommodates), '')::integer AS accommodates,
        nullif(replace(replace(trim(price), '$', ''), ',', ''), '')::numeric AS price,
        lower(trim(has_availability)) AS has_availability_raw,
        nullif(trim(availability_30), '')::integer AS availability_30,
        nullif(trim(number_of_reviews), '')::integer AS number_of_reviews,
        nullif(trim(review_scores_rating), '')::numeric AS review_scores_rating,
        nullif(trim(review_scores_accuracy), '')::numeric AS review_scores_accuracy,
        nullif(trim(review_scores_cleanliness), '')::numeric AS review_scores_cleanliness,
        nullif(trim(review_scores_checkin), '')::numeric AS review_scores_checkin,
        nullif(trim(review_scores_communication), '')::numeric AS review_scores_communication,
        nullif(trim(review_scores_value), '')::numeric AS review_scores_value,
        source_file,
        loaded_at
    FROM {{ source('bronze', 'raw_listings') }}
    {% if is_incremental() %}
    WHERE loaded_at > (SELECT coalesce(max(loaded_at), '1900-01-01'::timestamp) FROM {{ this }})
    {% endif %}
)

SELECT
    listing_id,
    scrape_id,
    scraped_date,
    host_id,
    host_name,
    host_since,
    CASE
        WHEN host_is_superhost_raw IN ('t', 'true', '1') THEN TRUE
        WHEN host_is_superhost_raw IN ('f', 'false', '0') THEN FALSE
        ELSE NULL
    END AS host_is_superhost,
    host_neighbourhood,
    listing_neighbourhood,
    property_type,
    room_type,
    accommodates,
    price,
    CASE
        WHEN has_availability_raw IN ('t', 'true', '1') THEN TRUE
        WHEN has_availability_raw IN ('f', 'false', '0') THEN FALSE
        ELSE NULL
    END AS has_availability,
    availability_30,
    number_of_reviews,
    review_scores_rating,
    review_scores_accuracy,
    review_scores_cleanliness,
    review_scores_checkin,
    review_scores_communication,
    review_scores_value,
    source_file,
    loaded_at,
    CASE
        WHEN has_availability_raw IN ('t', 'true', '1')
            THEN greatest(0, 30 - coalesce(availability_30, 0))
        ELSE 0
    END AS number_of_stays,
    CASE
        WHEN has_availability_raw IN ('t', 'true', '1')
            THEN greatest(0, 30 - coalesce(availability_30, 0)) * coalesce(price, 0)
        ELSE 0
    END AS estimated_revenue,
    extract(year FROM scraped_date)::int AS year_num,
    extract(month FROM scraped_date)::int AS month_num,
    to_char(scraped_date, 'YYYY-MM') AS year_month
FROM src
WHERE listing_id IS NOT NULL
  AND scraped_date IS NOT NULL
