{{
  config(
    schema='silver',
    materialized='incremental',
    unique_key=['listing_id', 'scraped_date']
  )
}}

/*
  Clean Bronze listings:
  - normalise boolean flags
  - cast numerics / dates
  - derive stays and estimated revenue for active listings
  Active listing := has_availability = 't'
  Stays (active) := 30 - availability_30
  Est. revenue (active) := stays * price
*/

WITH src AS (
    SELECT
        listing_id::bigint                         AS listing_id,
        scrape_id::bigint                          AS scrape_id,
        scraped_date::date                         AS scraped_date,
        host_id::bigint                            AS host_id,
        nullif(trim(host_name), '')                AS host_name,
        host_since::date                           AS host_since,
        lower(trim(host_is_superhost))             AS host_is_superhost_raw,
        nullif(trim(host_neighbourhood), '')       AS host_neighbourhood,
        nullif(trim(listing_neighbourhood), '')    AS listing_neighbourhood,
        nullif(trim(property_type), '')            AS property_type,
        nullif(trim(room_type), '')                AS room_type,
        accommodates::integer                      AS accommodates,
        replace(price::text, '$', '')::numeric     AS price,
        lower(trim(has_availability))              AS has_availability_raw,
        availability_30::integer                   AS availability_30,
        number_of_reviews::integer                 AS number_of_reviews,
        review_scores_rating::numeric              AS review_scores_rating,
        review_scores_accuracy::numeric            AS review_scores_accuracy,
        review_scores_cleanliness::numeric         AS review_scores_cleanliness,
        review_scores_checkin::numeric             AS review_scores_checkin,
        review_scores_communication::numeric       AS review_scores_communication,
        review_scores_value::numeric               AS review_scores_value
    FROM {{ source('bronze', 'listings_raw') }}
    {% if is_incremental() %}
    WHERE scraped_date > (SELECT coalesce(max(scraped_date), '1900-01-01'::date) FROM {{ this }})
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
    /* metrics helpers */
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
