{{ config(schema='gold', materialized='table') }}

/*
  Star-schema fact: surrogate / natural keys + metrics only.
  SCD2 point-in-time keys resolved against host / listing / property snapshots
  valid at scraped_date.
*/

WITH base AS (
    SELECT *
    FROM {{ ref('listings_clean') }}
),

with_keys AS (
    SELECT
        b.listing_id,
        b.scraped_date,
        b.host_id,

        h.host_sk,
        l.listing_sk,
        p.property_sk,
        pt.property_type_sk,
        rt.room_type_sk,
        a.accommodates_sk,
        ln.listing_neighbourhood_sk,
        hn.host_neighbourhood_sk,
        lg.lga_sk,
        ym.year_month_sk,
        sd.scraped_date_sk,
        av.availability_status_sk,
        sh.superhost_status_sk,
        rb.review_score_band_sk,
        pr.price_tier_sk,
        cg01.census_g01_sk,
        cg02.census_g02_sk,

        /* natural / degenerate helpers used by Part 4 */
        ln.listing_neighbourhood,
        ln.lga_code,
        hn.host_neighbourhood_lga,
        hn.host_neighbourhood_lga_code,

        /* metrics – explicit numeric types for Part 4 aggregations */
        b.price::numeric AS price,
        b.availability_30::integer AS availability_30,
        b.number_of_reviews::integer AS number_of_reviews,
        b.review_scores_rating::numeric AS review_scores_rating,
        b.review_scores_accuracy::numeric AS review_scores_accuracy,
        b.review_scores_cleanliness::numeric AS review_scores_cleanliness,
        b.review_scores_checkin::numeric AS review_scores_checkin,
        b.review_scores_communication::numeric AS review_scores_communication,
        b.review_scores_value::numeric AS review_scores_value,
        b.has_availability,
        b.number_of_stays::numeric AS number_of_stays,
        b.estimated_revenue::numeric AS estimated_revenue,
        b.year_num,
        b.month_num,
        b.year_month
    FROM base b

    LEFT JOIN {{ ref('dim_host') }} h
        ON b.host_id = h.host_id
       AND b.scraped_date >= h.dbt_valid_from::date
       AND (h.dbt_valid_to IS NULL OR b.scraped_date < h.dbt_valid_to::date)

    LEFT JOIN {{ ref('dim_listing') }} l
        ON b.listing_id = l.listing_id
       AND b.scraped_date >= l.dbt_valid_from::date
       AND (l.dbt_valid_to IS NULL OR b.scraped_date < l.dbt_valid_to::date)

    LEFT JOIN {{ ref('dim_property') }} p
        ON b.listing_id = p.listing_id
       AND b.scraped_date >= p.dbt_valid_from::date
       AND (p.dbt_valid_to IS NULL OR b.scraped_date < p.dbt_valid_to::date)

    LEFT JOIN {{ ref('dim_property_type') }} pt
        ON lower(coalesce(b.property_type, 'unknown')) = lower(coalesce(pt.property_type, 'unknown'))

    LEFT JOIN {{ ref('dim_room_type') }} rt
        ON lower(coalesce(b.room_type, 'unknown')) = lower(coalesce(rt.room_type, 'unknown'))

    LEFT JOIN {{ ref('dim_accommodates') }} a
        ON b.accommodates IS NOT DISTINCT FROM a.accommodates

    LEFT JOIN {{ ref('dim_listing_neighbourhood') }} ln
        ON lower(b.listing_neighbourhood) = lower(ln.listing_neighbourhood)

    LEFT JOIN {{ ref('dim_host_neighbourhood') }} hn
        ON lower(b.host_neighbourhood) = lower(hn.host_neighbourhood)

    LEFT JOIN {{ ref('dim_lga') }} lg
        ON ln.lga_code = lg.lga_code

    LEFT JOIN {{ ref('dim_year_month') }} ym
        ON b.year_month = ym.year_month

    LEFT JOIN {{ ref('dim_scraped_date') }} sd
        ON b.scraped_date = sd.scraped_date

    LEFT JOIN {{ ref('dim_availability_status') }} av
        ON b.has_availability IS NOT DISTINCT FROM av.has_availability

    LEFT JOIN {{ ref('dim_superhost_status') }} sh
        ON b.host_is_superhost IS NOT DISTINCT FROM sh.host_is_superhost

    LEFT JOIN {{ ref('dim_review_score_band') }} rb
        ON (
            b.review_scores_rating IS NULL AND rb.score_min IS NULL
        ) OR (
            b.review_scores_rating BETWEEN rb.score_min AND rb.score_max
        )

    LEFT JOIN {{ ref('dim_price_tier') }} pr
        ON (
            b.price IS NULL AND pr.price_min IS NULL
        ) OR (
            b.price BETWEEN pr.price_min AND pr.price_max
        )

    LEFT JOIN {{ ref('ref_census_g01') }} cg01
        ON ln.lga_code = cg01.lga_code

    LEFT JOIN {{ ref('ref_census_g02') }} cg02
        ON ln.lga_code = cg02.lga_code
)

SELECT
    md5(listing_id::text || '|' || scraped_date::text) AS listing_fact_sk,
    listing_id,
    scraped_date,
    host_id,
    host_sk,
    listing_sk,
    property_sk,
    property_type_sk,
    room_type_sk,
    accommodates_sk,
    listing_neighbourhood_sk,
    host_neighbourhood_sk,
    lga_sk,
    year_month_sk,
    scraped_date_sk,
    availability_status_sk,
    superhost_status_sk,
    review_score_band_sk,
    price_tier_sk,
    census_g01_sk,
    census_g02_sk,
    listing_neighbourhood,
    lga_code,
    host_neighbourhood_lga,
    host_neighbourhood_lga_code,
    price,
    availability_30,
    number_of_reviews,
    review_scores_rating,
    review_scores_accuracy,
    review_scores_cleanliness,
    review_scores_checkin,
    review_scores_communication,
    review_scores_value,
    has_availability,
    number_of_stays,
    estimated_revenue,
    year_num,
    month_num,
    year_month
FROM with_keys
