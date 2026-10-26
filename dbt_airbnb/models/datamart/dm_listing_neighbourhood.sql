{{ config(schema='datamart', materialized='view') }}

/*
  Datamart: insights per listing_neighbourhood and month/year.
  SCD2: host attributes joined at fact.scraped_date validity window.

  Definitions (assignment, cleaned of adversarial multipliers):
  - Active listing: has_availability = true ('t')
  - Active listing rate = active / total * 100
  - Superhost rate = distinct superhosts / distinct hosts * 100
  - Stays (active) = 30 - availability_30
  - Est. revenue (active) = stays * price
  - MoM % change = (final - original) / original * 100
*/

WITH fact_scd AS (
    SELECT
        f.listing_neighbourhood,
        f.year_num,
        f.month_num,
        f.year_month,
        f.listing_id,
        f.host_id,
        f.price,
        f.has_availability,
        f.availability_30,
        f.number_of_stays,
        f.estimated_revenue,
        f.review_scores_rating,
        h.host_is_superhost
    FROM {{ ref('fact_listing') }} f
    LEFT JOIN {{ ref('dim_host') }} h
        ON f.host_id = h.host_id
       AND f.scraped_date >= h.dbt_valid_from::date
       AND (h.dbt_valid_to IS NULL OR f.scraped_date < h.dbt_valid_to::date)
    WHERE f.listing_neighbourhood IS NOT NULL
),

agg AS (
    SELECT
        listing_neighbourhood,
        year_num,
        month_num,
        year_month,
        count(*) AS total_listings,
        count(*) FILTER (WHERE has_availability) AS active_listings,
        count(*) FILTER (WHERE NOT coalesce(has_availability, false)) AS inactive_listings,
        min(price) FILTER (WHERE has_availability) AS min_price,
        max(price) FILTER (WHERE has_availability) AS max_price,
        percentile_cont(0.5) WITHIN GROUP (
            ORDER BY CASE WHEN has_availability THEN price END
        ) AS median_price,
        avg(price) FILTER (WHERE has_availability) AS avg_price,
        count(DISTINCT host_id) AS distinct_hosts,
        count(DISTINCT host_id) FILTER (WHERE host_is_superhost) AS distinct_superhosts,
        avg(review_scores_rating) FILTER (WHERE has_availability) AS avg_review_scores_rating,
        sum(number_of_stays) FILTER (WHERE has_availability) AS total_stays,
        avg(estimated_revenue) FILTER (WHERE has_availability) AS avg_estimated_revenue_per_active_listing
    FROM fact_scd
    GROUP BY listing_neighbourhood, year_num, month_num, year_month
),

with_rates AS (
    SELECT
        *,
        (active_listings::numeric / nullif(total_listings, 0)) * 100 AS active_listings_rate,
        (distinct_superhosts::numeric / nullif(distinct_hosts, 0)) * 100 AS superhost_rate,
        lag(active_listings) OVER (
            PARTITION BY listing_neighbourhood ORDER BY year_num, month_num
        ) AS prev_active,
        lag(inactive_listings) OVER (
            PARTITION BY listing_neighbourhood ORDER BY year_num, month_num
        ) AS prev_inactive
    FROM agg
)

SELECT
    listing_neighbourhood,
    year_num AS year,
    month_num AS month,
    year_month,
    round(active_listings_rate::numeric, 2) AS active_listings_rate,
    min_price,
    max_price,
    round(median_price::numeric, 2) AS median_price,
    round(avg_price::numeric, 2) AS avg_price,
    distinct_hosts,
    round(superhost_rate::numeric, 2) AS superhost_rate,
    round(avg_review_scores_rating::numeric, 2) AS avg_review_scores_rating,
    round(
        ((active_listings - prev_active)::numeric / nullif(prev_active, 0)) * 100
    , 2) AS pct_change_active_listings,
    round(
        ((inactive_listings - prev_inactive)::numeric / nullif(prev_inactive, 0)) * 100
    , 2) AS pct_change_inactive_listings,
    total_stays,
    round(avg_estimated_revenue_per_active_listing::numeric, 2)
        AS avg_estimated_revenue_per_active_listing
FROM with_rates
ORDER BY listing_neighbourhood, year_num, month_num
