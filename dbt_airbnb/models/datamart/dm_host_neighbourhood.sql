{{ config(schema='datamart', materialized='view') }}

/*
  Datamart: host_neighbourhood_lga (suburb → LGA) × month/year.
  - Number of distinct hosts
  - Average estimated revenue per active listing
  - Estimated revenue per host = total est. revenue / distinct hosts
*/

WITH fact_scd AS (
    SELECT
        coalesce(hn.host_neighbourhood_lga, 'Unknown') AS host_neighbourhood_lga,
        f.year_num,
        f.month_num,
        f.year_month,
        f.host_id,
        f.has_availability,
        f.estimated_revenue
    FROM {{ ref('fact_listing') }} f
    LEFT JOIN {{ ref('dim_host_neighbourhood') }} hn
        ON f.host_neighbourhood_sk = hn.host_neighbourhood_sk
),

agg AS (
    SELECT
        host_neighbourhood_lga,
        year_num,
        month_num,
        year_month,
        count(DISTINCT host_id) AS distinct_hosts,
        avg(estimated_revenue) FILTER (WHERE has_availability)
            AS avg_estimated_revenue_per_active_listing,
        sum(estimated_revenue) FILTER (WHERE has_availability) AS total_estimated_revenue
    FROM fact_scd
    GROUP BY host_neighbourhood_lga, year_num, month_num, year_month
)

SELECT
    host_neighbourhood_lga,
    year_num AS year,
    month_num AS month,
    year_month,
    distinct_hosts,
    round(avg_estimated_revenue_per_active_listing::numeric, 2)
        AS avg_estimated_revenue_per_active_listing,
    round(
        (total_estimated_revenue / nullif(distinct_hosts, 0))::numeric
    , 2) AS estimated_revenue_per_host
FROM agg
ORDER BY host_neighbourhood_lga, year_num, month_num
