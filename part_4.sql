-- =============================================================================
-- Part 4: Ad-hoc analysis – Sydney Airbnb + Census (May 2020 – Apr 2021)
-- Student: Akshat Dubey
-- Clean brief: 4a top/bottom 3 LGAs; 4b neighbourhood-grain median age corr;
--              4c top 5 neighbourhoods; 4d multi-listing concentration;
--              4e single-listing vs annualised mortgage (ranked LGAs)
-- Run in DBeaver after full dbt build (gold.*). Use ::numeric casts throughout.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 4a. Ranked LGAs (one row per LGA): top 3 and lowest 3 by avg estimated
--     revenue per active listing over the last 12 months, with demographics.
--     revenue_rank is 1–3 within each performance_group.
-- -----------------------------------------------------------------------------
WITH revenue_by_lga AS (
    SELECT
        f.lga_code,
        avg(f.estimated_revenue::numeric) AS avg_revenue_per_active_listing
    FROM gold.fact_listing f
    WHERE f.has_availability IS TRUE
      AND f.lga_code IS NOT NULL
      AND f.scraped_date >= (
            SELECT max(scraped_date) - INTERVAL '12 months' FROM gold.fact_listing
          )
    GROUP BY f.lga_code
),
ranked AS (
    SELECT
        lga_code,
        avg_revenue_per_active_listing,
        dense_rank() OVER (ORDER BY avg_revenue_per_active_listing DESC) AS rank_top,
        dense_rank() OVER (ORDER BY avg_revenue_per_active_listing ASC)  AS rank_bottom
    FROM revenue_by_lga
),
top_bottom AS (
    SELECT
        lga_code,
        avg_revenue_per_active_listing,
        CASE
            WHEN rank_top <= 3 THEN 'Top 3'
            WHEN rank_bottom <= 3 THEN 'Bottom 3'
        END AS performance_group,
        CASE
            WHEN rank_top <= 3 THEN rank_top
            ELSE rank_bottom
        END AS revenue_rank
    FROM ranked
    WHERE rank_top <= 3 OR rank_bottom <= 3
)
SELECT
    tb.performance_group,
    tb.revenue_rank,
    lg.lga_name,
    tb.lga_code,
    round(tb.avg_revenue_per_active_listing, 2) AS avg_revenue_per_active_listing,
    round(
        (g01.age_0_4::numeric + g01.age_5_14::numeric)
            * 100.0 / nullif(g01.total_population::numeric, 0)
    , 2) AS pct_age_0_14,
    round(
        (g01.age_15_19::numeric + g01.age_20_24::numeric + g01.age_25_34::numeric
            + g01.age_35_44::numeric + g01.age_45_54::numeric + g01.age_55_64::numeric)
            * 100.0 / nullif(g01.total_population::numeric, 0)
    , 2) AS pct_age_15_64,
    round(
        (g01.age_65_74::numeric + g01.age_75_84::numeric + g01.age_85_plus::numeric)
            * 100.0 / nullif(g01.total_population::numeric, 0)
    , 2) AS pct_age_65_plus,
    round(g02.average_household_size::numeric, 2) AS average_household_size,
    round(g02.median_age_persons::numeric, 2) AS median_age_persons
FROM top_bottom tb
LEFT JOIN gold.dim_lga lg ON tb.lga_code = lg.lga_code
LEFT JOIN gold.ref_census_g01 g01 ON tb.lga_code = g01.lga_code
LEFT JOIN gold.ref_census_g02 g02 ON tb.lga_code = g02.lga_code
ORDER BY
    CASE tb.performance_group WHEN 'Top 3' THEN 0 ELSE 1 END,
    tb.revenue_rank;


-- -----------------------------------------------------------------------------
-- 4b-i. Neighbourhood scatter (for screenshots): listing_neighbourhood → LGA →
--       Census G02 median_age_persons, with avg revenue per active listing.
-- -----------------------------------------------------------------------------
WITH neighbourhood_metrics AS (
    SELECT
        f.listing_neighbourhood,
        coalesce(f.lga_code, ln.lga_code) AS lga_code,
        avg(f.estimated_revenue::numeric) AS avg_revenue_per_active_listing
    FROM gold.fact_listing f
    LEFT JOIN gold.dim_listing_neighbourhood ln
        ON f.listing_neighbourhood_sk = ln.listing_neighbourhood_sk
    WHERE f.has_availability IS TRUE
      AND f.listing_neighbourhood IS NOT NULL
    GROUP BY f.listing_neighbourhood, coalesce(f.lga_code, ln.lga_code)
)
SELECT
    dense_rank() OVER (ORDER BY n.avg_revenue_per_active_listing DESC)
        AS revenue_rank,
    n.listing_neighbourhood,
    lg.lga_name,
    n.lga_code,
    round(g02.median_age_persons::numeric, 2) AS median_age_persons,
    round(n.avg_revenue_per_active_listing, 2) AS avg_revenue_per_active_listing
FROM neighbourhood_metrics n
INNER JOIN gold.ref_census_g02 g02 ON n.lga_code = g02.lga_code
LEFT JOIN gold.dim_lga lg ON n.lga_code = lg.lga_code
WHERE nullif(trim(g02.median_age_persons::text), '') IS NOT NULL
ORDER BY revenue_rank;


-- -----------------------------------------------------------------------------
-- 4b-ii. Pearson correlation across neighbourhoods (median age vs avg revenue)
-- -----------------------------------------------------------------------------
WITH neighbourhood_metrics AS (
    SELECT
        f.listing_neighbourhood,
        coalesce(f.lga_code, ln.lga_code) AS lga_code,
        avg(f.estimated_revenue::numeric) AS avg_revenue_per_active_listing
    FROM gold.fact_listing f
    LEFT JOIN gold.dim_listing_neighbourhood ln
        ON f.listing_neighbourhood_sk = ln.listing_neighbourhood_sk
    WHERE f.has_availability IS TRUE
      AND f.listing_neighbourhood IS NOT NULL
    GROUP BY f.listing_neighbourhood, coalesce(f.lga_code, ln.lga_code)
),
joined AS (
    SELECT
        n.listing_neighbourhood,
        g02.median_age_persons::numeric AS median_age_persons,
        n.avg_revenue_per_active_listing
    FROM neighbourhood_metrics n
    INNER JOIN gold.ref_census_g02 g02 ON n.lga_code = g02.lga_code
    WHERE nullif(trim(g02.median_age_persons::text), '') IS NOT NULL
)
SELECT
    count(*) AS n_neighbourhoods,
    round(
        corr(median_age_persons, avg_revenue_per_active_listing)::numeric
    , 4) AS pearson_corr_median_age_vs_revenue
FROM joined;


-- -----------------------------------------------------------------------------
-- 4c. Best listing type for top 5 listing_neighbourhoods (by avg est. revenue
--     per active listing), chosen to maximise total stays.
--     neighbourhood_rank is 1–5 by revenue.
-- -----------------------------------------------------------------------------
WITH neighbourhood_revenue AS (
    SELECT
        listing_neighbourhood,
        avg(estimated_revenue::numeric) AS avg_revenue_per_active_listing
    FROM gold.fact_listing
    WHERE has_availability IS TRUE
      AND listing_neighbourhood IS NOT NULL
    GROUP BY listing_neighbourhood
),
top5 AS (
    SELECT *
    FROM (
        SELECT
            listing_neighbourhood,
            avg_revenue_per_active_listing,
            dense_rank() OVER (ORDER BY avg_revenue_per_active_listing DESC)
                AS neighbourhood_rank
        FROM neighbourhood_revenue
    ) r
    WHERE neighbourhood_rank <= 5
),
stays_by_type AS (
    SELECT
        f.listing_neighbourhood,
        pt.property_type,
        rt.room_type,
        ac.accommodates,
        sum(f.number_of_stays::numeric) AS total_stays,
        avg(f.estimated_revenue::numeric) AS avg_revenue_per_active_listing
    FROM gold.fact_listing f
    LEFT JOIN gold.dim_property_type pt ON f.property_type_sk = pt.property_type_sk
    LEFT JOIN gold.dim_room_type rt ON f.room_type_sk = rt.room_type_sk
    LEFT JOIN gold.dim_accommodates ac ON f.accommodates_sk = ac.accommodates_sk
    WHERE f.has_availability IS TRUE
      AND f.listing_neighbourhood IN (SELECT listing_neighbourhood FROM top5)
    GROUP BY f.listing_neighbourhood, pt.property_type, rt.room_type, ac.accommodates
),
best AS (
    SELECT DISTINCT ON (listing_neighbourhood)
        listing_neighbourhood,
        property_type,
        room_type,
        accommodates,
        total_stays,
        avg_revenue_per_active_listing
    FROM stays_by_type
    ORDER BY listing_neighbourhood, total_stays DESC
)
SELECT
    t.neighbourhood_rank,
    b.listing_neighbourhood,
    b.property_type,
    b.room_type,
    b.accommodates,
    b.total_stays,
    round(b.avg_revenue_per_active_listing, 2) AS avg_revenue_per_active_listing
FROM best b
INNER JOIN top5 t ON b.listing_neighbourhood = t.listing_neighbourhood
ORDER BY t.neighbourhood_rank;


-- -----------------------------------------------------------------------------
-- 4d. Multi-listing hosts: properties concentrated in the same LGA, or
--     distributed across different LGAs?
-- -----------------------------------------------------------------------------
WITH host_listings AS (
    SELECT
        host_id,
        lga_code,
        listing_id
    FROM gold.fact_listing
    WHERE host_id IS NOT NULL
      AND lga_code IS NOT NULL
    GROUP BY host_id, lga_code, listing_id
),
host_summary AS (
    SELECT
        host_id,
        count(DISTINCT listing_id) AS n_listings,
        count(DISTINCT lga_code) AS n_lgas,
        CASE
            WHEN count(DISTINCT lga_code) = 1 THEN 'Concentrated (same LGA)'
            ELSE 'Distributed (multiple LGAs)'
        END AS concentration_label
    FROM host_listings
    GROUP BY host_id
    HAVING count(DISTINCT listing_id) > 1
)
SELECT
    count(*) AS multi_listing_hosts,
    count(*) FILTER (WHERE concentration_label = 'Concentrated (same LGA)')
        AS concentrated_same_lga_hosts,
    count(*) FILTER (WHERE concentration_label = 'Distributed (multiple LGAs)')
        AS distributed_across_lgas_hosts,
    round(
        count(*) FILTER (WHERE concentration_label = 'Concentrated (same LGA)')
            * 100.0 / nullif(count(*), 0)
    , 2) AS pct_concentrated_same_lga,
    round(
        count(*) FILTER (WHERE concentration_label = 'Distributed (multiple LGAs)')
            * 100.0 / nullif(count(*), 0)
    , 2) AS pct_distributed_across_lgas
FROM host_summary;


-- -----------------------------------------------------------------------------
-- 4e. Single-listing hosts: does 12-month estimated revenue cover the
--     annualised median mortgage (median_mortgage_repay_monthly * 12)?
--     Rank LGAs by % of hosts that can cover it (coverage_rank).
-- -----------------------------------------------------------------------------
WITH single_hosts AS (
    SELECT host_id
    FROM gold.fact_listing
    WHERE host_id IS NOT NULL
    GROUP BY host_id
    HAVING count(DISTINCT listing_id) = 1
),
host_annual_revenue AS (
    SELECT
        f.host_id,
        f.lga_code,
        sum(f.estimated_revenue::numeric) AS revenue_last_12_months
    FROM gold.fact_listing f
    INNER JOIN single_hosts s ON f.host_id = s.host_id
    WHERE f.has_availability IS TRUE
      AND f.lga_code IS NOT NULL
      AND f.scraped_date >= (
            SELECT max(scraped_date) - INTERVAL '12 months' FROM gold.fact_listing
          )
    GROUP BY f.host_id, f.lga_code
),
coverage AS (
    SELECT
        h.host_id,
        h.lga_code,
        h.revenue_last_12_months,
        (g02.median_mortgage_repay_monthly::numeric * 12) AS annual_median_mortgage,
        CASE
            WHEN h.revenue_last_12_months
                 >= (g02.median_mortgage_repay_monthly::numeric * 12)
                THEN 1 ELSE 0
        END AS covers_mortgage
    FROM host_annual_revenue h
    INNER JOIN gold.ref_census_g02 g02 ON h.lga_code = g02.lga_code
    WHERE nullif(trim(g02.median_mortgage_repay_monthly::text), '') IS NOT NULL
),
lga_summary AS (
    SELECT
        lga_code,
        count(*) AS single_listing_hosts,
        sum(covers_mortgage) AS hosts_covering_mortgage,
        round(sum(covers_mortgage) * 100.0 / nullif(count(*), 0), 2) AS pct_covering
    FROM coverage
    GROUP BY lga_code
    HAVING count(*) >= 10
)
SELECT
    dense_rank() OVER (ORDER BY s.pct_covering DESC, s.single_listing_hosts DESC)
        AS coverage_rank,
    lg.lga_name,
    s.lga_code,
    s.single_listing_hosts,
    s.hosts_covering_mortgage,
    s.pct_covering
FROM lga_summary s
LEFT JOIN gold.dim_lga lg ON s.lga_code = lg.lga_code
ORDER BY coverage_rank
LIMIT 15;
