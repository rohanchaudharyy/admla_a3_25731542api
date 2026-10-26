{{ config(schema='gold', materialized='table') }}

/*
  Host neighbourhood is typically a suburb; map to host_neighbourhood_lga via suburb→LGA.
*/

WITH host_nb AS (
    SELECT DISTINCT host_neighbourhood
    FROM {{ ref('listings_clean') }}
    WHERE host_neighbourhood IS NOT NULL
)

SELECT
    md5(lower(h.host_neighbourhood)) AS host_neighbourhood_sk,
    h.host_neighbourhood,
    s.lga_name AS host_neighbourhood_lga,
    s.lga_code AS host_neighbourhood_lga_code
FROM host_nb h
LEFT JOIN {{ ref('lga_suburb_clean') }} s
    ON lower(h.host_neighbourhood) = s.suburb_name_norm
