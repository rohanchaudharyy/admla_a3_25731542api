{{ config(schema='gold', materialized='table') }}

/*
  listing_neighbourhood in the curated Sydney files aligns with LGA display names.
  Enrich with LGA code where possible.
*/

WITH neighbourhoods AS (
    SELECT DISTINCT listing_neighbourhood
    FROM {{ ref('listings_clean') }}
    WHERE listing_neighbourhood IS NOT NULL
)

SELECT
    md5(lower(n.listing_neighbourhood)) AS listing_neighbourhood_sk,
    n.listing_neighbourhood,
    c.lga_code,
    c.lga_name
FROM neighbourhoods n
LEFT JOIN {{ ref('lga_code_clean') }} c
    ON lower(n.listing_neighbourhood) = c.lga_name_norm
