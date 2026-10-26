{{ config(schema='silver', materialized='table') }}

/*
  G01 subset loaded by airbnb_census_elt_bronze_load (see G01_COLS in the DAG):
  totals by sex + age-band person counts only.
*/

SELECT
    trim(lga_code_2016) AS lga_code_2016,
    replace(trim(lga_code_2016), 'LGA', '') AS lga_code,
    nullif(trim(tot_p_m), '')::integer AS tot_p_m,
    nullif(trim(tot_p_f), '')::integer AS tot_p_f,
    nullif(trim(tot_p_p), '')::integer AS total_population,
    nullif(trim(age_0_4_yr_p), '')::integer AS age_0_4,
    nullif(trim(age_5_14_yr_p), '')::integer AS age_5_14,
    nullif(trim(age_15_19_yr_p), '')::integer AS age_15_19,
    nullif(trim(age_20_24_yr_p), '')::integer AS age_20_24,
    nullif(trim(age_25_34_yr_p), '')::integer AS age_25_34,
    nullif(trim(age_35_44_yr_p), '')::integer AS age_35_44,
    nullif(trim(age_45_54_yr_p), '')::integer AS age_45_54,
    nullif(trim(age_55_64_yr_p), '')::integer AS age_55_64,
    nullif(trim(age_65_74_yr_p), '')::integer AS age_65_74,
    nullif(trim(age_75_84_yr_p), '')::integer AS age_75_84,
    nullif(trim(age_85ov_p), '')::integer AS age_85_plus
FROM {{ source('bronze', 'raw_census_g01') }}
WHERE nullif(trim(lga_code_2016), '') IS NOT NULL
