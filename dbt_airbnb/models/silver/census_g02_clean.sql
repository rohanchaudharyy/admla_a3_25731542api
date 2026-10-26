{{ config(schema='silver', materialized='table') }}

SELECT
    trim(lga_code_2016) AS lga_code_2016,
    replace(trim(lga_code_2016), 'LGA', '') AS lga_code,
    nullif(trim(median_age_persons), '')::numeric AS median_age_persons,
    nullif(trim(median_mortgage_repay_monthly), '')::numeric AS median_mortgage_repay_monthly,
    nullif(trim(median_tot_prsnl_inc_weekly), '')::numeric AS median_tot_prsnl_inc_weekly,
    nullif(trim(median_rent_weekly), '')::numeric AS median_rent_weekly,
    nullif(trim(median_tot_fam_inc_weekly), '')::numeric AS median_tot_fam_inc_weekly,
    nullif(trim(average_num_psns_per_bedroom), '')::numeric AS average_num_psns_per_bedroom,
    nullif(trim(median_tot_hhd_inc_weekly), '')::numeric AS median_tot_hhd_inc_weekly,
    nullif(trim(average_household_size), '')::numeric AS average_household_size
FROM {{ source('bronze', 'raw_census_g02') }}
WHERE nullif(trim(lga_code_2016), '') IS NOT NULL
