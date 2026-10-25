{{ config(schema='silver', materialized='table') }}

SELECT
    lga_code_2016,
    replace(lga_code_2016, 'LGA', '') AS lga_code,
    median_age_persons,
    median_mortgage_repay_monthly,
    median_tot_prsnl_inc_weekly,
    median_rent_weekly,
    median_tot_fam_inc_weekly,
    average_num_psns_per_bedroom,
    median_tot_hhd_inc_weekly,
    average_household_size
FROM {{ source('bronze', 'census_g02_raw') }}
