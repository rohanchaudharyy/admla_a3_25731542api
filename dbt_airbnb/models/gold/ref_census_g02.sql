{{ config(schema='gold', materialized='table') }}

/* Census G02 reference data in Gold (medians / averages). */

SELECT
    md5(lga_code) AS census_g02_sk,
    lga_code_2016,
    lga_code,
    median_age_persons::numeric AS median_age_persons,
    median_mortgage_repay_monthly::numeric AS median_mortgage_repay_monthly,
    median_tot_prsnl_inc_weekly::numeric AS median_tot_prsnl_inc_weekly,
    median_rent_weekly::numeric AS median_rent_weekly,
    median_tot_fam_inc_weekly::numeric AS median_tot_fam_inc_weekly,
    average_num_psns_per_bedroom::numeric AS average_num_psns_per_bedroom,
    median_tot_hhd_inc_weekly::numeric AS median_tot_hhd_inc_weekly,
    average_household_size::numeric AS average_household_size
FROM {{ ref('census_g02_clean') }}
