{{ config(schema='silver', materialized='table') }}

SELECT
    lga_code_2016,
    replace(lga_code_2016, 'LGA', '') AS lga_code,
    tot_p_m,
    tot_p_f,
    tot_p_p AS total_population,
    age_0_4_yr_p AS age_0_4,
    age_5_14_yr_p AS age_5_14,
    age_15_19_yr_p AS age_15_19,
    age_20_24_yr_p AS age_20_24,
    age_25_34_yr_p AS age_25_34,
    age_35_44_yr_p AS age_35_44,
    age_45_54_yr_p AS age_45_54,
    age_55_64_yr_p AS age_55_64,
    age_65_74_yr_p AS age_65_74,
    age_75_84_yr_p AS age_75_84,
    age_85ov_p AS age_85_plus,
    indigenous_p_tot_p AS indigenous_population,
    birthplace_elsewhere_p AS born_overseas_population,
    birthplace_australia_p AS born_australia_population,
    australian_citizen_p AS australian_citizen_population,
    lang_spoken_home_eng_only_p AS english_only_home,
    lang_spoken_home_oth_lang_p AS other_language_home
FROM {{ source('bronze', 'census_g01_raw') }}
