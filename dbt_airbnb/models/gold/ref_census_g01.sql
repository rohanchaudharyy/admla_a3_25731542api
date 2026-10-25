{{ config(schema='gold', materialized='table') }}

/* Census G01 reference data in Gold (not SCD – static 2016 snapshot). */

SELECT
    md5(lga_code) AS census_g01_sk,
    lga_code_2016,
    lga_code,
    total_population,
    tot_p_m AS male_population,
    tot_p_f AS female_population,
    age_0_4,
    age_5_14,
    age_15_19,
    age_20_24,
    age_25_34,
    age_35_44,
    age_45_54,
    age_55_64,
    age_65_74,
    age_75_84,
    age_85_plus,
    indigenous_population,
    born_overseas_population,
    born_australia_population,
    australian_citizen_population,
    english_only_home,
    other_language_home
FROM {{ ref('census_g01_clean') }}
