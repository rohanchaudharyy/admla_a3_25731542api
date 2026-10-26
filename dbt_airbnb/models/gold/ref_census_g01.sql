{{ config(schema='gold', materialized='table') }}

/* Census G01 reference – columns available from the Composer DAG load subset. */

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
    age_85_plus
FROM {{ ref('census_g01_clean') }}
