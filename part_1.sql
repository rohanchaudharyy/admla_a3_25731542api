-- =============================================================================
-- Part 1: Bronze DDL aligned with DAG airbnb_census_elt_bronze_load
-- Student: Akshat Dubey
--
-- The working Composer DAG (dags/airbnb_census_elt_dag.py) runs the same DDL
-- via task create_bronze_tables (CREATE TABLE IF NOT EXISTS). This file mirrors
-- that DDL for marking / DBeaver review. Prefer letting the DAG create tables;
-- run this only if you need an explicit one-shot setup before the first trigger.
--
-- Connection id used by the DAG: postgres
-- Data root on Composer: /home/airflow/gcs/data/
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS bronze;

-- All Bronze columns are TEXT (raw). Casting happens in dbt Silver.
CREATE TABLE IF NOT EXISTS bronze.raw_listings (
    listing_id TEXT,
    scrape_id TEXT,
    scraped_date TEXT,
    host_id TEXT,
    host_name TEXT,
    host_since TEXT,
    host_is_superhost TEXT,
    host_neighbourhood TEXT,
    listing_neighbourhood TEXT,
    property_type TEXT,
    room_type TEXT,
    accommodates TEXT,
    price TEXT,
    has_availability TEXT,
    availability_30 TEXT,
    number_of_reviews TEXT,
    review_scores_rating TEXT,
    review_scores_accuracy TEXT,
    review_scores_cleanliness TEXT,
    review_scores_checkin TEXT,
    review_scores_communication TEXT,
    review_scores_value TEXT,
    source_file TEXT,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bronze.raw_census_g01 (
    lga_code_2016 TEXT,
    tot_p_m TEXT,
    tot_p_f TEXT,
    tot_p_p TEXT,
    age_0_4_yr_p TEXT,
    age_5_14_yr_p TEXT,
    age_15_19_yr_p TEXT,
    age_20_24_yr_p TEXT,
    age_25_34_yr_p TEXT,
    age_35_44_yr_p TEXT,
    age_45_54_yr_p TEXT,
    age_55_64_yr_p TEXT,
    age_65_74_yr_p TEXT,
    age_75_84_yr_p TEXT,
    age_85ov_p TEXT,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bronze.raw_census_g02 (
    lga_code_2016 TEXT,
    median_age_persons TEXT,
    median_mortgage_repay_monthly TEXT,
    median_tot_prsnl_inc_weekly TEXT,
    median_rent_weekly TEXT,
    median_tot_fam_inc_weekly TEXT,
    average_num_psns_per_bedroom TEXT,
    median_tot_hhd_inc_weekly TEXT,
    average_household_size TEXT,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bronze.raw_nsw_lga_code (
    lga_code TEXT,
    lga_name TEXT,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bronze.raw_nsw_lga_suburb (
    lga_name TEXT,
    suburb_name TEXT,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- Post-load checks (run in DBeaver after a successful DAG trigger)
-- -----------------------------------------------------------------------------
-- SELECT COUNT(*) FROM bronze.raw_listings;
-- SELECT source_file, COUNT(*) FROM bronze.raw_listings GROUP BY 1 ORDER BY 1;
-- SELECT COUNT(*) FROM bronze.raw_census_g01;
-- SELECT COUNT(*) FROM bronze.raw_census_g02;
-- SELECT COUNT(*) FROM bronze.raw_nsw_lga_code;
-- SELECT COUNT(*) FROM bronze.raw_nsw_lga_suburb;
