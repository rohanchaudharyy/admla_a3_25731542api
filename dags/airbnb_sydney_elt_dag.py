"""
DEPRECATED – do not deploy on Composer for Assignment 3.

Use the working submission DAG instead:
    dags/airbnb_census_elt_dag.py
    dag_id = airbnb_census_elt_bronze_load
    connection = postgres
    tables = bronze.raw_listings / raw_census_g01 / raw_census_g02 /
             raw_nsw_lga_code / raw_nsw_lga_suburb

This file is retained only as a historical stub so an accidental upload is
obvious. It defines no DAG.
"""
