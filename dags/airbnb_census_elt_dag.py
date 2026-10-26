"""
Airbnb + Census Bronze loader for Cloud Composer.
Author: Akshat Dubey. Submission DAG for Assignment 3.

Upload this file into the Composer bucket's dags/ folder. Remove any other
DAG file defining the same dag_id, including an earlier v2 copy.

Expected bucket files (names and capitals must match):
  data/listings/05_2020.csv ... 04_2021.csv
  data/Census LGA/2016Census_G01_NSW_LGA.csv
  data/Census LGA/2016Census_G02_NSW_LGA.csv
  data/NSW_LGA/NSW_LGA_CODE.csv
  data/NSW_LGA/NSW_LGA_SUBURB.csv

Check PG_CONN_ID and REFERENCE_FILES below before uploading. Census filenames
match the supplied Airflow log; folder names match the bucket screenshots.
Unnamed extra CSV columns are ignored. Duplicate named headers are rejected.

Manual triggers:
  {}                       -> 05_2020.csv
  {"month": "06_2020"}      -> 06_2020.csv
Continue in order through {"month": "04_2021"}. Run dbt separately after
EACH successful monthly load; this DAG does not run dbt automatically.

Existing Bronze table names and columns are retained from the supplied DAG.
CREATE TABLE IF NOT EXISTS does not migrate an incompatible existing schema.
Requires the Airflow PostgreSQL provider and a configured database connection.
"""
import csv
import inspect
import io
import logging
import os
import re
from contextlib import closing
from datetime import datetime, timezone

# Airflow 3 uses the SDK and standard provider; Airflow 2 uses legacy paths.
try:
    from airflow.sdk import DAG
except ImportError:
    from airflow import DAG

try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator

PG_CONN_ID = "postgres"  # Match the exact connection ID in Airflow Connections.
DATA_DIR = "/home/airflow/gcs/data"
DEFAULT_MONTH = "05_2020"
AIRBNB_DIR = os.path.join(DATA_DIR, "listings")
CENSUS_DIR = os.path.join(DATA_DIR, "Census LGA")
MAPPING_DIR = os.path.join(DATA_DIR, "NSW_LGA")

# Bronze stays raw TEXT. Extra CSV columns are ignored, as in the original DAG.
# Missing expected columns now raise a clear error rather than loading NULLs.
LISTING_COLS = [
    "listing_id", "scrape_id", "scraped_date", "host_id", "host_name", "host_since",
    "host_is_superhost", "host_neighbourhood", "listing_neighbourhood", "property_type",
    "room_type", "accommodates", "price", "has_availability", "availability_30",
    "number_of_reviews", "review_scores_rating", "review_scores_accuracy",
    "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication",
    "review_scores_value",
]
G01_COLS = [
    "lga_code_2016", "tot_p_m", "tot_p_f", "tot_p_p", "age_0_4_yr_p", "age_5_14_yr_p",
    "age_15_19_yr_p", "age_20_24_yr_p", "age_25_34_yr_p", "age_35_44_yr_p",
    "age_45_54_yr_p", "age_55_64_yr_p", "age_65_74_yr_p", "age_75_84_yr_p", "age_85ov_p",
]
G02_COLS = [
    "lga_code_2016", "median_age_persons", "median_mortgage_repay_monthly",
    "median_tot_prsnl_inc_weekly", "median_rent_weekly", "median_tot_fam_inc_weekly",
    "average_num_psns_per_bedroom", "median_tot_hhd_inc_weekly", "average_household_size",
]
LGA_CODE_COLS = ["lga_code", "lga_name"]
LGA_SUBURB_COLS = ["lga_name", "suburb_name"]

# Static reference files: (table, file path, columns). Reloaded in full on every run.
REFERENCE_FILES = [
    ("bronze.raw_census_g01", os.path.join(CENSUS_DIR, "2016Census_G01_NSW_LGA.csv"), G01_COLS),
    ("bronze.raw_census_g02", os.path.join(CENSUS_DIR, "2016Census_G02_NSW_LGA.csv"), G02_COLS),
    ("bronze.raw_nsw_lga_code", os.path.join(MAPPING_DIR, "NSW_LGA_CODE.csv"), LGA_CODE_COLS),
    ("bronze.raw_nsw_lga_suburb", os.path.join(MAPPING_DIR, "NSW_LGA_SUBURB.csv"), LGA_SUBURB_COLS),
]

DDL = """
CREATE SCHEMA IF NOT EXISTS bronze;

CREATE TABLE IF NOT EXISTS bronze.raw_listings (
    listing_id TEXT, scrape_id TEXT, scraped_date TEXT, host_id TEXT, host_name TEXT,
    host_since TEXT, host_is_superhost TEXT, host_neighbourhood TEXT,
    listing_neighbourhood TEXT, property_type TEXT, room_type TEXT, accommodates TEXT,
    price TEXT, has_availability TEXT, availability_30 TEXT, number_of_reviews TEXT,
    review_scores_rating TEXT, review_scores_accuracy TEXT, review_scores_cleanliness TEXT,
    review_scores_checkin TEXT, review_scores_communication TEXT, review_scores_value TEXT,
    source_file TEXT, loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS bronze.raw_census_g01 (
    lga_code_2016 TEXT, tot_p_m TEXT, tot_p_f TEXT, tot_p_p TEXT, age_0_4_yr_p TEXT,
    age_5_14_yr_p TEXT, age_15_19_yr_p TEXT, age_20_24_yr_p TEXT, age_25_34_yr_p TEXT,
    age_35_44_yr_p TEXT, age_45_54_yr_p TEXT, age_55_64_yr_p TEXT, age_65_74_yr_p TEXT,
    age_75_84_yr_p TEXT, age_85ov_p TEXT, loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS bronze.raw_census_g02 (
    lga_code_2016 TEXT, median_age_persons TEXT, median_mortgage_repay_monthly TEXT,
    median_tot_prsnl_inc_weekly TEXT, median_rent_weekly TEXT, median_tot_fam_inc_weekly TEXT,
    average_num_psns_per_bedroom TEXT, median_tot_hhd_inc_weekly TEXT,
    average_household_size TEXT, loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS bronze.raw_nsw_lga_code (
    lga_code TEXT, lga_name TEXT, loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS bronze.raw_nsw_lga_suburb (
    lga_name TEXT, suburb_name TEXT, loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

log = logging.getLogger(__name__)

# Explicit alternatives for commonly used listing column names.
HEADER_ALIASES = {
    "listing_id": ("listing_id", "id"),
    "scraped_date": ("scraped_date", "scrapped_date", "last_scraped"),
}


def _month_file(context):
    dag_run = context.get("dag_run")
    conf = (dag_run.conf or {}) if dag_run else {}
    month = conf.get("month", DEFAULT_MONTH)
    if not isinstance(month, str) or not re.fullmatch(r"(0[1-9]|1[0-2])_[0-9]{4}", month):
        raise ValueError('Trigger configuration must use {"month": "06_2020"}.')
    selected = datetime.strptime(month, "%m_%Y")
    if not datetime(2020, 5, 1) <= selected <= datetime(2021, 4, 1):
        raise ValueError("Choose an assignment month between 05_2020 and 04_2021.")
    file_name = month + ".csv"
    return file_name, os.path.join(AIRBNB_DIR, file_name)


def _check_file(path):
    if not os.path.isfile(path):
        folder = os.path.dirname(path)
        available = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
        raise FileNotFoundError(
            "Expected CSV not found: {}. Files in its folder: {}. "
            "Check the exact filename, capitals and bucket data/ folder.".format(path, available)
        )


def _reader(fh, path, wanted_cols):
    reader = csv.DictReader(fh)
    if not reader.fieldnames:
        raise ValueError("CSV is empty or has no header: " + path)
    headers = [name.strip().lower() for name in reader.fieldnames]
    named_headers = [name for name in headers if name]
    duplicates = sorted({name for name in named_headers if named_headers.count(name) > 1})
    if duplicates:
        raise ValueError("CSV contains duplicate named columns {}: {}".format(duplicates, path))
    # Spreadsheet exports can contain repeated blank headers for unused columns.
    # Give each an internal name so DictReader retains field positions and widths.
    # Use names that cannot collide with any real source header.
    used_names = set(named_headers)
    unnamed_count = 0
    for index, name in enumerate(headers):
        if not name:
            placeholder = "__unnamed_column_{}".format(index + 1)
            while placeholder in used_names:
                placeholder += "_"
            headers[index] = placeholder
            used_names.add(placeholder)
            unnamed_count += 1
    if unnamed_count:
        log.info("Ignoring %s unnamed extra columns in %s", unnamed_count, path)
    reader.fieldnames = headers
    selected = [
        next((name for name in HEADER_ALIASES.get(col, (col,)) if name in headers), None)
        for col in wanted_cols
    ]
    missing = [col for col, name in zip(wanted_cols, selected) if name is None]
    if missing:
        raise ValueError(
            "CSV {} is missing expected columns {}. Actual headers: {}. "
            "Check that the correct source CSV was uploaded.".format(path, missing, headers)
        )
    return reader, selected


def _read_csv(path, wanted_cols, extra_values=None):
    _check_file(path)
    out = io.StringIO()
    try:
        writer = csv.writer(out)
        n_rows = 0
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader, selected = _reader(fh, path, wanted_cols)
            for row in reader:
                if None in row or any(row[name] is None for name in reader.fieldnames):
                    raise ValueError(
                        "CSV row has the wrong number of fields at line {} in {}".format(
                            reader.line_num, path
                        )
                    )
                writer.writerow([row[name] for name in selected] + list(extra_values or []))
                n_rows += 1
        if n_rows == 0:
            raise ValueError("CSV contains no data rows; existing data was retained: " + path)
        out.seek(0)
        return out, n_rows
    except Exception:
        out.close()
        raise


def _postgres_hook():
    # Import and connect only while executing a task, never while parsing the DAG.
    try:
        from airflow.providers.postgres.hooks.postgres import PostgresHook
    except ImportError as exc:
        raise RuntimeError(
            "Install a compatible apache-airflow-providers-postgres package in Composer."
        ) from exc
    return PostgresHook(postgres_conn_id=PG_CONN_ID)


def _copy(cur, table, cols, buffer):
    # Identifiers are fixed constants in this file, never values from trigger config.
    sql = "COPY {} ({}) FROM STDIN WITH (FORMAT CSV, NULL '')".format(table, ", ".join(cols))
    if hasattr(cur, "copy_expert"):
        cur.copy_expert(sql, buffer)  # psycopg2
    else:
        with cur.copy(sql) as copy:  # psycopg3
            while True:
                chunk = buffer.read(65536)
                if not chunk:
                    break
                copy.write(chunk)


def validate_source_files(**context):
    _, monthly_path = _month_file(context)
    sources = [(path, cols) for _, path, cols in REFERENCE_FILES]
    sources.append((monthly_path, LISTING_COLS))
    for path, cols in sources:
        _check_file(path)
        with open(path, newline="", encoding="utf-8-sig") as fh:
            _reader(fh, path, cols)
        log.info("Found source file with expected headers: %s", path)


def create_bronze_tables():
    _postgres_hook().run(DDL)


def load_reference_data():
    # One transaction covers all four reference tables. A failure rolls it back.
    with closing(_postgres_hook().get_conn()) as conn:
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                for table, path, cols in REFERENCE_FILES:
                    buffer, n_rows = _read_csv(path, cols)
                    with closing(buffer):
                        cur.execute("TRUNCATE TABLE " + table)
                        _copy(cur, table, cols, buffer)
                    log.info("Prepared %s reference rows for %s", n_rows, table)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    log.info("All four reference tables committed successfully.")


def load_airbnb_month(**context):
    file_name, path = _month_file(context)
    buffer, n_rows = _read_csv(path, LISTING_COLS, extra_values=[file_name])
    with closing(buffer), closing(_postgres_hook().get_conn()) as conn:
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM bronze.raw_listings WHERE source_file = %s",
                    (file_name,),
                )
                _copy(cur, "bronze.raw_listings", LISTING_COLS + ["source_file"], buffer)
                cur.execute(
                    "SELECT COUNT(*) FROM bronze.raw_listings WHERE source_file = %s",
                    (file_name,),
                )
                actual_rows = cur.fetchone()[0]
                if actual_rows != n_rows:
                    raise ValueError(
                        "Row-count mismatch for {}: expected {}, found {}".format(
                            file_name, n_rows, actual_rows
                        )
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    log.info("Committed %s listing rows from %s", n_rows, file_name)


# schedule is used in Airflow 2.4+ and 3; older Airflow 2 uses schedule_interval.
schedule_args = (
    {"schedule": None}
    if "schedule" in inspect.signature(DAG.__init__).parameters
    else {"schedule_interval": None}
)

with DAG(
    dag_id="airbnb_census_elt_bronze_load",
    description="Monthly Airbnb and Census/mapping CSVs to Postgres Bronze",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "akshat_dubey", "retries": 0},
    tags=["bde", "assignment3", "bronze"],
    **schedule_args
) as dag:
    t_validate = PythonOperator(
        task_id="validate_source_files", python_callable=validate_source_files
    )
    t_create = PythonOperator(
        task_id="create_bronze_tables", python_callable=create_bronze_tables
    )
    t_reference = PythonOperator(
        task_id="load_census_and_mapping", python_callable=load_reference_data
    )
    t_airbnb = PythonOperator(
        task_id="load_airbnb_month", python_callable=load_airbnb_month
    )
    t_validate >> t_create >> t_reference >> t_airbnb
