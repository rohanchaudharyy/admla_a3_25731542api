"""
Combined Airflow DAG for Parts 1 and 3 – Sydney Airbnb + Census ELT (Bronze load).

- schedule_interval=None  (manual trigger only)
- Loads Census G01/G02 + NSW LGA mappings once, then Airbnb months chronologically
- Does NOT trigger dbt Cloud (run dbt manually after each monthly load per Part 3)

Upload to Cloud Composer DAGs folder. Configure:
  - Airflow connection `postgres_bde` → Cloud SQL Postgres
  - Variable / const BUCKET_NAME → Composer GCS data bucket
  - Place files under data/ in that bucket (see STATIC_FILES / MONTHLY_FILES)
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.models.baseoperator import chain
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from google.cloud import storage

# ---------------------------------------------------------------------------
# Configuration – update BUCKET_NAME / POSTGRES_CONN_ID for your GCP project
# ---------------------------------------------------------------------------
BUCKET_NAME = "REPLACE_WITH_COMPOSER_BUCKET"  # e.g. us-central1-bde-xxxx-bucket
POSTGRES_CONN_ID = "postgres_bde"
AIRBNB_TABLE = "bronze.listings_raw"

# Static reference files (Part 1 initial load)
STATIC_FILES = {
    "bronze.census_g01_raw": "data/Census LGA/2016Census_G01_NSW_LGA.csv",
    "bronze.census_g02_raw": "data/Census LGA/2016Census_G02_NSW_LGA.csv",
    "bronze.lga_code_raw": "data/NSW_LGA/NSW_LGA_CODE.csv",
    "bronze.lga_suburb_raw": "data/NSW_LGA/NSW_LGA_SUBURB.csv",
}

# 12 months May 2020 – April 2021 (chronological)
MONTHLY_FILES = [
    "05_2020",
    "06_2020",
    "07_2020",
    "08_2020",
    "09_2020",
    "10_2020",
    "11_2020",
    "12_2020",
    "01_2021",
    "02_2021",
    "03_2021",
    "04_2021",
]

TABLES_TO_TRUNCATE = [
    "bronze.listings_raw",
    "bronze.census_g01_raw",
    "bronze.census_g02_raw",
    "bronze.lga_code_raw",
    "bronze.lga_suburb_raw",
]

# Canonical column order for curated Airbnb CSVs
LISTINGS_COLUMNS = [
    "listing_id",
    "scrape_id",
    "scraped_date",
    "host_id",
    "host_name",
    "host_since",
    "host_is_superhost",
    "host_neighbourhood",
    "listing_neighbourhood",
    "property_type",
    "room_type",
    "accommodates",
    "price",
    "has_availability",
    "availability_30",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_value",
]

default_args = {
    "owner": "rohan_chaudhary",
    "depends_on_past": False,
    "start_date": datetime(2020, 5, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _normalize_date(value):
    """Accept YYYY-MM-DD or DD/MM/YYYY strings from curated CSVs."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return value


def _download_csv_df(file_key: str) -> pd.DataFrame:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{BUCKET_NAME}/{file_key} not found")
    content = blob.download_as_text()
    df = pd.read_csv(io.StringIO(content))
    # Drop junk Unnamed columns sometimes present in Excel exports
    keep = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
    df = df[keep]
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def truncate_bronze_tables(**_context):
    """Empty Bronze tables before a full reload (Part 1 advice)."""
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = hook.get_conn()
    cur = conn.cursor()
    for table in TABLES_TO_TRUNCATE:
        cur.execute(f"TRUNCATE TABLE {table} CASCADE;")
        logging.info("Truncated %s", table)
    conn.commit()
    cur.close()
    conn.close()


def load_csv_to_postgres(file_key: str, table_name: str, listings: bool = False, **_context):
    """Read one CSV from GCS and INSERT into the target Bronze table."""
    df = _download_csv_df(file_key)

    if listings:
        missing = [c for c in LISTINGS_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{file_key} missing columns: {missing}")
        df = df[LISTINGS_COLUMNS]
        for col in ("scraped_date", "host_since"):
            df[col] = df[col].apply(_normalize_date)

    # Census / LGA CSVs may arrive with original ABS casing – already lowercased
    cols = list(df.columns)
    placeholders = ", ".join(["%s"] * len(cols))
    col_sql = ", ".join(cols)
    insert_sql = f"INSERT INTO {table_name} ({col_sql}) VALUES ({placeholders})"

    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = hook.get_conn()
    cur = conn.cursor()

    rows = []
    for record in df.itertuples(index=False, name=None):
        clean = tuple(None if (isinstance(v, float) and pd.isna(v)) or pd.isna(v) else v for v in record)
        rows.append(clean)

    from psycopg2.extras import execute_batch

    execute_batch(cur, insert_sql, rows, page_size=1000)
    conn.commit()
    cur.close()
    conn.close()
    logging.info("Loaded %s rows from %s into %s", len(rows), file_key, table_name)


with DAG(
    dag_id="sydney_airbnb_bronze_elt",
    default_args=default_args,
    description=(
        "Part 1+3: load Census/LGA + Sydney Airbnb months into Bronze Postgres. "
        "dbt Cloud is triggered manually after each month."
    ),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=["assignment3", "airbnb", "sydney", "bronze"],
) as dag:

    start = DummyOperator(task_id="start")

    truncate = PythonOperator(
        task_id="truncate_bronze_tables",
        python_callable=truncate_bronze_tables,
    )

    load_static = []
    for table, path in STATIC_FILES.items():
        short = table.split(".")[-1]
        task = PythonOperator(
            task_id=f"load_{short}",
            python_callable=load_csv_to_postgres,
            op_kwargs={"file_key": path, "table_name": table, "listings": False},
        )
        load_static.append(task)

    # Sequential monthly loads so Part 3 chronological integrity is preserved
    monthly_tasks = []
    for month in MONTHLY_FILES:
        task = PythonOperator(
            task_id=f"load_listings_{month}",
            python_callable=load_csv_to_postgres,
            op_kwargs={
                "file_key": f"data/listings/{month}.csv",
                "table_name": AIRBNB_TABLE,
                "listings": True,
            },
        )
        monthly_tasks.append(task)

    # Reminder node only – dbt is NOT invoked by Airflow
    remind_dbt = DummyOperator(
        task_id="manual_dbt_cloud_after_each_month",
    )

    end = DummyOperator(task_id="end")

    start >> truncate >> load_static
    chain(*load_static, *monthly_tasks, remind_dbt, end)
