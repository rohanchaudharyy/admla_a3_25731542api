"""
Legacy DAG (v1) – row-by-row INSERT loads.

Prefer the Composer-optimised DAG instead:
    dags/airbnb_sydney_elt_dag_v2.py
    dag_id = sydney_airbnb_bronze_elt_v2

See docs/dag-run-smoother.md (also in the submission zip / project store).
This file is kept only for reference; pause it in Composer if both are deployed.
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

BUCKET_NAME = "REPLACE_WITH_COMPOSER_BUCKET"
POSTGRES_CONN_ID = "postgres_bde"
AIRBNB_TABLE = "bronze.listings_raw"

STATIC_FILES = {
    "bronze.census_g01_raw": "data/Census LGA/2016Census_G01_NSW_LGA.csv",
    "bronze.census_g02_raw": "data/Census LGA/2016Census_G02_NSW_LGA.csv",
    "bronze.lga_code_raw": "data/NSW_LGA/NSW_LGA_CODE.csv",
    "bronze.lga_suburb_raw": "data/NSW_LGA/NSW_LGA_SUBURB.csv",
}

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
    "owner": "akshat_dubey",
    "depends_on_past": False,
    "start_date": datetime(2020, 5, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _normalize_date(value):
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
    keep = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
    df = df[keep]
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def truncate_bronze_tables(**_context):
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
    """Legacy path – prefer airbnb_sydney_elt_dag_v2.py (COPY)."""
    df = _download_csv_df(file_key)
    if listings:
        missing = [c for c in LISTINGS_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{file_key} missing columns: {missing}")
        df = df[LISTINGS_COLUMNS]
        for col in ("scraped_date", "host_since"):
            df[col] = df[col].apply(_normalize_date)

    cols = list(df.columns)
    placeholders = ", ".join(["%s"] * len(cols))
    col_sql = ", ".join(cols)
    insert_sql = f"INSERT INTO {table_name} ({col_sql}) VALUES ({placeholders})"

    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = hook.get_conn()
    cur = conn.cursor()
    from psycopg2.extras import execute_batch

    rows = [
        tuple(None if (isinstance(v, float) and pd.isna(v)) or pd.isna(v) else v for v in record)
        for record in df.itertuples(index=False, name=None)
    ]
    execute_batch(cur, insert_sql, rows, page_size=1000)
    conn.commit()
    cur.close()
    conn.close()
    logging.info("Loaded %s rows from %s into %s", len(rows), file_key, table_name)


with DAG(
    dag_id="sydney_airbnb_bronze_elt",
    default_args=default_args,
    description="LEGACY v1 – use sydney_airbnb_bronze_elt_v2 instead",
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=["assignment3", "airbnb", "sydney", "bronze", "legacy"],
) as dag:

    start = DummyOperator(task_id="start")
    truncate = PythonOperator(
        task_id="truncate_bronze_tables",
        python_callable=truncate_bronze_tables,
    )
    static_load_tasks = [
        PythonOperator(
            task_id=f"load_{table.split('.')[-1]}",
            python_callable=load_csv_to_postgres,
            op_kwargs={"file_key": path, "table_name": table, "listings": False},
        )
        for table, path in STATIC_FILES.items()
    ]
    monthly_tasks = [
        PythonOperator(
            task_id=f"load_listings_{month}",
            python_callable=load_csv_to_postgres,
            op_kwargs={
                "file_key": f"data/listings/{month}.csv",
                "table_name": AIRBNB_TABLE,
                "listings": True,
            },
        )
        for month in MONTHLY_FILES
    ]
    remind_dbt = DummyOperator(task_id="manual_dbt_cloud_after_each_month")
    end = DummyOperator(task_id="end")

    start >> truncate >> static_load_tasks
    chain(*static_load_tasks, *monthly_tasks, remind_dbt, end)
