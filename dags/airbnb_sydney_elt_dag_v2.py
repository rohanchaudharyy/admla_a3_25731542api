"""
Sydney Airbnb + Census Bronze ELT – Composer-optimised DAG (v2)

Author / owner: Akshat Dubey

Improvements vs v1:
  - Postgres COPY FROM STDIN (fast bulk load) instead of row-by-row INSERT
  - Airflow Variables control month + truncate without re-uploading the DAG
  - GCS bucket from Variable `airbnb_gcs_bucket` (fallback constant below)

Airflow Variables
  airbnb_gcs_bucket   Composer data bucket name (no gs:// prefix)
  airbnb_load_month   e.g. 05_2020  |  all   (default: all)
  airbnb_truncate     true | false           (default: true)  – first run only

Connection: postgres_bde
schedule_interval=None  – manual trigger; dbt is NOT invoked by this DAG

Use this file on Cloud Composer. Keep or pause the legacy
`sydney_airbnb_bronze_elt` DAG (airbnb_sydney_elt_dag.py).
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.models.baseoperator import chain
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from google.cloud import storage

# ---------------------------------------------------------------------------
# Constants – set BUCKET_FALLBACK once if you prefer not to use the Variable
# ---------------------------------------------------------------------------
BUCKET_FALLBACK = "REPLACE_WITH_COMPOSER_BUCKET"  # e.g. us-central1-bde-xxxx-bucket
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
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(hours=2),
}


# ---------------------------------------------------------------------------
# Variable helpers (read at task runtime – no DAG re-upload needed)
# ---------------------------------------------------------------------------
def _bucket_name() -> str:
    return Variable.get("airbnb_gcs_bucket", default_var=BUCKET_FALLBACK).strip()


def _load_month_setting() -> str:
    return Variable.get("airbnb_load_month", default_var="all").strip()


def _truncate_enabled() -> bool:
    return Variable.get("airbnb_truncate", default_var="true").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _normalize_date_series(series: pd.Series) -> pd.Series:
    """Vectorised-friendly date cleanup for scraped_date / host_since."""
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    # Keep ISO strings; NaT -> None for COPY NULL
    out = parsed.dt.strftime("%Y-%m-%d")
    return out.where(parsed.notna(), None)


def _download_blob_bytes(file_key: str) -> bytes:
    bucket = _bucket_name()
    if not bucket or bucket.startswith("REPLACE_"):
        raise ValueError(
            "Set Airflow Variable airbnb_gcs_bucket (or BUCKET_FALLBACK in the DAG) "
            "to your Composer GCS bucket name."
        )
    client = storage.Client()
    blob = client.bucket(bucket).blob(file_key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{file_key} not found")
    logging.info("Downloading gs://%s/%s", bucket, file_key)
    return blob.download_as_bytes()


def _prepare_dataframe(raw: bytes, listings: bool) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    keep = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
    df = df.loc[:, keep]
    df.columns = [str(c).strip().lower() for c in df.columns]

    if listings:
        missing = [c for c in LISTINGS_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Listings CSV missing columns: {missing}")
        df = df.loc[:, LISTINGS_COLUMNS]
        for col in ("scraped_date", "host_since"):
            df[col] = _normalize_date_series(df[col])
        # Strip $ from price if present
        if df["price"].dtype == object:
            df["price"] = (
                df["price"].astype(str).str.replace(r"[\$,]", "", regex=True).replace({"nan": None, "": None})
            )
    return df


def _copy_dataframe(df: pd.DataFrame, table_name: str) -> int:
    """Bulk-load via COPY FROM STDIN; release the frame promptly after buffering."""
    cols = list(df.columns)
    buf = io.StringIO()
    # na_rep='\\N' matches COPY NULL '\\N'
    df.to_csv(buf, index=False, header=False, na_rep="\\N")
    row_count = len(df)
    del df  # drop reference before network COPY
    buf.seek(0)

    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = hook.get_conn()
    try:
        with conn.cursor() as cur:
            col_sql = ", ".join(cols)
            copy_sql = (
                f"COPY {table_name} ({col_sql}) FROM STDIN WITH "
                f"(FORMAT csv, NULL '\\N')"
            )
            cur.copy_expert(copy_sql, buf)
        conn.commit()
    finally:
        conn.close()
        buf.close()
    return row_count


def truncate_bronze_tables(**_context):
    """TRUNCATE Bronze tables when airbnb_truncate=true; otherwise no-op."""
    if not _truncate_enabled():
        logging.info("airbnb_truncate=false – skipping TRUNCATE")
        return
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = hook.get_conn()
    try:
        with conn.cursor() as cur:
            for table in TABLES_TO_TRUNCATE:
                cur.execute(f"TRUNCATE TABLE {table} CASCADE;")
                logging.info("Truncated %s", table)
        conn.commit()
    finally:
        conn.close()


def load_static_file(file_key: str, table_name: str, **_context):
    """
    Load one Census/LGA CSV with COPY.
    Skipped when truncate is false (later monthly runs) unless Variable
    airbnb_load_static=true forces a reload.
    """
    force_static = Variable.get("airbnb_load_static", default_var="false").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if not _truncate_enabled() and not force_static:
        logging.info(
            "Skipping static load of %s (set airbnb_truncate=true or "
            "airbnb_load_static=true to load)",
            file_key,
        )
        return

    raw = _download_blob_bytes(file_key)
    df = _prepare_dataframe(raw, listings=False)
    del raw
    n = _copy_dataframe(df, table_name)
    logging.info("COPY loaded %s rows into %s from %s", n, table_name, file_key)


def load_listings_month(month: str, **_context):
    """
    Load one Airbnb month with COPY when airbnb_load_month is that month or 'all'.
    Other month tasks exit immediately (no DAG edit required between months).
    """
    selected = _load_month_setting()
    if selected.lower() != "all" and selected != month:
        logging.info(
            "Skipping listings %s.csv (airbnb_load_month=%s)", month, selected
        )
        return
    if selected.lower() != "all" and selected not in MONTHLY_FILES:
        raise ValueError(
            f"airbnb_load_month={selected!r} is not in {MONTHLY_FILES} or 'all'"
        )

    file_key = f"data/listings/{month}.csv"
    raw = _download_blob_bytes(file_key)
    df = _prepare_dataframe(raw, listings=True)
    del raw
    n = _copy_dataframe(df, AIRBNB_TABLE)
    logging.info("COPY loaded %s rows into %s from %s", n, AIRBNB_TABLE, file_key)


with DAG(
    dag_id="sydney_airbnb_bronze_elt_v2",
    default_args=default_args,
    description=(
        "Part 1+3 Bronze load (v2): GCS → Postgres COPY. "
        "Variables: airbnb_gcs_bucket, airbnb_load_month, airbnb_truncate. "
        "dbt Cloud is manual – not triggered here."
    ),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=["assignment3", "airbnb", "sydney", "bronze", "v2", "akshat_dubey"],
) as dag:

    start = DummyOperator(task_id="start")

    truncate = PythonOperator(
        task_id="truncate_bronze_if_enabled",
        python_callable=truncate_bronze_tables,
    )

    static_tasks = []
    for table, path in STATIC_FILES.items():
        short = table.split(".")[-1]
        static_tasks.append(
            PythonOperator(
                task_id=f"load_static_{short}",
                python_callable=load_static_file,
                op_kwargs={"file_key": path, "table_name": table},
            )
        )

    monthly_tasks = [
        PythonOperator(
            task_id=f"load_listings_{month}",
            python_callable=load_listings_month,
            op_kwargs={"month": month},
        )
        for month in MONTHLY_FILES
    ]

    static_complete = DummyOperator(task_id="static_loads_complete")
    remind_dbt = DummyOperator(
        task_id="manual_dbt_cloud_reminder",
    )
    end = DummyOperator(task_id="end")

    # Truncate → static files in parallel → join → months chronological → dbt reminder
    start >> truncate >> static_tasks >> static_complete
    chain(static_complete, *monthly_tasks, remind_dbt, end)
