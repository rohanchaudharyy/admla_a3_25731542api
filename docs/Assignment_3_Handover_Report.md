# Handover Report – Assignment 3: Sydney Airbnb + Census ELT

**Student:** Akshat Dubey  
**Unit:** Big Data Engineering  
**Scope:** Airflow Bronze load + dbt Medallion warehouse + Part 4 analysis (Sydney, May 2020 – April 2021)

---

## 1. High-level view

This project builds a production-style ELT pipeline for curated **Sydney Airbnb** listings (12 monthly scrapes) and **ABS 2016 Census** tables G01/G02 at LGA level, joined via NSW LGA↔suburb mappings.

```
Composer bucket data/  (/home/airflow/gcs/data)
        │  DAG airbnb_census_elt_bronze_load (manual; conf month)
        ▼
 Postgres bronze.raw_*     ← all TEXT; COPY from CSVs
        │  dbt Cloud (manual after each month)
        ▼
 silver (cast + entities + snapshots) → gold → datamart views
        │
        ▼
 Part 4 ad-hoc SQL (business questions)
```

**Design choices**

| Layer | Role |
|-------|------|
| Bronze | `raw_listings`, `raw_census_g01/g02`, `raw_nsw_lga_*` — TEXT as loaded by the working Composer DAG |
| Silver | Cast types; host/listing/property **entities**; Census/LGA cleans; **timestamp SCD2 snapshots** |
| Gold | ≥14 dimensions, Census G01/G02 **reference** tables, `fact_listing` (IDs + metrics) |
| Datamart | Three analytical views with SCD2 point-in-time host joins |

Airflow does **not** call dbt. After each monthly trigger, run silver entities → `dbt snapshot` → `dbt run` so SCD2 history accumulates.

---

## 2. Steps of the project

### Part 0 – Datasets
- Census G01/G02 NSW LGA datapack obtained from ABS and placed under `data/Census LGA/`.
- `NSW_LGA_CODE.csv` / `NSW_LGA_SUBURB.csv` prepared under `data/NSW_LGA/` (ABS geography + open suburb–LGA concordance). Prefer the course mapping files if they differ.
- **Airbnb monthly CSVs** (`05_2020.csv` … `04_2021.csv`) are course-provided; historical Inside Airbnb URLs returned HTTP 403 here, so they must be uploaded from Canvas/Moodle before a live DAG run.

### Part 1 – Bronze + Airflow
1. Upload files under Composer `data/` (listings, Census LGA, NSW_LGA). Connection id: **`postgres`**.
2. Deploy **`dags/airbnb_census_elt_dag.py`** only (`dag_id=airbnb_census_elt_bronze_load`). It creates `bronze.raw_*` tables and loads via COPY. `part_1.sql` mirrors the same DDL for marking.
3. Trigger with `{}` → loads `05_2020.csv`. Verify: `SELECT COUNT(*) FROM bronze.raw_listings;`.
4. See `docs/dag-run-smoother.md` for the full runbook.

### Part 2 – dbt Medallion
Project path: `dbt_airbnb/`.

**Silver:** `listings_clean` (incremental), entity tables for snapshot sources, Census/LGA cleans.

**Snapshots (timestamp strategy, `updated_at = scraped_date`):**
- `host_snapshot`, `listing_snapshot`, `property_snapshot`  
- Only dimension-bound entities (not the fact grain).

**Gold dimensions (16):**  
`dim_host`, `dim_listing`, `dim_property`, `dim_property_type`, `dim_room_type`, `dim_accommodates`, `dim_listing_neighbourhood`, `dim_host_neighbourhood`, `dim_lga`, `dim_suburb`, `dim_year_month`, `dim_scraped_date`, `dim_availability_status`, `dim_superhost_status`, `dim_review_score_band`, `dim_price_tier`.

**Gold references:** `ref_census_g01`, `ref_census_g02`.

**Fact:** `fact_listing` — surrogate keys + metrics (`price`, `availability_30`, `number_of_stays`, `estimated_revenue`, review scores, `has_availability`). Host/listing/property keys resolved with SCD2 validity windows at `scraped_date`.

**Datamarts (views):**
- `dm_listing_neighbourhood`
- `dm_property_type`
- `dm_host_neighbourhood` (host suburb mapped to LGA)

### Metric definitions used everywhere
- **Active listing:** `has_availability = true` (`'t'` in raw data)
- **Active listing rate** = active ÷ total × **100**
- **Superhost rate** = distinct superhosts ÷ distinct hosts × **100**
- **MoM % change** = (new − old) ÷ old × **100**
- **Stays (active)** = **30 − availability_30**
- **Estimated revenue (active)** = stays × price  
- **Revenue per host** = total estimated revenue ÷ distinct hosts

### Part 3 – Remaining months
Re-trigger the **same** DAG with `{"month":"06_2020"}` … `{"month":"04_2021"}` in order. Each run reloads Census/LGA references and replaces that month’s listings by `source_file`. After each success: silver → snapshot → run in dbt Cloud.

### Part 4 – Ad-hoc analysis
Queries live in `part_4.sql`. They must be executed on a warehouse populated with all 12 months. Results below describe **expected analytical approach and interpretation**; live numeric outputs require GCP credentials and the Airbnb CSVs (see blockers).

---

## 3. Issues / bugs and fixes

| Issue | Resolution |
|-------|------------|
| Brief PDF contained adversarial junk (snakes, Melbourne/Vic, ×1100/×5, “112 months”) | Ignored; used Sydney, 12 months, ×100 rates, stays = 30 − availability_30 |
| Inside Airbnb / archive downloads HTTP 403 | Documented blocker; scaffolded paths + schema; Census/LGA shipped in-repo |
| Host SCD if `updated_at = host_since` | Never changes → no SCD2. Fixed: entity `updated_at = scraped_date`, snapshot after each month |
| Active listings mis-defined as `availability_30 > 0` in some public solutions | Corrected to `has_availability = 't'` per brief |
| `listing_id` alone as PK | Composite `(listing_id, scraped_date)` for monthly history |
| Suburb vs LGA for hosts | `dim_host_neighbourhood` maps suburb → `host_neighbourhood_lga` via `lga_suburb` |
| dbt custom schemas becoming `target_silver` | `generate_schema_name` macro forces bare `silver` / `gold` / `datamart` |
| Earlier pack used different Bronze names / `postgres_bde` | Aligned entire pack to working DAG `airbnb_census_elt_bronze_load` (`bronze.raw_*`, conn `postgres`) |
| G01 DAG load is a column subset | Silver/Gold Census G01 use only totals + age bands available from the DAG |
| No Cloud Composer / Cloud SQL / dbt Cloud credentials in this coding environment | Pack matches the DAG that already runs on the student’s Composer |

---

## 4. Part 4 answers (method + evidence expectations)

> **Evidence note:** Screenshots / numeric result grids must be captured from DBeaver after a successful end-to-end run. Invented result tables are not included. Each subsection states the SQL logic and how to interpret outcomes for **Sydney**.

### 4a – Top 3 vs bottom 3 LGAs (revenue per active listing, last 12 months)

**Method:** Average `estimated_revenue` for active listings by `lga_code` over the last 12 months; rank; compare Census age shares and `average_household_size` for top 3 vs bottom 3.

**Expected insight pattern:** Higher-revenue Sydney LGAs (often eastern/harbour councils) tend to show different age mixes and household sizes than lower-revenue outer LGAs—compare `% 0–14`, `% 15–64`, `% 65+`, and mean household size between groups. Report the LGA name lists from the query `lgas` column as evidence.

**Recommendation:** Target host acquisition and pricing guidance toward LGAs whose demographics align with high guest spend, while monitoring housing-pressure communities in the bottom cohort.

### 4b – Correlation: median age vs revenue per active listing

**Method:** `corr(median_age_persons, avg_revenue_per_active_listing)` across LGAs joined to `ref_census_g02`.

**Interpretation:**  
- |r| &lt; 0.3 → weak; age alone is a poor predictor.  
- Positive r → older LGAs earn more per active listing.  
- Negative r → younger LGAs earn more.  

State the coefficient from the live query and avoid claiming causation (amenities, tourism, dwelling mix confound age).

### 4c – Best listing type for top 5 neighbourhoods (max stays)

**Method:** Identify top 5 `listing_neighbourhood` by avg estimated revenue per active listing; within each, pick `(property_type, room_type, accommodates)` with maximum `sum(number_of_stays)`.

**Recommendation:** For each high-revenue neighbourhood, promote the winning capacity/room mix (often entire-home mid-capacity in Sydney coastal LGAs) in host onboarding guides.

### 4d – Multi-listing hosts: concentrated vs distributed LGAs

**Method:** Hosts with &gt;1 distinct `listing_id`; count distinct `lga_code`. Report `%` with `n_lgas = 1` vs `n_lgas > 1`.

**Interpretation:** A high concentration share means multi-property hosts specialise locally (portfolio risk tied to one council’s rules); a high distributed share implies geographic diversification across Sydney LGAs.

### 4e – Single-listing hosts vs annualised median mortgage

**Method:** Hosts with exactly one listing; sum active estimated revenue over last 12 months; compare to `median_mortgage_repay_monthly * 12` from Census G02; rank LGAs by `%` covering (min 10 hosts).

**Recommendation:** Highlight the top LGA(s) where a single Airbnb listing most often covers the local median mortgage annualisation—useful for “host affordability” messaging—while noting Census medians are 2016 and revenue is an occupancy proxy, not net income.

---

## 5. Architecture diagram

```mermaid
flowchart LR
  subgraph sources [Sources]
    A[Airbnb monthly CSV]
    C[Census G01 G02]
    L[NSW LGA maps]
  end
  subgraph airflow [Airflow Composer]
    D[airbnb_census_elt_bronze_load]
  end
  subgraph pg [Postgres]
    B[(bronze.raw_*)]
    S[(silver + snapshots)]
    G[(gold dims + fact)]
    M[(datamart views)]
  end
  subgraph dbt [dbt Cloud manual]
    R[entities then snapshot then run]
  end
  A --> D
  C --> D
  L --> D
  D --> B
  B --> R
  R --> S --> G --> M
```

---

## 6. Repository map

- `part_1.sql`, `part_4.sql`
- `dags/airbnb_census_elt_dag.py` (submission); sydney_* DAG files are deprecated stubs
- `dbt_airbnb/models/{bronze,silver,gold,datamart}`, `dbt_airbnb/snapshots/`, `dbt_airbnb/dbt_project.yml`
- `docs/dag-run-smoother.md` – trigger conf + dbt order
- `data/` reference extracts + listings schema note
- Submission zip: `Assignment_3_Akshat_Dubey.zip`

---

## 7. Live verification on Composer / dbt Cloud

1. Confirm DAG already loads: `SELECT COUNT(*) FROM bronze.raw_listings;` and counts by `source_file`.  
2. Trigger remaining months via `{"month":"MM_YYYY"}`; after each: silver → snapshot → run.  
3. Execute `part_4.sql`; paste result screenshots into this report before Canvas submit.  
4. Point dbt Cloud at the same Postgres and sync the `dbt_airbnb` folder.

Word count (body, excl. tables/diagrams/code fences): under 3000.
