# Dataset layout (upload to Composer GCS bucket)

Mirror this structure under the Composer data bucket (paths match the Airflow DAG):

```
data/
  Census LGA/
    2016Census_G01_NSW_LGA.csv
    2016Census_G02_NSW_LGA.csv
  NSW_LGA/
    NSW_LGA_CODE.csv
    NSW_LGA_SUBURB.csv
  listings/
    05_2020.csv
    06_2020.csv
    …
    04_2021.csv
```

## Included in this repo
- Census G01/G02 (ABS 2016 NSW LGA datapack) — downloaded and checked in under `Census LGA/`.
- `NSW_LGA_CODE.csv` — built from ABS 2016 ASGS LGA geography (suffixes like `(C)` stripped for Airbnb name joins).
- `NSW_LGA_SUBURB.csv` — NSW suburb→LGA pairs derived from open Australian postcodes + LGA codes (stand-in if the course mapping file differs; prefer the Canvas/Moodle mapping when available).

## Not included (blocked / course-provided)
- Monthly Sydney Airbnb curated CSVs (`05_2020.csv` … `04_2021.csv`) from the assignment link / Canvas. Inside Airbnb historical archives return HTTP 403 from this environment. Upload those 12 files before running the DAG.

Expected Airbnb columns (lowercase after DAG normalisation):

`listing_id, scrape_id, scraped_date, host_id, host_name, host_since, host_is_superhost, host_neighbourhood, listing_neighbourhood, property_type, room_type, accommodates, price, has_availability, availability_30, number_of_reviews, review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_value`
