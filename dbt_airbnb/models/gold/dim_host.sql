{{ config(schema='gold', materialized='table') }}

/*
  SCD2 host dimension from timestamp snapshot.
  Use dbt_valid_from / dbt_valid_to for point-in-time joins from facts/datamarts.
*/

SELECT
    md5(host_id::text || '|' || coalesce(dbt_valid_from::text, '')) AS host_sk,
    host_id,
    host_name,
    host_since,
    host_is_superhost,
    host_neighbourhood,
    dbt_valid_from,
    dbt_valid_to,
    dbt_updated_at
FROM {{ ref('host_snapshot') }}
