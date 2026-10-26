{{ config(schema='gold', materialized='table') }}

SELECT *
FROM (
    VALUES
        (md5('superhost'),     TRUE,  'Superhost'),
        (md5('not_superhost'), FALSE, 'Not Superhost'),
        (md5('unknown'),       NULL::boolean, 'Unknown')
) AS t(superhost_status_sk, host_is_superhost, superhost_status_label)
