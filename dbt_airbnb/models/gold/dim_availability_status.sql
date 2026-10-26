{{ config(schema='gold', materialized='table') }}

SELECT *
FROM (
    VALUES
        (md5('active'),   TRUE,  'Active'),
        (md5('inactive'), FALSE, 'Inactive'),
        (md5('unknown'),  NULL::boolean, 'Unknown')
) AS t(availability_status_sk, has_availability, availability_status_label)
