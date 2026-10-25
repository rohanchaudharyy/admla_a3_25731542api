{{ config(schema='gold', materialized='table') }}

SELECT *
FROM (
    VALUES
        (md5('budget'),     'Budget (<$100)',        0::numeric,    99.99::numeric),
        (md5('mid'),        'Mid ($100-$199)',     100::numeric,   199.99::numeric),
        (md5('premium'),    'Premium ($200-$399)', 200::numeric,   399.99::numeric),
        (md5('luxury'),     'Luxury ($400+)',      400::numeric, 999999::numeric),
        (md5('unknown'),    'Unknown',            NULL::numeric, NULL::numeric)
) AS t(price_tier_sk, price_tier_label, price_min, price_max)
