{{ config(schema='gold', materialized='table') }}

SELECT *
FROM (
    VALUES
        (md5('excellent'), 'Excellent (95-100)', 95::numeric, 100::numeric),
        (md5('great'),     'Great (90-94)',      90::numeric, 94.999::numeric),
        (md5('good'),      'Good (80-89)',       80::numeric, 89.999::numeric),
        (md5('fair'),      'Fair (70-79)',       70::numeric, 79.999::numeric),
        (md5('poor'),      'Poor (<70)',          0::numeric, 69.999::numeric),
        (md5('unrated'),   'Unrated',            NULL::numeric, NULL::numeric)
) AS t(review_score_band_sk, review_score_band_label, score_min, score_max)
