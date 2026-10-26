{% snapshot listing_snapshot %}
{{
  config(
    target_schema='silver',
    unique_key='listing_id',
    strategy='timestamp',
    updated_at='updated_at',
    invalidate_hard_deletes=True
  )
}}

SELECT
    listing_id,
    host_id,
    listing_neighbourhood,
    property_type,
    room_type,
    accommodates,
    updated_at
FROM {{ ref('listing_entity') }}

{% endsnapshot %}
