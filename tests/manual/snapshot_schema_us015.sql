-- US-015 schema snapshot query (round-trip validation)
SELECT
    table_name,
    column_name,
    data_type,
    udt_name,
    is_nullable,
    column_default,
    ordinal_position
FROM information_schema.columns
WHERE table_name IN ('parcels', 'features_parcels')
ORDER BY table_name, ordinal_position;
