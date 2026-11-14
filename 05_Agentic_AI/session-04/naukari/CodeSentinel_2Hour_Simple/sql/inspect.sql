-- Live SQL proof: run after notebook section 3/4/5/6.
\echo '=== Memory rows by category ==='
SELECT prefix, COUNT(*) AS records
FROM store WHERE prefix LIKE 'class-memory.%'
GROUP BY prefix ORDER BY prefix;

\echo '=== Inspect real stored memories ==='
SELECT prefix, key, value->>'kind' AS kind,
       value->>'source_url' AS source_url,
       left(value->>'text', 200) AS preview
FROM store WHERE prefix LIKE 'class-memory.%'
ORDER BY updated_at DESC LIMIT 30;

\echo '=== pgvector extension enabled? ==='
SELECT extname, extversion FROM pg_extension WHERE extname='vector';

\echo '=== Vector storage table exists? ==='
SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename LIKE 'store%';

\echo '=== Embedding row count ==='
SELECT COUNT(*) AS embedding_rows FROM store_vectors;
