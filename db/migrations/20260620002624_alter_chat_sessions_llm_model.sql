-- migrate:up
-- US-054: rename chat_sessions.llm_variant -> llm_model and widen the enum-like CHECK
-- from 2 values ('gemini','qwen35') to the 4 hybrid LLM variants the hot-switch supports:
--   gemini       -> cloud (Vertex / AI Studio, google-genai)
--   qwen-api     -> hosted OpenAI-compatible API (Together / Fireworks / OpenRouter)
--   qwen-onprem  -> self-hosted Qwen vLLM on H100/L4 (:8002) -- data sovereignty
--   gemma        -> Google AI Studio or on-prem Ollama (:11434)
--
-- The column is RENAMED (not a second column added) so the name expresses the new domain
-- and no two semantically overlapping columns coexist. The concrete model id
-- (gemini-3.5-flash, served-model qwen35, gemma4:...) is NOT persisted here -- it lives in
-- env vars / settings (routing table), so swapping the underlying model never touches data.
--
-- Data migration (rollforward, no rows lost): existing 'gemini' rows are kept; historic
-- 'qwen35' rows -> 'qwen-onprem' (the historic qwen35 WAS the on-prem vLLM backend). DEFAULT
-- stays 'gemini'.
--
-- The original inline CHECK on the column was named by Postgres convention
-- '<table>_<col>_check' -> 'chat_sessions_llm_variant_check' (verified: the constraint is
-- declared inline in 20260511213942_initial_schema.sql, so Postgres auto-names it that way).
-- DROP CONSTRAINT IF EXISTS makes the drop idempotent regardless of the exact name match.
--
-- RLS note (US-051): this ALTER TABLE runs as the migration role `agrosat` (compose
-- superuser, implicit BYPASSRLS) -> it does not collide with the `tenant_isolation` policy
-- on chat_sessions. The runtime UPDATE (LLMSwitchService) instead runs scoped as
-- `agrosat_app` (NOBYPASSRLS) and is filtered to its own session by the policy.

-- Drop the old 2-value CHECK before renaming the column (drop by its auto-generated name).
ALTER TABLE chat_sessions DROP CONSTRAINT IF EXISTS chat_sessions_llm_variant_check;

-- Rename the column to express the new domain.
ALTER TABLE chat_sessions RENAME COLUMN llm_variant TO llm_model;

-- Drop the old DEFAULT before migrating the data so the UPDATE is unambiguous.
ALTER TABLE chat_sessions ALTER COLUMN llm_model DROP DEFAULT;

-- Migrate historic data: qwen35 was the on-prem vLLM backend -> qwen-onprem. gemini stays.
UPDATE chat_sessions SET llm_model = 'qwen-onprem' WHERE llm_model = 'qwen35';

-- Add the widened 4-value CHECK under its new convention name.
ALTER TABLE chat_sessions
    ADD CONSTRAINT chat_sessions_llm_model_check
    CHECK (llm_model IN ('gemini', 'qwen-api', 'qwen-onprem', 'gemma'));

-- Restore the DEFAULT (unchanged: 'gemini' preserves current behavior).
ALTER TABLE chat_sessions ALTER COLUMN llm_model SET DEFAULT 'gemini';

-- migrate:down
-- Exact reverse: collapse the 4 values back to the historic 2, re-rename the column, and
-- restore the original 2-value CHECK. This is an HONEST (lossy-by-necessity) down:
--   qwen-onprem -> qwen35   (round-trip of the up migration's data step)
--   qwen-api, gemma -> gemini (these values cannot exist under the old CHECK; fold to the
--                              default so no row violates the restored 2-value constraint)

-- Drop the 4-value CHECK and the DEFAULT before remapping data.
ALTER TABLE chat_sessions DROP CONSTRAINT IF EXISTS chat_sessions_llm_model_check;
ALTER TABLE chat_sessions ALTER COLUMN llm_model DROP DEFAULT;

-- Reverse the data migration. qwen-onprem round-trips to qwen35; the two values that have
-- no representation under the old CHECK fold to 'gemini' (documented, lossy down).
UPDATE chat_sessions SET llm_model = 'qwen35' WHERE llm_model = 'qwen-onprem';
UPDATE chat_sessions SET llm_model = 'gemini' WHERE llm_model IN ('qwen-api', 'gemma');

-- Re-rename the column back to its original name.
ALTER TABLE chat_sessions RENAME COLUMN llm_model TO llm_variant;

-- Restore the original 2-value CHECK under its original auto-generated name.
ALTER TABLE chat_sessions
    ADD CONSTRAINT chat_sessions_llm_variant_check
    CHECK (llm_variant IN ('gemini', 'qwen35'));

-- Restore the original DEFAULT.
ALTER TABLE chat_sessions ALTER COLUMN llm_variant SET DEFAULT 'gemini';
