-- migrate:up
-- E12: widen chat_sessions.llm_model CHECK to add the on-prem multimodal variant
-- 'qwen-vl', so the per-session reasoner hot-switch can expose three real backends:
--   gemini      -> cloud (Vertex / AI Studio, google-genai)
--   qwen-onprem -> self-hosted Qwen text vLLM on H100/L4 (:8002) -- data sovereignty
--   qwen-vl     -> self-hosted multimodal Qwen3.6-VL via llama.cpp + mmproj (:8003)
--
-- This is a pure CHECK widening (rollforward, no data touched): the existing four
-- values stay valid and 'qwen-vl' is added. The concrete served-model id
-- ('qwen36-vl') and host URL live in env vars / settings (routing table), never in
-- this column, so adding a multimodal host is a CHECK + env edit, zero data change.
--
-- RLS note (US-051): this ALTER TABLE runs as the migration role `agrosat` (compose
-- superuser, implicit BYPASSRLS) -> it does not collide with the `tenant_isolation`
-- policy on chat_sessions. The runtime UPDATE (LLMSwitchService) instead runs scoped
-- as `agrosat_app` (NOBYPASSRLS) and is filtered to its own session by the policy.

-- Drop the 4-value CHECK and re-add it widened to include 'qwen-vl'.
ALTER TABLE chat_sessions DROP CONSTRAINT IF EXISTS chat_sessions_llm_model_check;

ALTER TABLE chat_sessions
    ADD CONSTRAINT chat_sessions_llm_model_check
    CHECK (llm_model IN ('gemini', 'qwen-api', 'qwen-onprem', 'gemma', 'qwen-vl'));

-- migrate:down
-- Exact reverse: collapse 'qwen-vl' back into the previous 4-value domain and
-- restore the original CHECK. Honest (lossy-by-necessity) down: any row already on
-- 'qwen-vl' has no representation under the old CHECK, so it folds to the always-
-- resolvable default 'gemini' before the narrower constraint is restored.

UPDATE chat_sessions SET llm_model = 'gemini' WHERE llm_model = 'qwen-vl';

ALTER TABLE chat_sessions DROP CONSTRAINT IF EXISTS chat_sessions_llm_model_check;

ALTER TABLE chat_sessions
    ADD CONSTRAINT chat_sessions_llm_model_check
    CHECK (llm_model IN ('gemini', 'qwen-api', 'qwen-onprem', 'gemma'));
