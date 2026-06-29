-- migrate:up
-- US-080: server-side chat history + named sessions for the multi-chat UI.
--
-- Adds (1) chat_sessions.title so the in-app chat switcher can name each chat,
-- and (2) chat_messages: one row per conversation turn, owned by its session
-- (multi-tenant via session_id, same RLS contract as aois/parcels). The chat
-- transcript is persisted here so a chat can be restored on reload (by id),
-- instead of living only in the browser.

-- Optional human-facing name for the chat tab (NULL -> the UI shows a default).
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS title TEXT;

CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    -- 'user' | 'assistant' | 'system' (the reasoner roles the frontend renders).
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    -- Optional structured payload (citations, perceiver observation, llm_model)
    -- kept as JSONB so the wire shape can evolve without a migration.
    extra JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Transcript is always read in insertion order, scoped to one session.
CREATE INDEX IF NOT EXISTS chat_messages_session_id_created_idx
    ON chat_messages (session_id, created_at);

-- Multi-tenant RLS (US-051 contract): a message is owned by its session via the
-- direct session_id column, identical to the aois/parcels policies.
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON chat_messages
    FOR ALL
    USING (session_id = current_setting('app.current_session', true)::uuid)
    WITH CHECK (session_id = current_setting('app.current_session', true)::uuid);

-- DML grants for the non-superuser application role + its BIGSERIAL sequence.
GRANT SELECT, INSERT, UPDATE, DELETE ON chat_messages TO agrosat_app;
GRANT USAGE, SELECT ON SEQUENCE chat_messages_id_seq TO agrosat_app;

-- migrate:down
DROP POLICY IF EXISTS tenant_isolation ON chat_messages;
ALTER TABLE chat_messages NO FORCE ROW LEVEL SECURITY;
ALTER TABLE chat_messages DISABLE ROW LEVEL SECURITY;
REVOKE SELECT, INSERT, UPDATE, DELETE ON chat_messages FROM agrosat_app;
DROP TABLE IF EXISTS chat_messages;
ALTER TABLE chat_sessions DROP COLUMN IF EXISTS title;
